/**
 * GLE_solver-GSL.c
 *
 * Main program for the Generalized Lubrication Equations (GLE) solver
 * for modeling contact line dynamics in thin liquid films.
 *
 * SCIENTIFIC BACKGROUND:
 * =====================
 * This code solves the contact line problem where a liquid-gas interface meets
 * a solid substrate. The challenge is to bridge microscopic physics (molecular
 * scale) with macroscopic fluid dynamics. The GLE provide this bridge by
 * incorporating slip effects near the contact line.
 *
 * MATHEMATICAL FORMULATION:
 * ========================
 * The solver addresses the coupled system of ODEs:
 *   dh/ds = sin(θ)                    ... (1) Interface height evolution
 *   dθ/ds = ω                          ... (2) Interface angle evolution  
 *   dω/ds = 3Ca·f(θ,μᵣ)/(h(h+3λ)) - cos(θ) ... (3) Curvature evolution
 *
 * Where:
 * - s: Arc length coordinate along the interface
 * - h(s): Film thickness profile
 * - θ(s): Local interface angle with respect to substrate
 * - ω(s): Interface curvature (dθ/ds)
 * - Ca: Capillary number (viscous forces / surface tension)
 * - λ: Navier slip length (molecular scale parameter)
 * - μᵣ: Viscosity ratio (gas/liquid)
 * - f(θ,μᵣ): Complex function encoding viscous dissipation
 *
 * BOUNDARY CONDITIONS:
 * ===================
 * - At contact line (s=0): θ(0) = π/6 (30°), h(0) = λ
 * - At far field (s=s_max): ω(s_max) = 0 (matches outer solution)
 *
 * NUMERICAL METHOD:
 * ================
 * The code uses an Initial Value Problem (IVP) approach with shooting method:
 * 
 * 1. SHOOTING METHOD STRATEGY:
 *    - Guess initial curvature ω₀ = ω(0)
 *    - Integrate ODEs from s=0 to s=s_max
 *    - Check if ω(s_max) = 0 (boundary condition)
 *    - Adjust ω₀ until boundary condition is satisfied
 *
 * 2. INTEGRATION:
 *    - Uses GSL's adaptive Runge-Kutta-Fehlberg (4,5) method
 *    - Tight tolerances (1e-10 relative, 1e-12 absolute)
 *    - Adaptive step size for efficiency and accuracy
 *
 * 3. ROOT FINDING FOR ω₀:
 *    - Primary: Exponential search for bracketing + Brent's method
 *    - Fallback: Gradient descent with adaptive learning rate
 *    - Convergence criterion: |ω(s_max)| < 1e-8
 *
 * 4. PHYSICAL CONSTRAINTS:
 *    - h > 0 (positive film thickness)
 *    - 0 < θ < π (interface remains physical)
 *    - Singularity handling in f(θ,μᵣ) near θ = 0, π
 *
 * OUTPUT:
 * =======
 * The solver generates profiles h(s) and θ(s) that describe the interface
 * shape from molecular (contact line) to macroscopic scales.
 *
 * Date: 2025-05-31
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <gsl/gsl_errno.h>
#include "src-local/GLE_solver-GSL.h"

// Main function: Only compile if not building for tests
#ifndef COMPILING_TESTS
int main(int argc, char *argv[]) {
    printf("=== GLE Solver (C Implementation) ===\n");
    printf("Parameters:\n");
    printf("  Ca = %.2f\n", CA);
    printf("  lambda_slip = %.2e\n", LAMBDA_SLIP);
    printf("  mu_r = %.2e\n", MU_R);
    printf("  theta0 = %.2f degrees\n", THETA0 * 180.0 / M_PI);
    printf("  Domain: s ∈ [0, %.2e]\n", S_MAX);
    printf("\n");

    // Check for --quiet flag
    int verbose = 1;
    if (argc > 1 && strcmp(argv[1], "--quiet") == 0) {
        verbose = 0;
    }

    // Run solver
    size_t num_nodes = 1000;  // Number of output points
    int status = solve_gle_shooting_and_save(num_nodes, verbose);

    if (status == GSL_SUCCESS) {
        printf("\nSolution successful!\n");
        return 0;
    } else {
        printf("\nSolution failed with status: %d\n", status);
        return 1;
    }
}
#endif // COMPILING_TESTS