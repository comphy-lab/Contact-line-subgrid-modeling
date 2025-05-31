#include <stdio.h>
#include <math.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>

#ifdef HAVE_GSL_BVP_H
#include <gsl/gsl_bvp.h>
#include <gsl/gsl_errno.h> // For gsl_strerror
#endif

#include "GLE_solver-GSL.h"

// GLE ODE system helper functions
double f1(double w) {
    return tanh(K_VALUE * w);
}

double f2(double w) {
    double sech_kw = 1.0 / cosh(K_VALUE * w);
    return K_VALUE * sech_kw * sech_kw;
}

double f3(double w) {
    double tanh_kw = tanh(K_VALUE * w);
    double sech_kw = 1.0 / cosh(K_VALUE * w);
    return -2.0 * K_VALUE * K_VALUE * tanh_kw * sech_kw * sech_kw;
}

void f_ode(double w, double* f1_val, double* f2_val, double* f3_val) {
    *f1_val = f1(w);
    *f2_val = f2(w);
    *f3_val = f3(w);
}

// GLE system definition for GSL BVP solver
int gle_system(double s, const double y[], double dyds[], void *params) {
    (void)params;

    double h = y[0];
    double theta = y[1];
    double w = y[2];

    double f1_w, f2_w, f3_w;
    f_ode(w, &f1_w, &f2_w, &f3_w);

    dyds[0] = tan(theta);
    // Guard against division by zero or very small h, or invalid theta
    if (h == 0.0 || cos(theta) == 0.0) {
        return 1; // GSL_FAILURE, indicates issue with inputs
    }
    dyds[1] = (LAMBDA_S_VALUE / (h * h)) * f1_w - (1.0 / (h * cos(theta)));
    dyds[2] = R_VALUE / h * cos(theta);

    return 0; // GSL_SUCCESS
}

#ifdef HAVE_GSL_BVP_H
// Boundary conditions for GSL BVP solver
void boundary_conditions(const gsl_vector * y_a, const gsl_vector * y_b, gsl_vector * resid, void *params) {
    (void)params; // Avoid unused parameter warning

    // y_a: h(s_min), theta(s_min), w(s_min)
    // y_b: h(s_max), theta(s_max), w(s_max)

    // resid[0]: theta(s_min) = 0
    gsl_vector_set(resid, 0, gsl_vector_get(y_a, 1));
    // resid[1]: theta(s_max) = 0
    gsl_vector_set(resid, 1, gsl_vector_get(y_b, 1));
    // resid[2]: w(s_min) = 1.0
    gsl_vector_set(resid, 2, gsl_vector_get(y_a, 2) - 1.0);
}

// GSL BVP solver function that also saves h(s) and theta(s) to CSV files
int solve_gle_bvp_and_save(gsl_vector *y_initial_mesh,
                           size_t num_components,
                           size_t num_nodes,
                           gsl_vector *sol_output_flat, // Flat vector for solution output
                           const char* h_output_path,
                           const char* theta_output_path) {

    gsl_bvp_workspace *bvp_ws = gsl_bvp_alloc(num_components, num_nodes);
    if (!bvp_ws) {
        fprintf(stderr, "Error allocating BVP workspace\n");
        return 1; // GSL_ENOMEM or similar
    }

    double s_min = 0.0;
    double s_max = 1.0; // Assuming normalized domain [0,1]

    // Initialize the BVP problem
    gsl_bvp_init(bvp_ws, gle_system, boundary_conditions, y_initial_mesh, s_min, s_max, NULL);

    // Check if sol_output_flat has the correct size
    if (gsl_vector_size(sol_output_flat) != num_components * num_nodes) {
        fprintf(stderr, "Solution output vector has incorrect size. Expected %zu, got %zu.\n",
                num_components * num_nodes, gsl_vector_size(sol_output_flat));
        gsl_bvp_free(bvp_ws);
        return 1; // GSL_EINVAL
    }

    // Solve the system
    int status = gsl_bvp_solve(bvp_ws, sol_output_flat);
    if (status != 0) { // GSL_SUCCESS is 0
        fprintf(stderr, "BVP solver failed to converge. GSL error: %s\n", gsl_strerror(status));
        gsl_bvp_free(bvp_ws);
        return status;
    }

    printf("BVP solver converged successfully!\n");

    // --- Save h(s) and theta(s) to CSV files ---
    FILE *h_file = fopen(h_output_path, "w");
    FILE *theta_file = fopen(theta_output_path, "w");

    if (!h_file) {
        fprintf(stderr, "Error opening file for h(s) output: %s\n", h_output_path);
        gsl_bvp_free(bvp_ws);
        return 1; // Indicate error
    }
    if (!theta_file) {
        fprintf(stderr, "Error opening file for theta(s) output: %s\n", theta_output_path);
        fclose(h_file); // Close already opened file
        gsl_bvp_free(bvp_ws);
        return 1; // Indicate error
    }

    // Write headers
    fprintf(h_file, "s,h\n");
    fprintf(theta_file, "s,theta\n");

    // Write data
    // The sol_output_flat vector stores y_i(x_j) = gsl_vector_get(sol_output_flat, j*num_components + i)
    // x_j = s_min + j * (s_max - s_min) / (num_nodes - 1) for node j
    for (size_t j = 0; j < num_nodes; ++j) {
        double s_j = s_min + j * (s_max - s_min) / (double)(num_nodes - 1);
        double h_j = gsl_vector_get(sol_output_flat, j * num_components + 0);    // h is the 0-th component
        double theta_j = gsl_vector_get(sol_output_flat, j * num_components + 1); // theta is the 1st component
        // double w_j = gsl_vector_get(sol_output_flat, j * num_components + 2); // w is the 2nd component (not saved here)

        fprintf(h_file, "%f,%f\n", s_j, h_j);
        fprintf(theta_file, "%f,%f\n", s_j, theta_j);
    }

    fclose(h_file);
    fclose(theta_file);
    printf("h(s) data saved to %s\n", h_output_path);
    printf("theta(s) data saved to %s\n", theta_output_path);

    gsl_bvp_free(bvp_ws);
    return 0; // GSL_SUCCESS
}

#else // Fallback if HAVE_GSL_BVP_H is not defined

// Provide a stub for solve_gle_bvp_and_save if GSL BVP is not available
int solve_gle_bvp_and_save(gsl_vector *y_initial_mesh,
                           size_t num_components,
                           size_t num_nodes,
                           gsl_vector *sol_output,
                           const char* h_output_path,
                           const char* theta_output_path) {
    (void)y_initial_mesh; (void)num_components; (void)num_nodes; (void)sol_output;
    (void)h_output_path; (void)theta_output_path;
    fprintf(stderr, "GSL BVP functionality is not compiled. Cannot solve or save data.\n");
    fprintf(stderr, "Please ensure GSL development libraries (including gsl_bvp.h) are installed and compile with HAVE_GSL_BVP_H defined.\n");
    return 1; // Indicate failure
}

void boundary_conditions(const gsl_vector * y_a, const gsl_vector * y_b, gsl_vector * resid, void *params) {
    (void)y_a; (void)y_b; (void)resid; (void)params;
    fprintf(stderr, "GSL BVP functionality is not compiled. Boundary conditions unavailable.\n");
}

#endif // HAVE_GSL_BVP_H

// Main function: Only compile if not building for tests (COMPILING_TESTS is not defined)
#ifndef COMPILING_TESTS
int main() {
    printf("GLE Solver GSL C version - main function.\n");

#ifdef HAVE_GSL_BVP_H
    const size_t n_nodes = 100;
    const size_t n_components = 3;
    const char* h_file_out = "output_h.csv";
    const char* theta_file_out = "output_theta.csv";

    gsl_vector *y_initial_guess = gsl_vector_alloc(n_components * n_nodes);
    gsl_vector *solution_output = gsl_vector_alloc(n_components * n_nodes);

    if(!y_initial_guess || !solution_output) {
        fprintf(stderr, "Error allocating GSL vectors in main.\n");
        if(y_initial_guess) gsl_vector_free(y_initial_guess);
        if(solution_output) gsl_vector_free(solution_output);
        return 1;
    }

    // Populate the initial guess vector (flat)
    for (size_t i = 0; i < n_nodes; ++i) {
        double s_i = (double)i / (n_nodes - 1);
        gsl_vector_set(y_initial_guess, i * n_components + 0, 1.0);      // Initial guess for h(s)
        gsl_vector_set(y_initial_guess, i * n_components + 1, 0.0);      // Initial guess for theta(s)
        gsl_vector_set(y_initial_guess, i * n_components + 2, 1.0 - s_i); // Initial guess for w(s)
    }
    // Refine guess based on known boundary conditions
    gsl_vector_set(y_initial_guess, 0 * n_components + 1, 0.0); // theta(s=0) = 0
    gsl_vector_set(y_initial_guess, (n_nodes - 1) * n_components + 1, 0.0); // theta(s=s_max) = 0
    gsl_vector_set(y_initial_guess, 0 * n_components + 2, 1.0); // w(s=0) = 1.0

    printf("Attempting to call BVP solver and save results...\n");
    int solver_status = solve_gle_bvp_and_save(y_initial_guess, n_components, n_nodes, solution_output, h_file_out, theta_file_out);

    if (solver_status == 0) { // GSL_SUCCESS
        printf("Main function: BVP solver successful. Data saved.\n");
    } else {
        printf("Main function: BVP solver failed or is disabled.\n");
    }

    gsl_vector_free(y_initial_guess);
    gsl_vector_free(solution_output);
    return (solver_status == 0) ? 0 : 1;

#else
    printf("BVP solver functionality is not compiled in this version (missing gsl_bvp.h or HAVE_GSL_BVP_H not defined).\n");
    return 0;
#endif
}
#endif // COMPILING_TESTS
