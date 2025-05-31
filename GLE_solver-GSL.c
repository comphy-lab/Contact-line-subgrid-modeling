/**
 * GLE_solver-GSL.c
 * 
 * C implementation of the Generalized Lubrication Equations (GLE) solver
 * that matches the Python implementation exactly.
 * 
 * This solver solves the coupled ODEs for contact line dynamics:
 * - dh/ds = sin(theta)
 * - dtheta/ds = omega
 * - domega/ds = 3*Ca*f(theta, mu_r)/(h*(h + 3*lambda_slip)) - cos(theta)
 * 
 * Boundary conditions:
 * - theta(0) = pi/6
 * - h(0) = lambda_slip
 * - omega(s_max) = w (curvature at outer boundary)
 * 
 * Author: Claude
 * Date: 2025-05-31
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_roots.h>

#ifdef HAVE_GSL_BVP_H
#include <gsl/gsl_bvp.h>
#endif

#include "src-local/GLE_solver-GSL.h"

// Physical parameters matching Python implementation
#define CA 1.0              // Capillary number
#define LAMBDA_SLIP 1e-5    // Slip length
#define MU_R 1e-3           // Viscosity ratio mu_g/mu_l
#define THETA0 (M_PI/6.0)   // Initial contact angle (30 degrees)
#define W_BOUNDARY 0.0      // Curvature at outer boundary
#define DELTA 1e-4          // Minimum grid cell size
#define S_MAX (4.0*DELTA)   // Maximum arc length

// Helper functions for f(theta, mu_r) calculation
double f1_trig(double theta) {
    return theta * theta - sin(theta) * sin(theta);
}

double f2_trig(double theta) {
    return theta - sin(theta) * cos(theta);
}

double f3_trig(double theta) {
    return theta * (M_PI - theta) + sin(theta) * sin(theta);
}

// Main f function combining the helpers
double f_combined(double theta, double mu_r) {
    double f1_theta = f1_trig(theta);
    double f1_pi_minus_theta = f1_trig(M_PI - theta);
    double f2_theta = f2_trig(theta);
    double f2_pi_minus_theta = f2_trig(M_PI - theta);
    double f3_theta = f3_trig(theta);
    
    double numerator = 2.0 * pow(sin(theta), 3) * 
                      (mu_r * mu_r * f1_theta + 2.0 * mu_r * f3_theta + f1_pi_minus_theta);
    double denominator = 3.0 * (mu_r * f1_theta * f2_pi_minus_theta - f1_pi_minus_theta * f2_theta);
    
    // Guard against division by zero
    if (fabs(denominator) < 1e-15) {
        // For very small denominators, use L'Hôpital's rule or return a large value
        // to prevent the solver from exploring this region
        return 1e6;
    }
    
    return numerator / denominator;
}

// GLE ODE system matching Python implementation
int gle_system(double s, const double y[], double dyds[], void *params) {
    (void)params;  // Unused
    (void)s;       // s is not used in the equations
    
    double h = y[0];
    double theta = y[1];
    double omega = y[2];
    
    // dh/ds = sin(theta)
    dyds[0] = sin(theta);
    
    // dtheta/ds = omega
    dyds[1] = omega;
    
    // domega/ds = 3*Ca*f(theta, mu_r)/(h*(h + 3*lambda_slip)) - cos(theta)
    double f_val = f_combined(theta, MU_R);
    double h_factor = h * (h + 3.0 * LAMBDA_SLIP);
    
    // Guard against division by zero
    if (fabs(h_factor) < 1e-20) {
        return GSL_EBADFUNC;
    }
    
    dyds[2] = 3.0 * CA * f_val / h_factor - cos(theta);
    
    return GSL_SUCCESS;
}

#ifdef HAVE_GSL_BVP_H
// Boundary conditions matching Python implementation
void boundary_conditions(const gsl_vector *y_a, const gsl_vector *y_b, 
                        gsl_vector *resid, void *params) {
    (void)params;
    
    // y_a: values at s=0
    // y_b: values at s=s_max
    
    double h_a = gsl_vector_get(y_a, 0);
    double theta_a = gsl_vector_get(y_a, 1);
    double omega_b = gsl_vector_get(y_b, 2);
    
    // BC1: theta(0) = pi/6
    gsl_vector_set(resid, 0, theta_a - THETA0);
    
    // BC2: h(0) = lambda_slip
    gsl_vector_set(resid, 1, h_a - LAMBDA_SLIP);
    
    // BC3: omega(s_max) = w
    gsl_vector_set(resid, 2, omega_b - W_BOUNDARY);
}

// BVP solver with file output
int solve_gle_bvp_and_save(size_t num_nodes, const char *h_output_path, 
                          const char *theta_output_path, int verbose) {
    const size_t num_components = 3;
    
    // Allocate workspace
    gsl_bvp_workspace *bvp_ws = gsl_bvp_alloc(num_components, num_nodes);
    if (!bvp_ws) {
        fprintf(stderr, "Error allocating BVP workspace\n");
        return GSL_ENOMEM;
    }
    
    // Allocate vectors for initial guess and solution
    gsl_vector *y_initial = gsl_vector_alloc(num_components * num_nodes);
    gsl_vector *solution = gsl_vector_alloc(num_components * num_nodes);
    
    if (!y_initial || !solution) {
        fprintf(stderr, "Error allocating vectors\n");
        gsl_bvp_free(bvp_ws);
        if (y_initial) gsl_vector_free(y_initial);
        if (solution) gsl_vector_free(solution);
        return GSL_ENOMEM;
    }
    
    // Set up initial guess matching Python strategy
    for (size_t i = 0; i < num_nodes; ++i) {
        double s_i = i * S_MAX / (double)(num_nodes - 1);
        
        // Linear guess for h from lambda_slip to DELTA
        double h_guess = LAMBDA_SLIP + (DELTA - LAMBDA_SLIP) * s_i / S_MAX;
        gsl_vector_set(y_initial, i * num_components + 0, h_guess);
        
        // Constant guess for theta at pi/6
        gsl_vector_set(y_initial, i * num_components + 1, THETA0);
        
        // Constant guess for omega at 0
        gsl_vector_set(y_initial, i * num_components + 2, 0.0);
    }
    
    // Initialize BVP problem
    gsl_bvp_init(bvp_ws, gle_system, boundary_conditions, 
                 y_initial, 0.0, S_MAX, NULL);
    
    // Solve the system
    int status = gsl_bvp_solve(bvp_ws, solution);
    
    if (status != GSL_SUCCESS) {
        fprintf(stderr, "BVP solver failed: %s\n", gsl_strerror(status));
        gsl_bvp_free(bvp_ws);
        gsl_vector_free(y_initial);
        gsl_vector_free(solution);
        return status;
    }
    
    if (verbose) {
        printf("BVP solver converged successfully!\n");
        printf("Number of iterations: %zu\n", gsl_bvp_niter(bvp_ws));
    }
    
    // Save results to a single file
    FILE *data_file = fopen("output/data-c-gsl.csv", "w");
    
    if (!data_file) {
        fprintf(stderr, "Error opening output file\n");
        gsl_bvp_free(bvp_ws);
        gsl_vector_free(y_initial);
        gsl_vector_free(solution);
        return GSL_EFAILED;
    }
    
    // Write header
    fprintf(data_file, "s,h,theta\n");
    
    // Write data
    for (size_t j = 0; j < num_nodes; ++j) {
        double s_j = j * S_MAX / (double)(num_nodes - 1);
        double h_j = gsl_vector_get(solution, j * num_components + 0);
        double theta_j = gsl_vector_get(solution, j * num_components + 1);
        
        fprintf(data_file, "%.12e,%.12e,%.12e\n", s_j, h_j, theta_j);
    }
    
    fclose(data_file);
    
    if (verbose) {
        printf("Results saved to: output/data-c-gsl.csv\n");
    }
    
    // Keep the old file format for backward compatibility
    FILE *h_file = fopen(h_output_path, "w");
    FILE *theta_file = fopen(theta_output_path, "w");
    
    if (h_file && theta_file) {
        fprintf(h_file, "s,h\n");
        fprintf(theta_file, "s,theta_deg\n");
        
        for (size_t j = 0; j < num_nodes; ++j) {
            double s_j = j * S_MAX / (double)(num_nodes - 1);
            double h_j = gsl_vector_get(solution, j * num_components + 0);
            double theta_j = gsl_vector_get(solution, j * num_components + 1);
            double theta_deg = theta_j * 180.0 / M_PI;
            
            fprintf(h_file, "%.12e,%.12e\n", s_j, h_j);
            fprintf(theta_file, "%.12e,%.6f\n", s_j, theta_deg);
        }
        
        fclose(h_file);
        fclose(theta_file);
        
        if (verbose) {
            printf("  h(s): %s\n", h_output_path);
            printf("  theta(s): %s\n", theta_output_path);
        }
    }
    
    // Clean up
    gsl_bvp_free(bvp_ws);
    gsl_vector_free(y_initial);
    gsl_vector_free(solution);
    
    return GSL_SUCCESS;
}

#else
// Stub implementation when GSL BVP is not available
int solve_gle_bvp_and_save(size_t num_nodes, const char *h_output_path, 
                          const char *theta_output_path, int verbose) {
    (void)num_nodes;
    (void)h_output_path;
    (void)theta_output_path;
    (void)verbose;
    
    fprintf(stderr, "GSL BVP functionality not available.\n");
    fprintf(stderr, "Using shooting method as fallback...\n");
    
    // Fall back to shooting method
    gle_parameters params = {
        .Ca = CA,
        .lambda_slip = LAMBDA_SLIP,
        .mu_r = MU_R,
        .Delta = DELTA
    };
    
    double *s_out, *h_out, *theta_out;
    int n_points;
    
    int status = solve_gle_shooting_method(&params, S_MAX, &s_out, &h_out, &theta_out, &n_points);
    
    if (status == 0) {
        // Write results to single file
        FILE *data_file = fopen("output/data-c-gsl.csv", "w");
        
        if (data_file) {
            fprintf(data_file, "s,h,theta\n");
            
            for (int i = 0; i < n_points; i++) {
                fprintf(data_file, "%.12e,%.12e,%.12e\n", s_out[i], h_out[i], theta_out[i]);
            }
            
            fclose(data_file);
            
            if (verbose) {
                printf("Results saved to: output/data-c-gsl.csv (using shooting method)\n");
            }
        }
        
        // Also write to separate files for backward compatibility
        FILE *h_file = fopen(h_output_path, "w");
        FILE *theta_file = fopen(theta_output_path, "w");
        
        if (h_file && theta_file) {
            fprintf(h_file, "s,h\n");
            fprintf(theta_file, "s,theta_deg\n");
            
            for (int i = 0; i < n_points; i++) {
                fprintf(h_file, "%.12e,%.12e\n", s_out[i], h_out[i]);
                fprintf(theta_file, "%.12e,%.6f\n", s_out[i], theta_out[i] * 180.0 / M_PI);
            }
            
            fclose(h_file);
            fclose(theta_file);
        }
        
        free(s_out);
        free(h_out);
        free(theta_out);
        
        return GSL_SUCCESS;
    }
    
    return GSL_EUNIMPL;
}
#endif

// ============================================================================
// SHOOTING METHOD IMPLEMENTATION
// ============================================================================

/**
 * ODE system matching Python implementation for shooting method
 * y[0] = h, y[1] = theta, y[2] = omega (dtheta/ds)
 */
int gle_ode_system_python(double s, const double y[], double dyds[], void *params) {
    gle_parameters *p = (gle_parameters *)params;
    
    double h = y[0];
    double theta = y[1];
    double omega = y[2];
    
    // Avoid negative h
    if (h <= 0.0) {
        return GSL_EDOM;
    }
    
    // Avoid theta outside reasonable range
    // For contact line problems, theta should stay between 0 and pi
    if (theta < 1e-6 || theta > M_PI - 1e-6) {
        return GSL_EDOM;
    }
    
    // dh/ds = sin(theta)
    dyds[0] = sin(theta);
    
    // dtheta/ds = omega
    dyds[1] = omega;
    
    // domega/ds = 3*Ca*f(theta,mu_r)/(h*(h + 3*lambda_slip)) - cos(theta)
    double f_val = f_combined(theta, p->mu_r);
    double denominator = h * (h + 3.0 * p->lambda_slip);
    
    // Check for potential overflow
    if (denominator < 1e-15) {
        return GSL_EDOM;
    }
    
    // Check if f_val is finite
    if (!isfinite(f_val)) {
        return GSL_EDOM;
    }
    
    dyds[2] = 3.0 * p->Ca * f_val / denominator - cos(theta);
    
    return GSL_SUCCESS;
}

/**
 * Function to integrate from s=0 to s=s_max with given initial omega
 * Returns the final state values
 */
int integrate_ode(double omega0, shooting_context *ctx, double *h_final, double *theta_final, double *omega_final) {
    // Initial conditions
    double y[3] = {ctx->h0, ctx->theta0, omega0};
    double s = 0.0;
    
    // Reset the driver
    gsl_odeiv2_driver_reset(ctx->driver);
    
    // Try to integrate in smaller steps to detect where it fails
    int n_substeps = 10;
    double ds = ctx->s_max / n_substeps;
    
    for (int i = 0; i < n_substeps; i++) {
        double s_target = (i + 1) * ds;
        int status = gsl_odeiv2_driver_apply(ctx->driver, &s, s_target, y);
        
        if (status != GSL_SUCCESS) {
            return status;
        }
    }
    
    *h_final = y[0];
    *theta_final = y[1];
    *omega_final = y[2];
    
    return GSL_SUCCESS;
}

/**
 * Shooting residual function for root finding
 * We want omega(s_max) = 0
 */
double shooting_residual_function(double omega0, void *params) {
    shooting_context *ctx = (shooting_context *)params;
    double h_final, theta_final, omega_final;
    
    int status = integrate_ode(omega0, ctx, &h_final, &theta_final, &omega_final);
    
    if (status != GSL_SUCCESS) {
        // Return large residual on integration failure
        printf("Integration failed for omega0 = %g\n", omega0);
        return 1e10;
    }
    
    // We want omega(s_max) = 0
    // Debug print to see what's happening for values near expected solution
    if (fabs(omega0 - 82135.0) < 5000.0) {
        printf("omega0 = %.10f -> omega_final = %.10f, h_final = %.10e, theta_final = %.10f\n", 
               omega0, omega_final, h_final, theta_final);
    }
    
    return omega_final;
}

/**
 * Main shooting method solver
 */
int solve_gle_shooting_method(gle_parameters *params, double s_max, 
                             double **s_out, double **h_out, double **theta_out, 
                             int *n_points) {
    // Set up shooting context
    shooting_context ctx;
    ctx.gle_params = params;
    ctx.theta0 = M_PI / 6.0;  // 30 degrees
    ctx.h0 = params->lambda_slip;
    ctx.s_max = s_max;
    
    // Create ODE system
    gsl_odeiv2_system sys = {gle_ode_system_python, NULL, 3, params};
    
    // Create driver with adaptive step control - use tighter tolerances matching Python
    ctx.driver = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rkf45, 
                                               1e-12, 1e-10, 1e-8);
    
    if (!ctx.driver) {
        fprintf(stderr, "Failed to allocate ODE driver\n");
        return -1;
    }
    
    // Set up root finder for shooting method
    const gsl_root_fsolver_type *T = gsl_root_fsolver_brent;
    gsl_root_fsolver *solver = gsl_root_fsolver_alloc(T);
    
    if (!solver) {
        gsl_odeiv2_driver_free(ctx.driver);
        fprintf(stderr, "Failed to allocate root solver\n");
        return -1;
    }
    
    // Set up function for root finding
    gsl_function F;
    F.function = &shooting_residual_function;
    F.params = &ctx;
    
    // Initial bracket for omega0 - based on Python solution, omega0 should be approximately 82148.91
    double omega0_low = 82000.0;
    double omega0_high = 82300.0;
    
    // Find better initial bracket
    double f_low = shooting_residual_function(omega0_low, &ctx);
    double f_high = shooting_residual_function(omega0_high, &ctx);
    
    printf("Initial bracket test: f(%g) = %g, f(%g) = %g\n", omega0_low, f_low, omega0_high, f_high);
    
    if (f_low * f_high > 0) {
        // Try to find a bracket using a more systematic approach
        printf("Initial bracket doesn't contain root. Searching...\n");
        
        // Try different ranges around the expected value of ~82148.91
        double ranges[][2] = {{82100.0, 82200.0}, {82140.0, 82160.0}, {82145.0, 82155.0}, 
                              {82148.0, 82150.0}, {80000.0, 85000.0}, {70000.0, 90000.0}};
        int found = 0;
        
        for (int i = 0; i < 6; i++) {
            omega0_low = ranges[i][0];
            omega0_high = ranges[i][1];
            f_low = shooting_residual_function(omega0_low, &ctx);
            f_high = shooting_residual_function(omega0_high, &ctx);
            
            printf("Testing range [%g, %g]: f_low = %g, f_high = %g\n", omega0_low, omega0_high, f_low, f_high);
            
            if (f_low * f_high < 0) {
                printf("Found bracket: [%g, %g]\n", omega0_low, omega0_high);
                found = 1;
                break;
            }
        }
        
        if (!found) {
            fprintf(stderr, "Cannot find valid bracket for omega0\n");
            gsl_root_fsolver_free(solver);
            gsl_odeiv2_driver_free(ctx.driver);
            return -1;
        }
    }
    
    // Set up solver
    int status = gsl_root_fsolver_set(solver, &F, omega0_low, omega0_high);
    if (status != GSL_SUCCESS) {
        fprintf(stderr, "Failed to initialize root solver: %s\n", gsl_strerror(status));
        gsl_root_fsolver_free(solver);
        gsl_odeiv2_driver_free(ctx.driver);
        return -1;
    }
    
    // Iterate to find omega0
    int iter = 0;
    int max_iter = 100;
    double omega0;
    
    printf("Starting shooting method iterations...\n");
    
    do {
        iter++;
        status = gsl_root_fsolver_iterate(solver);
        omega0 = gsl_root_fsolver_root(solver);
        omega0_low = gsl_root_fsolver_x_lower(solver);
        omega0_high = gsl_root_fsolver_x_upper(solver);
        
        status = gsl_root_test_interval(omega0_low, omega0_high, 0, 1e-8);
        
        if (iter % 10 == 0 || status == GSL_SUCCESS) {
            printf("iter %d: omega0 = %.10f, interval = [%.10f, %.10f]\n", 
                   iter, omega0, omega0_low, omega0_high);
        }
    } while (status == GSL_CONTINUE && iter < max_iter);
    
    if (status != GSL_SUCCESS) {
        fprintf(stderr, "Root finding did not converge\n");
        gsl_root_fsolver_free(solver);
        gsl_odeiv2_driver_free(ctx.driver);
        return -1;
    }
    
    printf("Converged! omega0 = %.15f\n", omega0);
    
    // Now integrate with the converged omega0 to get the full solution
    *n_points = 1000;
    *s_out = (double *)malloc(*n_points * sizeof(double));
    *h_out = (double *)malloc(*n_points * sizeof(double));
    *theta_out = (double *)malloc(*n_points * sizeof(double));
    
    if (!*s_out || !*h_out || !*theta_out) {
        fprintf(stderr, "Memory allocation failed\n");
        gsl_root_fsolver_free(solver);
        gsl_odeiv2_driver_free(ctx.driver);
        return -1;
    }
    
    // Generate output points
    for (int i = 0; i < *n_points; i++) {
        (*s_out)[i] = i * s_max / (*n_points - 1);
    }
    
    // Reset driver and integrate again, storing intermediate values
    gsl_odeiv2_driver_reset(ctx.driver);
    double y[3] = {ctx.h0, ctx.theta0, omega0};
    double s = 0.0;
    
    (*h_out)[0] = ctx.h0;
    (*theta_out)[0] = ctx.theta0;
    
    for (int i = 1; i < *n_points; i++) {
        double s_target = (*s_out)[i];
        status = gsl_odeiv2_driver_apply(ctx.driver, &s, s_target, y);
        
        if (status != GSL_SUCCESS) {
            fprintf(stderr, "Integration failed at s = %g\n", s);
            break;
        }
        
        (*h_out)[i] = y[0];
        (*theta_out)[i] = y[1];
    }
    
    // Clean up
    gsl_root_fsolver_free(solver);
    gsl_odeiv2_driver_free(ctx.driver);
    
    return 0;
}

// Main function: Only compile if not building for tests (COMPILING_TESTS is not defined)
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
    
    // Check for --gui flag (for consistency with Python)
    int verbose = 1;
    if (argc > 1 && strcmp(argv[1], "--quiet") == 0) {
        verbose = 0;
    }
    
    // Create output directory if it doesn't exist
    system("mkdir -p output");
    
    // Run solver
    size_t num_nodes = 10000;  // Match Python's 10000 points
    int status = solve_gle_bvp_and_save(num_nodes, 
                                        "output/GLE_h_profile_c.csv",
                                        "output/GLE_theta_profile_c.csv",
                                        verbose);
    
    if (status == GSL_SUCCESS) {
        printf("\nSolution successful!\n");
        return 0;
    } else {
        printf("\nSolution failed with status: %d\n", status);
        return 1;
    }
}
#endif // COMPILING_TESTS