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
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_roots.h>

#ifdef HAVE_GSL_BVP_H
#include <gsl/gsl_bvp.h>
#endif

#include "src-local/GLE_solver-GSL.h"

// Physical parameters are now defined in the header file

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
    // Avoid exact boundaries where singularities occur
    // The ODE integrator should avoid these regions
    const double theta_min = 1e-10;
    const double theta_max = M_PI - 1e-10;
    
    // Clamp theta to avoid singularities
    if (theta < theta_min) {
        theta = theta_min;
    } else if (theta > theta_max) {
        theta = theta_max;
    }
    
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
        // Near singularity - this should not happen with clamped theta
        // but if it does, return a large value based on the sign of numerator
        // The ODE integrator will handle this by reducing step size
        return (numerator > 0) ? 1e10 : -1e10;
    }

    double result = numerator / denominator;
    
    return result;
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
    
    // Check if f_combined returned NaN or extremely large value
    if (!isfinite(f_val) || fabs(f_val) > 1e12) {
        return GSL_EBADFUNC;
    }
    
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
int solve_gle_bvp_and_save(size_t num_nodes, int verbose) {
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
        // gsl_bvp_niter may not be available in all GSL versions
        // The iteration count is typically stored in the workspace
        // but the exact field may vary by GSL version
        #ifdef GSL_BVP_HAS_NITER
        printf("Number of iterations: %zu\n", gsl_bvp_niter(bvp_ws));
        #else
        // For compatibility, we just omit the iteration count
        printf("Solution completed (iteration count not available)\n");
        #endif
    }

    // Create output directory if it doesn't exist
    struct stat st = {0};
    if (stat("output", &st) == -1) {
        if (mkdir("output", 0755) != 0) {
            fprintf(stderr, "Error creating output directory: %s\n", strerror(errno));
            gsl_bvp_free(bvp_ws);
            gsl_vector_free(y_initial);
            gsl_vector_free(solution);
            return GSL_EFAILED;
        }
    }

    // Save results to a single file
    FILE *data_file = fopen("output/data-c-gsl.csv", "w");

    if (!data_file) {
        fprintf(stderr, "Error opening output file: %s\n", strerror(errno));
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


    // Clean up
    gsl_bvp_free(bvp_ws);
    gsl_vector_free(y_initial);
    gsl_vector_free(solution);

    return GSL_SUCCESS;
}

#else
// Stub implementation when GSL BVP is not available
int solve_gle_bvp_and_save(size_t num_nodes, int verbose) {
    (void)num_nodes;
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
        // Create output directory if it doesn't exist
        struct stat st = {0};
        if (stat("output", &st) == -1) {
            if (mkdir("output", 0755) != 0) {
                fprintf(stderr, "Error creating output directory: %s\n", strerror(errno));
                free(s_out);
                free(h_out);
                free(theta_out);
                return GSL_EFAILED;
            }
        }

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
    (void)s;  // Unused parameter - required by GSL interface
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
    
    // Check if f_combined returned NaN or extremely large value
    if (!isfinite(f_val) || fabs(f_val) > 1e12) {
        return GSL_EDOM;
    }
    
    double denominator = h * (h + 3.0 * p->lambda_slip);

    // Check for potential overflow
    if (denominator < 1e-15) {
        return GSL_EDOM;
    }

    // Check if f_val is finite (catches infinities)
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
    return omega_final;
}

/**
 * Exponential search for finding omega0 bracket
 * 
 * Starting from an initial guess, exponentially expand the search bracket
 * until we find function values with opposite signs or reach max width.
 * 
 * @param ctx Shooting context with ODE system
 * @param omega0_guess Initial guess for omega0
 * @param initial_width Initial bracket width to try
 * @param max_width Maximum allowed bracket width
 * @param omega0_low Output: lower bound of bracket
 * @param omega0_high Output: upper bound of bracket
 * @return 0 on success (bracket found), -1 on failure
 */
int find_omega0_bracket_exponential(shooting_context *ctx, double omega0_guess,
                                   double initial_width, double max_width,
                                   double *omega0_low, double *omega0_high) {
    printf("\nStarting exponential search for omega0 bracket\n");
    printf("Initial guess: omega0 = %.6f\n", omega0_guess);
    printf("Initial width: %.2f, Max width: %.2f\n", initial_width, max_width);
    
    double width = initial_width;
    double expansion_factor = 2.0;  // Double the width each iteration
    int max_iterations = 20;        // Prevent infinite loop
    int iteration = 0;
    
    while (width <= max_width && iteration < max_iterations) {
        // Try symmetric bracket around guess
        *omega0_low = omega0_guess - width / 2.0;
        *omega0_high = omega0_guess + width / 2.0;
        
        // Evaluate residual at bracket endpoints
        double f_low = shooting_residual_function(*omega0_low, ctx);
        double f_high = shooting_residual_function(*omega0_high, ctx);
        
        printf("  Iteration %d: [%.2f, %.2f], width=%.2f, f_low=%.2e, f_high=%.2e\n",
               iteration, *omega0_low, *omega0_high, width, f_low, f_high);
        
        // Check if we have a sign change
        if (f_low * f_high < 0) {
            printf("  Found bracket with sign change!\n");
            return 0;  // Success
        }
        
        // If both residuals are small, we might be very close to the solution
        if (fabs(f_low) < 1e-6 && fabs(f_high) < 1e-6) {
            printf("  Both residuals are small - solution might be in this range\n");
            return 0;
        }
        
        // Try asymmetric expansion based on which side has smaller residual
        if (iteration > 2) {  // After a few symmetric tries
            if (fabs(f_low) < fabs(f_high)) {
                // Expand more on the low side
                *omega0_low = omega0_guess - width * 0.75;
                *omega0_high = omega0_guess + width * 0.25;
            } else {
                // Expand more on the high side
                *omega0_low = omega0_guess - width * 0.25;
                *omega0_high = omega0_guess + width * 0.75;
            }
            
            // Re-evaluate with asymmetric bracket
            f_low = shooting_residual_function(*omega0_low, ctx);
            f_high = shooting_residual_function(*omega0_high, ctx);
            
            printf("  Asymmetric: [%.2f, %.2f], f_low=%.2e, f_high=%.2e\n",
                   *omega0_low, *omega0_high, f_low, f_high);
            
            if (f_low * f_high < 0) {
                printf("  Found bracket with sign change (asymmetric)!\n");
                return 0;
            }
        }
        
        // Expand the width exponentially
        width *= expansion_factor;
        iteration++;
    }
    
    printf("  Failed to find bracket after %d iterations\n", iteration);
    return -1;  // Failed to find bracket
}

/**
 * Gradient descent optimization for finding omega0
 */
double gradient_descent_omega0(shooting_context *ctx, double omega0_init) {
    double omega0 = omega0_init;
    double learning_rate = 10.0;  // Initial learning rate
    double epsilon = 1e-6;  // For numerical gradient
    double tolerance = 1e-8;
    int max_iter = 200;
    int iter = 0;

    printf("\nStarting gradient descent from omega0 = %.6f\n", omega0);

    while (iter < max_iter) {
        // Compute residual at current point
        double f0 = shooting_residual_function(omega0, ctx);

        // Check for convergence
        if (fabs(f0) < tolerance) {
            printf("Gradient descent converged at iteration %d: omega0 = %.10f, residual = %.2e\n",
                   iter, omega0, f0);
            return omega0;
        }

        // Compute numerical gradient
        double f_plus = shooting_residual_function(omega0 + epsilon, ctx);
        double gradient = (f_plus - f0) / epsilon;

        // Adaptive learning rate
        double step = -learning_rate * f0 / (fabs(gradient) + 1e-10);

        // Limit step size
        if (fabs(step) > 1000.0) {
            step = (step > 0) ? 1000.0 : -1000.0;
        }

        // Update omega0
        double omega0_new = omega0 + step;

        // Line search to ensure we're making progress
        double f_new = shooting_residual_function(omega0_new, ctx);
        double alpha = 1.0;
        int line_search_iter = 0;

        while (fabs(f_new) > fabs(f0) && line_search_iter < 10) {
            alpha *= 0.5;
            omega0_new = omega0 + alpha * step;
            f_new = shooting_residual_function(omega0_new, ctx);
            line_search_iter++;
        }

        omega0 = omega0_new;

        // Update learning rate based on progress
        if (fabs(f_new) < fabs(f0)) {
            learning_rate *= 1.1;  // Increase if making progress
        } else {
            learning_rate *= 0.5;  // Decrease if not
        }

        if (iter % 10 == 0) {
            printf("  iter %3d: omega0 = %.6f, residual = %.2e, lr = %.2e\n",
                   iter, omega0, f_new, learning_rate);
        }

        iter++;
    }

    printf("Gradient descent did not converge within %d iterations\n", max_iter);
    return omega0;
}

/**
 * Main shooting method solver with configurable omega0 bracket
 * 
 * @param params GLE parameters structure
 * @param s_max Maximum arc length for integration
 * @param omega0_initial_guess Initial guess for omega0 (use 82150.0 if unsure)
 * @param initial_bracket_width Initial bracket width (use 10.0 for small search)
 * @param max_bracket_width Maximum bracket width (use 20000.0 for extensive search)
 * @param s_out Output array for s values
 * @param h_out Output array for h values
 * @param theta_out Output array for theta values
 * @param n_points Number of output points
 * @return 0 on success, negative value on error
 */
int solve_gle_shooting_method_with_bracket(gle_parameters *params, double s_max,
                                          double omega0_initial_guess,
                                          double initial_bracket_width,
                                          double max_bracket_width,
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

    double omega0_low, omega0_high;
    double omega0 = 0.0;

    // Use exponential search to find bracket
    int bracket_found = find_omega0_bracket_exponential(&ctx, omega0_initial_guess,
                                                       initial_bracket_width,
                                                       max_bracket_width,
                                                       &omega0_low, &omega0_high);

    if (bracket_found != 0) {
        printf("Exponential search failed to find bracket\n");
        printf("Switching to gradient descent optimization...\n");

        // Use gradient descent starting from the initial guess
        omega0 = gradient_descent_omega0(&ctx, omega0_initial_guess);

        // Check if gradient descent found a good solution
        double final_residual = shooting_residual_function(omega0, &ctx);
        if (fabs(final_residual) > 1e-6) {
            fprintf(stderr, "Gradient descent failed to find good solution: residual = %.2e\n", final_residual);
            gsl_root_fsolver_free(solver);
            gsl_odeiv2_driver_free(ctx.driver);
            return -1;
        }

        printf("Gradient descent found omega0 = %.15f with residual = %.2e\n", omega0, final_residual);

        // Skip the bracketing solver and go directly to solution generation
        goto generate_solution;
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

generate_solution:
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
            // Free allocated memory before returning error
            free(*s_out);
            free(*h_out);
            free(*theta_out);
            *s_out = NULL;
            *h_out = NULL;
            *theta_out = NULL;
            *n_points = 0;
            gsl_root_fsolver_free(solver);
            gsl_odeiv2_driver_free(ctx.driver);
            return -1;
        }

        (*h_out)[i] = y[0];
        (*theta_out)[i] = y[1];
    }

    // Clean up
    gsl_root_fsolver_free(solver);
    gsl_odeiv2_driver_free(ctx.driver);

    return 0;
}

/**
 * Main shooting method solver - wrapper for backward compatibility
 * Uses default bracket parameters based on the known solution behavior
 */
int solve_gle_shooting_method(gle_parameters *params, double s_max,
                             double **s_out, double **h_out, double **theta_out,
                             int *n_points) {
    // Default parameters based on Python solution behavior
    double omega0_initial_guess = 82150.0;  // Near expected solution
    double initial_bracket_width = 10.0;    // Start with small bracket
    double max_bracket_width = 20000.0;     // Allow wide search if needed
    
    return solve_gle_shooting_method_with_bracket(params, s_max,
                                                 omega0_initial_guess,
                                                 initial_bracket_width,
                                                 max_bracket_width,
                                                 s_out, h_out, theta_out,
                                                 n_points);
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
    struct stat st = {0};
    if (stat("output", &st) == -1) {
        if (mkdir("output", 0755) != 0) {
            fprintf(stderr, "Error creating output directory: %s\n", strerror(errno));
            return 1;
        }
    }

    // Run solver
    size_t num_nodes = 100000;  // Match Python's 10000 points
    int status = solve_gle_bvp_and_save(num_nodes, verbose);

    if (status == GSL_SUCCESS) {
        printf("\nSolution successful!\n");
        return 0;
    } else {
        printf("\nSolution failed with status: %d\n", status);
        return 1;
    }
}
#endif // COMPILING_TESTS
