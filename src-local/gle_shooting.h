/**
 * gle_shooting.h
 * 
 * Shooting method implementation for the GLE solver
 * Contains the core shooting algorithm, integration routines, and optimization methods
 * 
 * Date: 2025-06-02
 */

#ifndef GLE_SHOOTING_H
#define GLE_SHOOTING_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_roots.h>
#include "GLE_solver-GSL.h"

/**
 * Function to integrate from s=0 to s=s_max with given initial omega
 * Returns the final state values
 */
static inline int integrate_ode(double omega0, shooting_context *ctx, double *h_final, double *theta_final, double *omega_final) {
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
 * Shooting residual function R(ω₀) = ω(s_max; ω₀)
 * 
 * For a given initial curvature ω₀, this integrates the ODEs and returns
 * the final curvature ω(s_max). The shooting method finds ω₀ such that R(ω₀) = 0,
 * satisfying the far-field boundary condition.
 */
static inline double shooting_residual_function(double omega0, void *params) {
    shooting_context *ctx = (shooting_context *)params;
    double h_final, theta_final, omega_final;

    int status = integrate_ode(omega0, ctx, &h_final, &theta_final, &omega_final);

    if (status != GSL_SUCCESS) {
        // Return signed sentinel value that preserves sign of omega0
        // This allows Brent's solver to detect sign changes correctly
        printf("Integration failed for omega0 = %g\n", omega0);
        return (omega0 > 0) ? 1e10 : -1e10;
    }

    // We want omega(s_max) = 0
    return omega_final;
}

/**
 * Exponential search for bracketing the root of R(ω₀)
 * 
 * ALGORITHM:
 * 1. Start with small bracket [ω₀ - w/2, ω₀ + w/2] around initial guess
 * 2. Evaluate R at both endpoints
 * 3. If R changes sign → found bracket
 * 4. Otherwise, double width w and repeat
 * 5. After a few iterations, try asymmetric expansion based on residuals
 * 
 * This is more robust than fixed bracketing for problems where the root
 * location is not well known a priori.
 */
static inline int find_omega0_bracket_exponential(shooting_context *ctx, double omega0_guess,
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
 * Gradient descent optimization for finding ω₀
 * 
 * Used as fallback when bracketing fails. This implements:
 * - Adaptive learning rate based on progress
 * - Line search to ensure descent
 * - Numerical gradient computation via finite differences
 * 
 * The method minimizes |R(ω₀)|² where R is the shooting residual.
 */
static inline double gradient_descent_omega0(shooting_context *ctx, double omega0_init) {
    double omega0 = omega0_init;
    double learning_rate = 1.0;  // Reduced initial learning rate for stability
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

        // Compute numerical gradient using central difference for better stability
        double f_plus = shooting_residual_function(omega0 + epsilon, ctx);
        double f_minus = shooting_residual_function(omega0 - epsilon, ctx);
        double gradient = (f_plus - f_minus) / (2.0 * epsilon);

        // Adaptive learning rate
        double step = -learning_rate * f0 / (fabs(gradient) + 1e-10);

        // Scale-aware step size limit based on omega0
        double max_step = 0.1 * fabs(omega0);
        if (max_step < 1.0) max_step = 1.0;  // Minimum step limit
        if (fabs(step) > max_step) {
            step = (step > 0) ? max_step : -max_step;
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
 * Main shooting method implementation
 * 
 * COMPLETE ALGORITHM:
 * 
 * 1. SETUP PHASE:
 *    - Initialize ODE system with physical parameters
 *    - Create GSL ODE driver with adaptive step control
 *    - Set initial conditions: h(0) = λ, θ(0) = π/6
 * 
 * 2. FIND ω₀ (SHOOTING PHASE):
 *    - Try exponential search to bracket root
 *    - If successful: Use Brent's method for root finding
 *    - If failed: Switch to gradient descent optimization
 *    - Continue until |ω(s_max)| < 1e-8
 * 
 * 3. SOLUTION GENERATION:
 *    - Integrate ODEs with converged ω₀
 *    - Store solution at 1000 points along s ∈ [0, s_max]
 *    - Output arrays: s, h(s), θ(s), ω(s)
 * 
 * 4. ERROR HANDLING:
 *    - Check for integration failures (unphysical solutions)
 *    - Verify memory allocation
 *    - Clean up resources on any failure
 */
static inline int solve_gle_shooting_method_with_bracket(gle_parameters *params, double s_max,
                                          double omega0_initial_guess,
                                          double initial_bracket_width,
                                          double max_bracket_width,
                                          double **s_out, double **h_out, double **theta_out, double **omega_out,
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
    *omega_out = (double *)malloc(*n_points * sizeof(double));

    if (!*s_out || !*h_out || !*theta_out || !*omega_out) {
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
    (*omega_out)[0] = omega0;

    for (int i = 1; i < *n_points; i++) {
        double s_target = (*s_out)[i];
        status = gsl_odeiv2_driver_apply(ctx.driver, &s, s_target, y);

        if (status != GSL_SUCCESS) {
            fprintf(stderr, "Integration failed at s = %g\n", s);
            // Free allocated memory before returning error
            free(*s_out);
            free(*h_out);
            free(*theta_out);
            free(*omega_out);
            *s_out = NULL;
            *h_out = NULL;
            *theta_out = NULL;
            *omega_out = NULL;
            *n_points = 0;
            gsl_root_fsolver_free(solver);
            gsl_odeiv2_driver_free(ctx.driver);
            return -1;
        }

        (*h_out)[i] = y[0];
        (*theta_out)[i] = y[1];
        (*omega_out)[i] = y[2];
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
static inline int solve_gle_shooting_method(gle_parameters *params, double s_max,
                             double **s_out, double **h_out, double **theta_out, double **omega_out,
                             int *n_points) {
    // Default parameters based on Python solution behavior
    double omega0_initial_guess = 82150.0;  // Near expected solution
    double initial_bracket_width = 10.0;    // Start with small bracket
    double max_bracket_width = 20000.0;     // Allow wide search if needed
    
    return solve_gle_shooting_method_with_bracket(params, s_max,
                                                 omega0_initial_guess,
                                                 initial_bracket_width,
                                                 max_bracket_width,
                                                 s_out, h_out, theta_out, omega_out,
                                                 n_points);
}

#endif // GLE_SHOOTING_H