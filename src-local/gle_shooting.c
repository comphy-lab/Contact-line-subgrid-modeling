/**
 * gle_shooting.c
 * 
 * Shooting method implementation for the GLE solver
 * Contains the core shooting algorithm and integration routines
 * 
 * Author: Vatsal Sanjay
 * Date: 2025-06-02
 */

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
 * Shooting residual function R(ω₀) = ω(s_max; ω₀)
 * 
 * For a given initial curvature ω₀, this integrates the ODEs and returns
 * the final curvature ω(s_max). The shooting method finds ω₀ such that R(ω₀) = 0,
 * satisfying the far-field boundary condition.
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
 *    - Output arrays: s, h(s), θ(s)
 * 
 * 4. ERROR HANDLING:
 *    - Check for integration failures (unphysical solutions)
 *    - Verify memory allocation
 *    - Clean up resources on any failure
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