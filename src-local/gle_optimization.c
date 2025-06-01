/**
 * gle_optimization.c
 * 
 * Optimization algorithms for the GLE solver
 * Contains bracketing and gradient descent methods for finding ω₀
 * 
 * Author: Vatsal Sanjay
 * Date: 2025-06-02
 */

#include <stdio.h>
#include <math.h>
#include "GLE_solver-GSL.h"

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
 * Gradient descent optimization for finding ω₀
 * 
 * Used as fallback when bracketing fails. This implements:
 * - Adaptive learning rate based on progress
 * - Line search to ensure descent
 * - Numerical gradient computation via finite differences
 * 
 * The method minimizes |R(ω₀)|² where R is the shooting residual.
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