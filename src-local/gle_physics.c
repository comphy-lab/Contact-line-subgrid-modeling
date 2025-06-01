/**
 * gle_physics.c
 * 
 * Physical functions for the GLE solver
 * Contains the viscous dissipation function f(θ, μᵣ) and its components
 * 
 * Author: Vatsal Sanjay
 * Date: 2025-06-02
 */

#include <math.h>
#include "GLE_solver-GSL.h"

/**
 * Helper functions for computing f(θ, μᵣ) - the viscous dissipation function
 * These arise from the exact solution of Stokes flow near a moving contact line
 */
double f1_trig(double theta) {
    return theta * theta - sin(theta) * sin(theta);
}

double f2_trig(double theta) {
    return theta - sin(theta) * cos(theta);
}

double f3_trig(double theta) {
    return theta * (M_PI - theta) + sin(theta) * sin(theta);
}

/**
 * Computes f(θ, μᵣ) - the dimensionless function that encodes viscous dissipation
 * in the wedge flow near the contact line. This function comes from matching
 * the inner (wedge) and outer (lubrication) asymptotic solutions.
 * 
 * The function has singularities at θ = 0 and θ = π which are handled by clamping.
 */
double f_combined(double theta, double mu_r) {
    // Avoid exact boundaries where singularities occur
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
        // Near singularity - return large value based on sign of numerator
        return (numerator > 0) ? 1e10 : -1e10;
    }

    return numerator / denominator;
}