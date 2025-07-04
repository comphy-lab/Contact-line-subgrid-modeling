/**
 * gle_ode_systems.h
 * 
 * ODE systems for the GLE solver
 * Contains the differential equation systems used for integration
 * 
 * Date: 2025-06-02
 */

#ifndef GLE_ODE_SYSTEMS_H
#define GLE_ODE_SYSTEMS_H

#include <math.h>
#include <gsl/gsl_errno.h>
#include "GLE_solver-GSL.h"

/**
 * GLE ODE system for the coupled evolution equations
 * This is used by the main solver for output generation
 * 
 * State vector: y = [h, θ, ω]
 * Derivatives: dyds = [dh/ds, dθ/ds, dω/ds]
 */
static inline int gle_system(double s, const double y[], double dyds[], void *params) {
    (void)params;  // Unused
    (void)s;       // s is not used in the equations

    double h = y[0];
    double theta = y[1];
    double omega = y[2];

    // Validate physical constraints early
    if (h <= 0.0) {
        // Negative or zero film thickness is physically invalid
        return GSL_EBADFUNC;
    }
    
    if (theta <= 0.0 || theta >= M_PI) {
        // Contact angle must be between 0 and π (exclusive)
        return GSL_EBADFUNC;
    }
    
    if (!isfinite(h) || !isfinite(theta) || !isfinite(omega)) {
        // Check for NaN or infinite values in state
        return GSL_EBADFUNC;
    }

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

/**
 * Wrapper for gle_system to ensure visibility in test files
 * This function provides an alternative name for testing
 */
static inline int gle_system_wrapper(double s, const double y[], double dyds[], void *params) {
    return gle_system(s, y, dyds, params);
}

/**
 * ODE system for the shooting method integration
 * Includes additional checks for physical validity during integration
 * 
 * State: y = [h, θ, ω] where ω = dθ/ds (curvature)
 * Returns GSL_EDOM if solution becomes unphysical
 */
static inline int gle_ode_system_python(double s, const double y[], double dyds[], void *params) {
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

    dyds[2] = 3.0 * p->Ca * f_val / denominator - cos(theta);

    return GSL_SUCCESS;
}

#endif // GLE_ODE_SYSTEMS_H