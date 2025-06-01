#include "src-local/gle_math.h"
#include <math.h>
#include <float.h>

/**
 * @file gle_math.c
 * @brief Implementation of mathematical functions for the GLE solver
 * 
 * Direct C translations of the Python mathematical functions from GLE_solver.py
 */

double f1(double theta) {
    return theta * theta - sin(theta) * sin(theta);
}

double f2(double theta) {
    return theta - sin(theta) * cos(theta);
}

double f3(double theta) {
    return theta * (M_PI - theta) + sin(theta) * sin(theta);
}

double f_combined(double theta, double mu_r) {
    /* Calculate numerator: 2 * sin^3(theta) * (mu_r^2 * f1(theta) + 2*mu_r * f3(theta) + f1(pi - theta)) */
    double sin_theta = sin(theta);
    double sin_cubed = sin_theta * sin_theta * sin_theta;
    
    double f1_theta = f1(theta);
    double f3_theta = f3(theta);
    double f1_pi_minus_theta = f1(M_PI - theta);
    
    double numerator = 2.0 * sin_cubed * (mu_r * mu_r * f1_theta + 2.0 * mu_r * f3_theta + f1_pi_minus_theta);
    
    /* Calculate denominator: 3 * (mu_r * f1(theta) * f2(pi - theta) - f1(pi - theta) * f2(theta)) */
    double f2_theta = f2(theta);
    double f2_pi_minus_theta = f2(M_PI - theta);
    
    double denominator = 3.0 * (mu_r * f1_theta * f2_pi_minus_theta - f1_pi_minus_theta * f2_theta);
    
    /* Check for division by zero */
    if (fabs(denominator) < DBL_EPSILON) {
        return NAN; /* Return NaN for undefined values */
    }
    
    return numerator / denominator;
}

int is_finite_value(double value) {
    return isfinite(value) && !isnan(value);
}
