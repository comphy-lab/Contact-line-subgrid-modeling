#ifndef GLE_MATH_H
#define GLE_MATH_H

#include <math.h>

/**
 * @file gle_math.h
 * @brief Mathematical functions for the Generalized Lubrication Equation (GLE)
 * 
 * This header defines the mathematical helper functions used in the GLE solver.
 * These functions are direct C translations of the Python implementation.
 */

/* Physical parameters - matching Python implementation */
#define CA_DEFAULT 1.0          /* Capillary number */
#define LAMBDA_SLIP_DEFAULT 1e-5 /* Slip length */
#define MU_R_DEFAULT 1e-3       /* Viscosity ratio (mu_g/mu_l) */
#define THETA0_DEFAULT (M_PI/6.0) /* Initial contact angle */
#define DELTA_DEFAULT 1e-4      /* Minimum grid cell size */

/**
 * @brief Helper function f1(theta) = theta^2 - sin^2(theta)
 * @param theta Contact angle in radians
 * @return Value of f1 function
 */
double f1(double theta);

/**
 * @brief Helper function f2(theta) = theta - sin(theta)*cos(theta)
 * @param theta Contact angle in radians
 * @return Value of f2 function
 */
double f2(double theta);

/**
 * @brief Helper function f3(theta) = theta*(pi-theta) + sin^2(theta)
 * @param theta Contact angle in radians
 * @return Value of f3 function
 */
double f3(double theta);

/**
 * @brief Combined function f(theta, mu_r) used in GLE formulation
 * @param theta Contact angle in radians
 * @param mu_r Viscosity ratio (gas/liquid)
 * @return Value of combined f function
 */
double f_combined(double theta, double mu_r);

/**
 * @brief Check if a value is finite and valid
 * @param value Value to check
 * @return 1 if finite, 0 otherwise
 */
int is_finite_value(double value);

#endif /* GLE_MATH_H */
