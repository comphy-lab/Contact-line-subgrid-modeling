#ifndef GLE_SOLVER_H
#define GLE_SOLVER_H

/**
 * # GLE Solver
 *
 * Provides functions to integrate the generalized lubrication equations.
 *
 * Features:
 * - Helper functions f1, f2, f3 implementing equation components.
 * - Combined function f_func for viscosity ratio dependence.
 * - RK4 integrator for the coupled ODE system.
 *
 * Author: Fluid Dynamics Team
 * Update History: Initial C implementation.
 */

/** Parameters for the solver */
typedef struct {
  double Ca;           /**< Capillary number */
  double lambda_slip;  /**< Slip length */
  double mu_r;         /**< Viscosity ratio */
} gle_params;

/** Compute f1(theta) = theta^2 - sin^2(theta) */
double f1(double theta);

/** Compute f2(theta) = theta - sin(theta)*cos(theta) */
double f2(double theta);

/** Compute f3(theta) = theta*(pi - theta) + sin^2(theta) */
double f3(double theta);

/**
 * Combined function appearing in the lubrication equations.
 *
 * f_func(theta, mu_r) =
 *   2*sin^3(theta) * (mu_r^2*f1(theta) + 2*mu_r*f3(theta)
 *                      + f1(pi - theta))
 *   / [3*(mu_r*f1(theta)*f2(pi - theta)
 *         - f1(pi - theta)*f2(theta))]
 */
double f_func(double theta, double mu_r);

/**
 * Evaluate the right-hand side of the ODE system.
 *
 * y[0] = h, y[1] = theta, y[2] = omega = dtheta/ds.
 * dyds returns derivatives of h, theta and omega.
 */
void gle_rhs(double s, const double y[3], double dyds[3],
             const gle_params *p);

/**
 * Integrate the GLE system using RK4 and save results to CSV.
 *
 * Returns 0 on success, non-zero on failure.
 */
int integrate_GLE(const gle_params *p, double s_start, double s_end,
                  int steps, double h0, double theta0, double omega0,
                  const char *csv_path);

#endif /* GLE_SOLVER_H */
