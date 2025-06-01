/**
 * GLE_solver-GSL.h
 * 
 * Header file for the GLE solver implementation
 * that matches the Python implementation exactly.
 * 
 * Author: Claude
 * Date: 2025-05-31
 */

#ifndef GLE_SOLVER_GSL_H
#define GLE_SOLVER_GSL_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>
#include <gsl/gsl_odeiv2.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_errno.h>

// Physical parameters matching Python implementation
#define CA 1.0              // Capillary number
#define LAMBDA_SLIP 1e-5    // Slip length
#define MU_R 1e-3           // Viscosity ratio mu_g/mu_l
#define THETA0 (M_PI/6.0)   // Initial contact angle (30 degrees)
#define W_BOUNDARY 0.0      // Curvature at outer boundary
#define DELTA 1e-4          // Minimum grid cell size
#define S_MAX (4.0*DELTA)   // Maximum arc length

// Physical parameters structure for shooting method
typedef struct {
    double Ca;           // Capillary number
    double lambda_slip;  // Slip length
    double mu_r;         // Viscosity ratio
    double Delta;        // Grid cell size
} gle_parameters;

// Shooting method structures
typedef struct {
    gle_parameters *gle_params;
    double theta0;       // Initial contact angle
    double h0;           // Initial film thickness
    double s_max;        // Maximum arc length
    gsl_odeiv2_driver *driver; // ODE driver for integration
} shooting_context;

// Helper functions for f(theta, mu_r) calculation
double f1_trig(double theta);
double f2_trig(double theta);
double f3_trig(double theta);
double f_combined(double theta, double mu_r);

// ODE system
int gle_system(double s, const double y[], double dyds[], void *params);

// Boundary conditions (if GSL BVP is available)
#ifdef HAVE_GSL_BVP_H
void boundary_conditions(const gsl_vector *y_a, const gsl_vector *y_b, 
                        gsl_vector *resid, void *params);
#endif

// Main solver function
int solve_gle_bvp_and_save(size_t num_nodes, int verbose);

// ODE system for shooting method (matching Python)
int gle_ode_system_python(double s, const double y[], double dyds[], void *params);

// Shooting method solver
int solve_gle_shooting_method(gle_parameters *params, double s_max, 
                             double **s_out, double **h_out, double **theta_out, 
                             int *n_points);

// Shooting method solver with configurable initial omega0 bracket
int solve_gle_shooting_method_with_bracket(gle_parameters *params, double s_max,
                                          double omega0_initial_guess,
                                          double initial_bracket_width,
                                          double max_bracket_width,
                                          double **s_out, double **h_out, double **theta_out,
                                          int *n_points);

// Helper function for shooting residual
double shooting_residual_function(double omega0, void *params);

// Helper function to integrate ODE
int integrate_ode(double omega0, shooting_context *ctx, double *h_final, double *theta_final, double *omega_final);

// Gradient descent optimization for omega0
double gradient_descent_omega0(shooting_context *ctx, double omega0_init);

// Exponential search for finding omega0 bracket
int find_omega0_bracket_exponential(shooting_context *ctx, double omega0_guess,
                                   double initial_width, double max_width,
                                   double *omega0_low, double *omega0_high);

#endif // GLE_SOLVER_GSL_H