/**
 * shooting_method.h
 * 
 * Shooting method interface for solving boundary value problems
 * using SUNDIALS IVP solver (CVODE)
 * 
 * This module implements a shooting method with gradient descent
 * to find the correct initial condition omega(0) such that the
 * boundary condition omega(s_end) = target is satisfied.
 */

#ifndef SHOOTING_METHOD_H
#define SHOOTING_METHOD_H

#include <stdbool.h>
#include "gle_solver.h"

typedef struct {
  GLEParams* gle_params;      // GLE parameters
  double target_omega;        // Target value of omega at s_end (usually 0)
  double s_end;              // End point of integration
  double tolerance;          // Convergence tolerance
  int max_iterations;        // Maximum shooting iterations
  double epsilon;            // Finite difference step for gradient
  bool verbose;              // Print iteration information
} ShootingParams;

typedef struct {
  double omega_0_guess;      // Final converged guess for omega(0)
  double residual;           // F(omega_0) = omega(s_end) - target
  int iterations;            // Number of iterations performed
  bool converged;            // Convergence status
  double* omega_0_history;   // History of omega_0 guesses
  double* residual_history;  // History of residuals
  int history_size;          // Size of history arrays
} ShootingResult;

/**
 * Solve the boundary value problem using shooting method
 * 
 * @param params Shooting method parameters
 * @return ShootingResult containing solution and convergence information
 */
ShootingResult solve_shooting_method(ShootingParams* params);

/**
 * Evaluate the shooting function F(omega_0) = omega(s_end) - target
 * 
 * @param omega_0 Initial guess for omega at s=0
 * @param params Shooting method parameters
 * @param omega_end Output: value of omega at s_end (optional)
 * @return Residual F(omega_0)
 */
double evaluate_shooting_function(double omega_0, ShootingParams* params, double* omega_end);

/**
 * Compute gradient of shooting function using finite differences
 * 
 * @param omega_0 Current guess for omega(0)
 * @param epsilon Finite difference step size
 * @param params Shooting method parameters
 * @return Gradient dF/d(omega_0)
 */
double compute_gradient(double omega_0, double epsilon, ShootingParams* params);

/**
 * Free memory allocated for ShootingResult
 * 
 * @param result Pointer to ShootingResult to free
 */
void free_shooting_result(ShootingResult* result);

/**
 * Create default shooting parameters
 * 
 * @param gle_params Pointer to GLE parameters
 * @return ShootingParams with default values
 */
ShootingParams create_default_shooting_params(GLEParams* gle_params);

#endif // SHOOTING_METHOD_H