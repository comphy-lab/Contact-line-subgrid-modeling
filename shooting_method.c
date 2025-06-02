/**
 * shooting_method.c
 * 
 * Implementation of shooting method for solving boundary value problems
 * using SUNDIALS CVODE for forward integration
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "src-local/shooting_method.h"
#include "src-local/gradient_descent.h"
#include "src-local/gle_solver.h"
#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>

// Forward declaration of the ODE RHS function
extern int gle_ode_system(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data);

ShootingParams create_default_shooting_params(GLEParams* gle_params) {
  ShootingParams params;
  params.gle_params = gle_params;
  params.target_omega = 0.0;
  params.s_end = gle_params->s_end;  // Use the s_end from GLEParams
  params.tolerance = 1e-6;  // Less strict tolerance
  params.max_iterations = 100;
  params.epsilon = 1e-8;  // Smaller finite difference step
  params.verbose = false;
  return params;
}

double evaluate_shooting_function(double omega_0, ShootingParams* params, double* omega_end) {
  // Create a copy of GLE parameters
  GLEParams local_params = *(params->gle_params);
  
  // Set initial conditions with the guessed omega_0
  local_params.h_init = local_params.lambda_slip;
  local_params.theta_init = M_PI / 6.0;  // 30 degrees
  local_params.omega_init = omega_0;
  
  // Set integration limits
  local_params.s_start = 0.0;
  local_params.s_end = params->s_end;
  
  // Create SUNDIALS context if not already created
  if (local_params.sunctx == NULL) {
    if (gle_create_context(&local_params) != 0) {
      fprintf(stderr, "Failed to create SUNDIALS context\n");
      return 1e10;
    }
  }
  
  // Create initial condition vector
  N_Vector y0 = N_VNew_Serial(NEQ, local_params.sunctx);
  N_Vector y = N_VNew_Serial(NEQ, local_params.sunctx);
  
  // Set initial conditions
  NV_Ith_S(y0, 0) = local_params.h_init;
  NV_Ith_S(y0, 1) = local_params.theta_init;
  NV_Ith_S(y0, 2) = omega_0;
  
  // Initialize CVODE
  void *cvode_mem = CVodeCreate(CV_ADAMS, local_params.sunctx);
  if (cvode_mem == NULL) {
    N_VDestroy(y0);
    N_VDestroy(y);
    return 1e10;
  }
  
  // Initialize CVODE memory
  int flag = CVodeInit(cvode_mem, gle_ode_system, local_params.s_start, y0);
  if (flag != CV_SUCCESS) {
    CVodeFree(&cvode_mem);
    N_VDestroy(y0);
    N_VDestroy(y);
    return 1e10;
  }
  
  // Set user data
  CVodeSetUserData(cvode_mem, &local_params);
  
  // Set tolerances
  CVodeSStolerances(cvode_mem, 1e-8, 1e-10);
  
  // Create dense matrix and linear solver
  SUNMatrix A = SUNDenseMatrix(NEQ, NEQ, local_params.sunctx);
  SUNLinearSolver LS = SUNLinSol_Dense(y0, A, local_params.sunctx);
  CVodeSetLinearSolver(cvode_mem, LS, A);
  CVodeSetMaxNumSteps(cvode_mem, 10000);
  
  // Integrate to the end point
  double s_out = local_params.s_end;
  double s = local_params.s_start;
  flag = CVode(cvode_mem, s_out, y, &s, CV_NORMAL);
  
  double omega_at_end, residual;
  if (flag != CV_SUCCESS) {
    fprintf(stderr, "Warning: IVP integration failed for omega_0 = %e\n", omega_0);
    omega_at_end = 0.0;
    residual = 1e10;  // Return large penalty for failed integration
  } else {
    // Extract omega at the end point
    omega_at_end = NV_Ith_S(y, 2);
    residual = omega_at_end - params->target_omega;
  }
  
  if (omega_end != NULL) {
    *omega_end = omega_at_end;
  }
  
  // Clean up
  SUNMatDestroy(A);
  SUNLinSolFree(LS);
  CVodeFree(&cvode_mem);
  N_VDestroy(y0);
  N_VDestroy(y);
  
  return residual;
}

double compute_gradient(double omega_0, double epsilon, ShootingParams* params) {
  // Use central finite differences for better accuracy
  double f_plus = evaluate_shooting_function(omega_0 + epsilon, params, NULL);
  double f_minus = evaluate_shooting_function(omega_0 - epsilon, params, NULL);
  
  return (f_plus - f_minus) / (2.0 * epsilon);
}

ShootingResult solve_shooting_method(ShootingParams* params) {
  ShootingResult result;
  result.converged = false;
  result.iterations = 0;
  result.history_size = params->max_iterations;
  result.omega_0_history = (double*)malloc(params->max_iterations * sizeof(double));
  result.residual_history = (double*)malloc(params->max_iterations * sizeof(double));
  
  // Initialize gradient descent parameters
  GradientDescentParams gd_params = create_default_gd_params();
  gd_params.learning_rate = 0.001;  // Very conservative learning rate
  gd_params.use_adaptive = true;
  gd_params.momentum = 0.3;  // Gentle momentum
  gd_params.gradient_clip = 0.1;  // Very tight gradient clipping
  
  // Initial guess for omega_0
  double omega_0 = 0.01;  // Start with very small positive curvature
  double velocity = 0.0;  // For momentum
  double current_lr = gd_params.learning_rate;
  
  if (params->verbose) {
    printf("Starting shooting method:\n");
    printf("  Target omega at s=%g: %g\n", params->s_end, params->target_omega);
    printf("  Tolerance: %e\n", params->tolerance);
    printf("  Max iterations: %d\n", params->max_iterations);
    printf("\n");
  }
  
  // Main shooting iteration loop
  for (int iter = 0; iter < params->max_iterations; iter++) {
    // Evaluate shooting function
    double omega_end;
    double residual = evaluate_shooting_function(omega_0, params, &omega_end);
    
    // Store history
    result.omega_0_history[iter] = omega_0;
    result.residual_history[iter] = residual;
    result.iterations = iter + 1;
    
    if (params->verbose) {
      printf("Iteration %3d: omega_0 = %12.6e, omega_end = %12.6e, residual = %12.6e\n", 
             iter + 1, omega_0, omega_end, residual);
    }
    
    // Check convergence
    if (fabs(residual) < params->tolerance) {
      result.converged = true;
      result.omega_0_guess = omega_0;
      result.residual = residual;
      if (params->verbose) {
        printf("\nConverged! Final omega_0 = %e\n", omega_0);
      }
      break;
    }
    
    // Compute gradient
    double gradient = compute_gradient(omega_0, params->epsilon, params);
    
    // Clip gradient if needed
    gradient = clip_gradient(gradient, gd_params.gradient_clip);
    
    // Adaptive learning rate
    if (iter > 0 && gd_params.use_adaptive) {
      double prev_residual = result.residual_history[iter - 1];
      current_lr = adjust_learning_rate(current_lr, fabs(residual), 
                                       fabs(prev_residual), &gd_params);
    }
    
    // Update learning rate in gd_params
    gd_params.learning_rate = current_lr;
    
    // Gradient descent step
    omega_0 = gradient_descent_step(omega_0, gradient, &velocity, &gd_params);
    
    // Check for stall (optional early termination)
    if (iter > 10) {
      double recent_change = 0.0;
      for (int i = iter - 5; i < iter; i++) {
        recent_change += fabs(result.residual_history[i] - result.residual_history[i-1]);
      }
      if (recent_change < 1e-10) {
        if (params->verbose) {
          printf("\nWarning: Optimization stalled at iteration %d\n", iter + 1);
        }
        break;
      }
    }
  }
  
  // Set final values if not converged
  if (!result.converged) {
    result.omega_0_guess = omega_0;
    result.residual = result.residual_history[result.iterations - 1];
    if (params->verbose) {
      printf("\nWarning: Did not converge after %d iterations\n", result.iterations);
      printf("Final residual: %e\n", result.residual);
    }
  }
  
  return result;
}

void free_shooting_result(ShootingResult* result) {
  if (result->omega_0_history != NULL) {
    free(result->omega_0_history);
    result->omega_0_history = NULL;
  }
  if (result->residual_history != NULL) {
    free(result->residual_history);
    result->residual_history = NULL;
  }
}