/**
 * gle_solver_wrapper.c
 * 
 * Wrapper functions for GLE solver that support both IVP and shooting methods
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "src-local/gle_solver.h"
#include "src-local/shooting_method.h"
#include <cvode/cvode.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>

// Forward declaration of the ODE RHS function
extern int gle_ode_system(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data);

GLESolution solve_gle(GLEParams *params) {
  GLESolution solution;
  solution.success = 0;
  solution.converged = 0;
  solution.iterations = 0;
  
  // Determine number of output points
  int n_points = 1000;
  double s_span = params->s_end - params->s_start;
  
  // Allocate solution arrays
  solution.n_points = n_points;
  solution.s_values = (double*)malloc(n_points * sizeof(double));
  solution.h_values = (double*)malloc(n_points * sizeof(double));
  solution.theta_values = (double*)malloc(n_points * sizeof(double));
  solution.omega = (double*)malloc(n_points * sizeof(double));
  
  if (!solution.s_values || !solution.h_values || 
      !solution.theta_values || !solution.omega) {
    fprintf(stderr, "Memory allocation failed in solve_gle\n");
    free_gle_solution(&solution);
    return solution;
  }
  
  // Create SUNDIALS context if not already created
  if (params->sunctx == NULL) {
    if (gle_create_context(params) != 0) {
      fprintf(stderr, "Failed to create SUNDIALS context\n");
      free_gle_solution(&solution);
      return solution;
    }
  }
  
  // Create initial condition vector
  N_Vector y0 = N_VNew_Serial(NEQ, params->sunctx);
  if (y0 == NULL) {
    fprintf(stderr, "Failed to create initial condition vector\n");
    free_gle_solution(&solution);
    return solution;
  }
  
  // Set initial conditions
  NV_Ith_S(y0, 0) = params->h_init;
  NV_Ith_S(y0, 1) = params->theta_init;
  NV_Ith_S(y0, 2) = params->omega_init;
  
  // Initialize CVODE
  void *cvode_mem = CVodeCreate(CV_ADAMS, params->sunctx);
  if (cvode_mem == NULL) {
    fprintf(stderr, "Failed to create CVODE solver\n");
    N_VDestroy(y0);
    free_gle_solution(&solution);
    return solution;
  }
  
  // Initialize CVODE memory
  int flag = CVodeInit(cvode_mem, gle_ode_system, params->s_start, y0);
  if (flag != CV_SUCCESS) {
    fprintf(stderr, "CVodeInit failed with flag %d\n", flag);
    CVodeFree(&cvode_mem);
    N_VDestroy(y0);
    free_gle_solution(&solution);
    return solution;
  }
  
  // Set user data
  flag = CVodeSetUserData(cvode_mem, params);
  if (flag != CV_SUCCESS) {
    fprintf(stderr, "CVodeSetUserData failed\n");
    CVodeFree(&cvode_mem);
    N_VDestroy(y0);
    free_gle_solution(&solution);
    return solution;
  }
  
  // Set tolerances
  double rtol = 1e-8;
  double atol = 1e-10;
  flag = CVodeSStolerances(cvode_mem, rtol, atol);
  if (flag != CV_SUCCESS) {
    fprintf(stderr, "CVodeSStolerances failed\n");
    CVodeFree(&cvode_mem);
    N_VDestroy(y0);
    free_gle_solution(&solution);
    return solution;
  }
  
  // Create dense matrix and linear solver
  SUNMatrix A = SUNDenseMatrix(NEQ, NEQ, params->sunctx);
  SUNLinearSolver LS = SUNLinSol_Dense(y0, A, params->sunctx);
  
  if (A == NULL || LS == NULL) {
    fprintf(stderr, "Failed to create linear solver\n");
    if (A) SUNMatDestroy(A);
    if (LS) SUNLinSolFree(LS);
    CVodeFree(&cvode_mem);
    N_VDestroy(y0);
    free_gle_solution(&solution);
    return solution;
  }
  
  // Attach linear solver
  flag = CVodeSetLinearSolver(cvode_mem, LS, A);
  if (flag != CV_SUCCESS) {
    fprintf(stderr, "CVodeSetLinearSolver failed\n");
    SUNMatDestroy(A);
    SUNLinSolFree(LS);
    CVodeFree(&cvode_mem);
    N_VDestroy(y0);
    free_gle_solution(&solution);
    return solution;
  }
  
  // Set maximum number of steps
  flag = CVodeSetMaxNumSteps(cvode_mem, 10000);
  
  // Integrate and store solution
  double s = params->s_start;
  double ds = s_span / (n_points - 1);
  N_Vector y = N_VClone(y0);
  N_VScale(1.0, y0, y);  // Copy initial conditions
  
  // Store initial point
  solution.s_values[0] = params->s_start;
  solution.h_values[0] = NV_Ith_S(y0, 0);
  solution.theta_values[0] = NV_Ith_S(y0, 1);
  solution.omega[0] = NV_Ith_S(y0, 2);
  
  for (int i = 1; i < n_points; i++) {
    double s_out = params->s_start + i * ds;
    
    flag = CVode(cvode_mem, s_out, y, &s, CV_NORMAL);
    
    if (flag != CV_SUCCESS) {
      fprintf(stderr, "CVode failed at s = %g with flag %d\n", s_out, flag);
      solution.n_points = i;  // Truncate solution
      break;
    }
    
    // Store solution
    solution.s_values[i] = s_out;
    solution.h_values[i] = NV_Ith_S(y, 0);
    solution.theta_values[i] = NV_Ith_S(y, 1);
    solution.omega[i] = NV_Ith_S(y, 2);
  }
  
  // Clean up
  N_VDestroy(y);
  SUNMatDestroy(A);
  SUNLinSolFree(LS);
  CVodeFree(&cvode_mem);
  N_VDestroy(y0);
  
  solution.success = (flag == CV_SUCCESS) ? 1 : 0;
  solution.converged = solution.success;
  
  return solution;
}

GLESolution solve_gle_shooting(GLEParams *params) {
  GLESolution solution;
  solution.success = 0;
  solution.converged = 0;
  solution.iterations = 0;
  
  // Create shooting parameters
  ShootingParams shooting_params = create_default_shooting_params(params);
  shooting_params.verbose = false;  // Disable verbose output for cleaner results
  
  // Solve using shooting method
  ShootingResult result = solve_shooting_method(&shooting_params);
  
  if (result.converged) {
    // Now integrate with the found omega_0 to get the full solution
    params->omega_init = result.omega_0_guess;
    solution = solve_gle(params);
    solution.iterations = result.iterations;
    
    printf("\nShooting method converged!\n");
    printf("Final omega(0) = %e\n", result.omega_0_guess);
    printf("Final residual = %e\n", result.residual);
    printf("Iterations = %d\n", result.iterations);
  } else {
    // Return empty solution if shooting failed
    solution.n_points = 0;
    solution.s_values = NULL;
    solution.h_values = NULL;
    solution.theta_values = NULL;
    solution.omega = NULL;
    
    fprintf(stderr, "Shooting method failed to converge\n");
  }
  
  // Free shooting result
  free_shooting_result(&result);
  
  return solution;
}

void free_gle_solution(GLESolution *solution) {
  if (solution->s_values) {
    free(solution->s_values);
    solution->s_values = NULL;
  }
  if (solution->h_values) {
    free(solution->h_values);
    solution->h_values = NULL;
  }
  if (solution->theta_values) {
    free(solution->theta_values);
    solution->theta_values = NULL;
  }
  if (solution->omega) {
    free(solution->omega);
    solution->omega = NULL;
  }
  solution->n_points = 0;
}