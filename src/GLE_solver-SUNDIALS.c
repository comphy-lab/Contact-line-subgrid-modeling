#include "gle_solver.h"
#include "gle_math.h"
#include <cvode/cvode.h>
#include <kinsol/kinsol.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
/* Dense matrix operations now in sunmatrix_dense.h */
#include <sundials/sundials_types.h>
#include <nvector/nvector_serial.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* For SUNDIALS built with MPI support */
#ifdef SUNDIALS_MPI_ENABLED
#include <mpi.h>
#endif

/**
 * @file GLE_solver-SUNDIALS.c
 * @brief SUNDIALS-based implementation of the GLE boundary value problem solver
 * 
 * This file implements the multiple shooting approach using CVODE for initial
 * value problems and KINSOL for boundary condition enforcement.
 */

void gle_params_init(GLEParams *params) {
    params->Ca = CA_DEFAULT;
    params->lambda_slip = LAMBDA_SLIP_DEFAULT;
    params->mu_r = MU_R_DEFAULT;
    params->theta0 = THETA0_DEFAULT;
    params->Delta = DELTA_DEFAULT;
    params->w = 0.0; /* Curvature boundary condition from DNS */
    params->sunctx = NULL; /* Will be initialized separately */
}

int gle_create_context(GLEParams *params) {
#ifdef SUNDIALS_MPI_ENABLED
    /* Initialize MPI if not already initialized */
    int mpi_initialized;
    MPI_Initialized(&mpi_initialized);
    if (!mpi_initialized) {
        int argc = 0;
        char **argv = NULL;
        MPI_Init(&argc, &argv);
    }
    int flag = SUNContext_Create(MPI_COMM_SELF, &(params->sunctx));
#else
    int flag = SUNContext_Create(NULL, &(params->sunctx));
#endif
    if (flag != 0) {
        printf("Error: Failed to create SUNDIALS context\n");
        return -1;
    }
    return 0;
}

void gle_destroy_context(GLEParams *params) {
    if (params->sunctx) {
        SUNContext_Free(&(params->sunctx));
        params->sunctx = NULL;
    }
}

GLESolution* gle_solution_alloc(int n_points) {
    GLESolution *solution = (GLESolution*)malloc(sizeof(GLESolution));
    if (!solution) return NULL;
    
    solution->s_values = (double*)malloc(n_points * sizeof(double));
    solution->h_values = (double*)malloc(n_points * sizeof(double));
    solution->theta_values = (double*)malloc(n_points * sizeof(double));
    solution->w_values = (double*)malloc(n_points * sizeof(double));
    
    if (!solution->s_values || !solution->h_values || 
        !solution->theta_values || !solution->w_values) {
        gle_solution_free(solution);
        return NULL;
    }
    
    solution->n_points = n_points;
    solution->converged = 0;
    solution->iterations = 0;
    
    return solution;
}

void gle_solution_free(GLESolution *solution) {
    if (!solution) return;
    
    if (solution->s_values) free(solution->s_values);
    if (solution->h_values) free(solution->h_values);
    if (solution->theta_values) free(solution->theta_values);
    if (solution->w_values) free(solution->w_values);
    
    free(solution);
}

int gle_ode_system(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data) {
    (void)t; /* Suppress unused parameter warning */
    GLEParams *params = (GLEParams*)user_data;
    
    /* Extract state variables: y = [h, theta, omega] */
    double h = NV_Ith_S(y, 0);
    double theta = NV_Ith_S(y, 1);
    double omega = NV_Ith_S(y, 2);
    
    /* Check for valid state */
    if (h <= 0 || theta <= 0 || theta >= M_PI) {
        return -1; /* Invalid state */
    }
    
    /* Compute derivatives based on GLE system */
    /* dh/ds = sin(theta) */
    NV_Ith_S(ydot, 0) = sin(theta);
    
    /* dtheta/ds = omega */
    NV_Ith_S(ydot, 1) = omega;
    
    /* domega/ds = 3*Ca*f(theta, mu_r)/(h*(h + 3*lambda_slip)) - cos(theta) */
    double f_val = f_combined(theta, params->mu_r);
    if (!is_finite_value(f_val)) {
        return -1; /* Invalid f value */
    }
    
    double denominator = h * (h + 3.0 * params->lambda_slip);
    if (fabs(denominator) < 1e-15) {
        return -1; /* Division by zero */
    }
    
    NV_Ith_S(ydot, 2) = 3.0 * params->Ca * f_val / denominator - cos(theta);
    
    return 0; /* Success */
}

void* gle_cvode_init(GLEParams *params) {
    void *cvode_mem;
    N_Vector y;
    int flag;
    
    /* Create initial condition vector */
    y = N_VNew_Serial(NEQ, params->sunctx);
    if (!y) return NULL;
    
    /* Set initial conditions */
    NV_Ith_S(y, 0) = params->lambda_slip; /* h(0) = lambda_slip */
    NV_Ith_S(y, 1) = params->theta0;      /* theta(0) = theta0 */
    NV_Ith_S(y, 2) = 0.0;                 /* omega(0) = initial guess */
    
    /* Create CVODE memory */
    cvode_mem = CVodeCreate(CV_BDF, params->sunctx);
    if (!cvode_mem) {
        N_VDestroy_Serial(y);
        return NULL;
    }
    
    /* Initialize CVODE */
    flag = CVodeInit(cvode_mem, gle_ode_system, 0.0, y);
    if (flag != CV_SUCCESS) {
        CVodeFree(&cvode_mem);
        N_VDestroy_Serial(y);
        return NULL;
    }
    
    /* Set tolerances */
    flag = CVodeSStolerances(cvode_mem, 1e-8, 1e-10);
    if (flag != CV_SUCCESS) {
        CVodeFree(&cvode_mem);
        N_VDestroy_Serial(y);
        return NULL;
    }
    
    /* Set user data */
    flag = CVodeSetUserData(cvode_mem, params);
    if (flag != CV_SUCCESS) {
        CVodeFree(&cvode_mem);
        N_VDestroy_Serial(y);
        return NULL;
    }
    
    /* Set up linear solver */
    SUNMatrix A = SUNDenseMatrix(NEQ, NEQ, params->sunctx);
    SUNLinearSolver LS = SUNLinSol_Dense(y, A, params->sunctx);
    flag = CVodeSetLinearSolver(cvode_mem, LS, A);
    if (flag != CV_SUCCESS) {
        SUNLinSolFree(LS);
        SUNMatDestroy(A);
        CVodeFree(&cvode_mem);
        N_VDestroy_Serial(y);
        return NULL;
    }
    
    N_VDestroy_Serial(y);
    return cvode_mem;
}

void gle_cvode_cleanup(void *cvode_mem) {
    if (cvode_mem) {
        CVodeFree(&cvode_mem);
    }
}

int gle_solve_segment(GLEShootingData *shooting_data, N_Vector y0, 
                      double s_start, double s_end, N_Vector y_final) {
    int flag;
    sunrealtype t_out;
    
    /* Reinitialize CVODE for this segment */
    flag = CVodeReInit(shooting_data->cvode_mem, s_start, y0);
    if (flag != CV_SUCCESS) return -1;
    
    /* Integrate to end of segment */
    flag = CVode(shooting_data->cvode_mem, s_end, y_final, &t_out, CV_NORMAL);
    if (flag < 0) return -1; /* Integration failed */
    
    return 0; /* Success */
}

int gle_boundary_residual(N_Vector u, N_Vector fval, void *user_data) {
    GLEShootingData *data = (GLEShootingData*)user_data;
    GLEParams *params = data->params;
    /* int n_seg = data->n_segments; */ /* TODO: Use for multiple shooting */
    
    /* u contains the shooting parameters (initial conditions for each segment) */
    /* For simplicity, we'll implement a basic multiple shooting approach */
    
    /* Initialize residual vector */
    for (int i = 0; i < NV_LENGTH_S(fval); i++) {
        NV_Ith_S(fval, i) = 0.0;
    }
    
    /* This is a simplified implementation - in practice, you would:
     * 1. Extract shooting parameters from u
     * 2. Solve each segment using gle_solve_segment
     * 3. Compute continuity residuals at interfaces
     * 4. Compute boundary condition residuals
     */
    
    /* For now, return a placeholder implementation */
    NV_Ith_S(fval, 0) = NV_Ith_S(u, 0) - params->theta0; /* theta(0) = theta0 */
    if (NV_LENGTH_S(fval) > 1) {
        NV_Ith_S(fval, 1) = NV_Ith_S(u, 1) - params->lambda_slip; /* h(0) = lambda_slip */
    }
    if (NV_LENGTH_S(fval) > 2) {
        NV_Ith_S(fval, 2) = NV_Ith_S(u, 2) - params->w; /* omega(Delta) = w */
    }
    
    return 0;
}

int gle_solve_bvp(GLEParams *params, GLESolution *solution) {
    void *cvode_mem;
    N_Vector y, ydot;
    int flag;
    double s_end = 4.0 * params->Delta;
    int n_points = solution->n_points;
    
    /* Initialize CVODE */
    cvode_mem = gle_cvode_init(params);
    if (!cvode_mem) {
        printf("Error: Failed to initialize CVODE\n");
        return -1;
    }
    
    /* Create working vectors */
    y = N_VNew_Serial(NEQ, params->sunctx);
    ydot = N_VNew_Serial(NEQ, params->sunctx);
    if (!y || !ydot) {
        gle_cvode_cleanup(cvode_mem);
        return -1;
    }
    
    /* Set initial conditions */
    NV_Ith_S(y, 0) = params->lambda_slip; /* h(0) = lambda_slip */
    NV_Ith_S(y, 1) = params->theta0;      /* theta(0) = theta0 */
    NV_Ith_S(y, 2) = 0.0;                 /* omega(0) = initial guess */
    
    /* Reinitialize CVODE with initial conditions */
    flag = CVodeReInit(cvode_mem, 0.0, y);
    if (flag != CV_SUCCESS) {
        printf("Error: CVodeReInit failed with flag %d\n", flag);
        N_VDestroy_Serial(y);
        N_VDestroy_Serial(ydot);
        gle_cvode_cleanup(cvode_mem);
        return -1;
    }
    
    /* Generate solution points */
    for (int i = 0; i < n_points; i++) {
        double s = (double)i * s_end / (double)(n_points - 1);
        solution->s_values[i] = s;
        
        if (i == 0) {
            /* Initial conditions */
            solution->h_values[i] = NV_Ith_S(y, 0);
            solution->theta_values[i] = NV_Ith_S(y, 1);
            solution->w_values[i] = NV_Ith_S(y, 2);
        } else {
            /* Integrate to next point */
            sunrealtype t_out;
            flag = CVode(cvode_mem, s, y, &t_out, CV_NORMAL);
            if (flag < 0) {
                printf("Warning: CVode integration failed at s=%g with flag %d\n", s, flag);
                solution->converged = 0;
                break;
            }
            
            /* Store solution */
            solution->h_values[i] = NV_Ith_S(y, 0);
            solution->theta_values[i] = NV_Ith_S(y, 1);
            solution->w_values[i] = NV_Ith_S(y, 2);
            
            /* Check for physical validity */
            if (solution->h_values[i] <= 0 || 
                solution->theta_values[i] <= 0 || 
                solution->theta_values[i] >= M_PI) {
                printf("Warning: Unphysical solution at s=%g\n", s);
                solution->converged = 0;
                break;
            }
        }
    }
    
    /* Check final boundary condition (simplified) */
    double final_omega = solution->w_values[n_points - 1];
    if (fabs(final_omega - params->w) > 1e-6) {
        printf("Warning: Final boundary condition not satisfied. Expected w=%g, got %g\n", 
               params->w, final_omega);
        solution->converged = 0;
    } else {
        solution->converged = 1;
    }
    
    /* Clean up */
    N_VDestroy_Serial(y);
    N_VDestroy_Serial(ydot);
    gle_cvode_cleanup(cvode_mem);
    
    return solution->converged ? 0 : -1;
}

void gle_print_stats(GLESolution *solution) {
    printf("=== GLE Solver Statistics ===\n");
    printf("Converged: %s\n", solution->converged ? "Yes" : "No");
    printf("Iterations: %d\n", solution->iterations);
    printf("Solution points: %d\n", solution->n_points);
    
    if (solution->n_points > 0) {
        printf("Domain: s ∈ [%g, %g]\n", 
               solution->s_values[0], 
               solution->s_values[solution->n_points - 1]);
        printf("h range: [%g, %g]\n",
               solution->h_values[0],
               solution->h_values[solution->n_points - 1]);
        printf("theta range: [%g, %g] rad = [%g, %g] deg\n",
               solution->theta_values[0],
               solution->theta_values[solution->n_points - 1],
               solution->theta_values[0] * 180.0 / M_PI,
               solution->theta_values[solution->n_points - 1] * 180.0 / M_PI);
    }
    printf("=============================\n");
}
