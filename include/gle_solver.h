#ifndef GLE_SOLVER_H
#define GLE_SOLVER_H

#include <sundials/sundials_types.h>
#include <sundials/sundials_context.h>
#include <nvector/nvector_serial.h>
#include <cvode/cvode.h>
#include <kinsol/kinsol.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include "gle_math.h"

/**
 * @file gle_solver.h
 * @brief SUNDIALS-based solver for the Generalized Lubrication Equation (GLE)
 * 
 * This header defines the data structures and functions for solving the GLE
 * boundary value problem using multiple shooting with CVODE and KINSOL.
 */

/* Problem dimensions */
#define NEQ 3           /* Number of equations: h, theta, omega */
#define NSHOTS 10       /* Number of shooting segments */

/**
 * @brief Structure to hold GLE problem parameters
 */
typedef struct {
    double Ca;          /* Capillary number */
    double lambda_slip; /* Slip length */
    double mu_r;        /* Viscosity ratio */
    double theta0;      /* Initial contact angle */
    double Delta;       /* Domain length */
    double w;           /* Curvature boundary condition */
    SUNContext sunctx;  /* SUNDIALS context */
} GLEParams;

/**
 * @brief Structure to hold solution data
 */
typedef struct {
    double *s_values;   /* Arc length coordinates */
    double *h_values;   /* Film thickness values */
    double *theta_values; /* Contact angle values */
    double *w_values;   /* Curvature values */
    int n_points;       /* Number of solution points */
    int converged;      /* Convergence flag */
    int iterations;     /* Number of iterations */
} GLESolution;

/**
 * @brief Structure for multiple shooting solver data
 */
typedef struct {
    GLEParams *params;  /* Problem parameters */
    double *s_grid;     /* Shooting grid points */
    int n_segments;     /* Number of shooting segments */
    void *cvode_mem;    /* CVODE memory block */
    N_Vector y_temp;    /* Temporary vector for CVODE */
} GLEShootingData;

/**
 * @brief Initialize GLE parameters with default values
 * @param params Pointer to parameter structure to initialize
 */
void gle_params_init(GLEParams *params);

/**
 * @brief Allocate memory for GLE solution structure
 * @param n_points Number of solution points
 * @return Pointer to allocated solution structure
 */
GLESolution* gle_solution_alloc(int n_points);

/**
 * @brief Free memory for GLE solution structure
 * @param solution Pointer to solution structure to free
 */
void gle_solution_free(GLESolution *solution);

/**
 * @brief GLE ODE system function for CVODE
 * @param t Independent variable (arc length s)
 * @param y State vector [h, theta, omega]
 * @param ydot Derivative vector [dh/ds, dtheta/ds, domega/ds]
 * @param user_data Pointer to GLEParams structure
 * @return CVODE return flag
 */
int gle_ode_system(sunrealtype t, N_Vector y, N_Vector ydot, void *user_data);

/**
 * @brief Boundary condition residual function for KINSOL
 * @param u Vector of shooting parameters
 * @param fval Residual vector
 * @param user_data Pointer to GLEShootingData structure
 * @return KINSOL return flag
 */
int gle_boundary_residual(N_Vector u, N_Vector fval, void *user_data);

/**
 * @brief Solve single shooting segment using CVODE
 * @param shooting_data Pointer to shooting data structure
 * @param y0 Initial conditions for segment
 * @param s_start Start of integration interval
 * @param s_end End of integration interval
 * @param y_final Final state vector (output)
 * @return Success flag (0 = success)
 */
int gle_solve_segment(GLEShootingData *shooting_data, N_Vector y0, 
                      double s_start, double s_end, N_Vector y_final);

/**
 * @brief Solve GLE boundary value problem using multiple shooting
 * @param params Problem parameters
 * @param solution Solution structure (output)
 * @return Success flag (0 = success)
 */
int gle_solve_bvp(GLEParams *params, GLESolution *solution);

/**
 * @brief Initialize CVODE solver for GLE system
 * @param params Problem parameters
 * @return CVODE memory pointer (NULL on failure)
 */
void* gle_cvode_init(GLEParams *params);

/**
 * @brief Create SUNDIALS context
 * @param params Problem parameters to initialize context
 * @return 0 on success, -1 on failure
 */
int gle_create_context(GLEParams *params);

/**
 * @brief Destroy SUNDIALS context
 * @param params Problem parameters containing context
 */
void gle_destroy_context(GLEParams *params);

/**
 * @brief Clean up CVODE solver
 * @param cvode_mem CVODE memory pointer
 */
void gle_cvode_cleanup(void *cvode_mem);

/**
 * @brief Print solver statistics and convergence information
 * @param solution Solution structure
 */
void gle_print_stats(GLESolution *solution);

#endif /* GLE_SOLVER_H */
