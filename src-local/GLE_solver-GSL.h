#ifndef GLE_SOLVER_GSL_H
#define GLE_SOLVER_GSL_H

#include <gsl/gsl_vector.h>
#include <gsl/gsl_multiroots.h>
// #include <gsl/gsl_bvp.h> // Temporarily commented out if not available

// Define constants used in the GLE system
#define K_VALUE 1.0
#define LAMBDA_S_VALUE 0.5
#define R_VALUE 0.1

// Function declarations for the GLE ODE system
double f1(double w);
double f2(double w);
double f3(double w);
void f_ode(double w, double* f1_val, double* f2_val, double* f3_val);
int gle_system(double s, const double y[], double dyds[], void *params);

// Function declaration for the boundary conditions
// void boundary_conditions(const gsl_vector * y_a, const gsl_vector * y_b, gsl_vector * resid, void *params); // BVP related

// Function declaration for the BVP solver
// int solve_gle_bvp(gsl_vector *y_init); // BVP related

// If GSL BVP is available, use the full function signature
#ifdef HAVE_GSL_BVP_H
int solve_gle_bvp_and_save(gsl_vector *y_initial_mesh,
                           size_t num_components,
                           size_t num_nodes,
                           gsl_vector *sol_output,
                           const char* h_output_path,
                           const char* theta_output_path);
#else
// Fallback for when gsl_bvp.h is not available
int solve_gle_bvp_and_save(gsl_vector *y_initial_mesh,
                           size_t num_components,
                           size_t num_nodes,
                           gsl_vector *sol_output,
                           const char* h_output_path,
                           const char* theta_output_path);
#endif


#endif // GLE_SOLVER_GSL_H
