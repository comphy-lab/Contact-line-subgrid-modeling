/**
 * GLE_solver-GSL.h
 * 
 * Header file for the GLE solver implementation
 * that matches the Python implementation exactly.
 * 
 * Date: 2025-05-31
 */

#ifndef GLE_SOLVER_GSL_H
#define GLE_SOLVER_GSL_H

#include <math.h>
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

// Include all implementation headers
#include "gle_physics.h"
#include "gle_ode_systems.h"
#include "gle_shooting.h"     // Contains shooting and optimization methods
#include "gle_optimization.h" // Stub for backward compatibility
#include "gle_io.h"

#endif // GLE_SOLVER_GSL_H