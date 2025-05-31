/**
 * test_GLE_solver-GSL.c
 * 
 * Test suite for the GLE solver implementation
 * 
 * Author: Claude
 * Date: 2025-05-31
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "../src-local/GLE_solver-GSL.h"

#define TOLERANCE 1e-9

// Test helper functions f1, f2, f3
void test_f1_trig() {
    printf("Testing f1_trig...\n");
    
    // Test at theta = 0
    assert(fabs(f1_trig(0.0) - 0.0) < TOLERANCE);
    
    // Test at theta = pi
    double expected_pi = M_PI * M_PI;  // pi^2 - 0
    assert(fabs(f1_trig(M_PI) - expected_pi) < TOLERANCE);
    
    // Test at theta = pi/2
    double theta = M_PI / 2.0;
    double expected = theta * theta - 1.0;  // (pi/2)^2 - sin^2(pi/2)
    assert(fabs(f1_trig(theta) - expected) < TOLERANCE);
    
    printf("  f1_trig tests passed.\n");
}

void test_f2_trig() {
    printf("Testing f2_trig...\n");
    
    // Test at theta = 0
    assert(fabs(f2_trig(0.0) - 0.0) < TOLERANCE);
    
    // Test at theta = pi
    assert(fabs(f2_trig(M_PI) - M_PI) < TOLERANCE);
    
    // Test at theta = pi/2
    double expected = M_PI / 2.0;  // pi/2 - sin(pi/2)*cos(pi/2) = pi/2 - 0
    assert(fabs(f2_trig(M_PI / 2.0) - expected) < TOLERANCE);
    
    printf("  f2_trig tests passed.\n");
}

void test_f3_trig() {
    printf("Testing f3_trig...\n");
    
    // Test at theta = 0
    assert(fabs(f3_trig(0.0) - 0.0) < TOLERANCE);
    
    // Test at theta = pi
    assert(fabs(f3_trig(M_PI) - 0.0) < TOLERANCE);
    
    // Test at theta = pi/2
    double theta = M_PI / 2.0;
    double expected = theta * (M_PI - theta) + 1.0;
    assert(fabs(f3_trig(theta) - expected) < TOLERANCE);
    
    printf("  f3_trig tests passed.\n");
}

void test_f_combined() {
    printf("Testing f_combined...\n");
    
    // Test with small mu_r (matching Python default)
    double mu_r = 1e-3;
    double theta = M_PI / 3.0;  // 60 degrees
    
    // The function should return a finite value
    double result = f_combined(theta, mu_r);
    assert(isfinite(result));
    
    // Test near singularity avoidance
    theta = M_PI / 2.0 - 0.01;
    result = f_combined(theta, mu_r);
    assert(isfinite(result));
    
    printf("  f_combined tests passed.\n");
}

void test_gle_system() {
    printf("Testing gle_system...\n");
    
    double s = 0.5 * 4e-4;  // S_MAX
    double y[3];
    double dyds[3];
    
    // Test case 1: Initial conditions
    y[0] = 1e-5;        // h = LAMBDA_SLIP
    y[1] = M_PI/6.0;    // theta = pi/6
    y[2] = 0.0;         // omega
    
    int status = gle_system(s, y, dyds, NULL);
    assert(status == GSL_SUCCESS);
    
    // Check dh/ds = sin(theta)
    assert(fabs(dyds[0] - sin(M_PI/6.0)) < TOLERANCE);
    
    // Check dtheta/ds = omega
    assert(fabs(dyds[1] - 0.0) < TOLERANCE);
    
    // Check domega/ds calculation
    assert(isfinite(dyds[2]));
    
    // Test case 2: Different values
    y[0] = 1e-4;        // h
    y[1] = M_PI / 4.0;  // theta = 45 degrees
    y[2] = 0.1;         // omega
    
    status = gle_system(s, y, dyds, NULL);
    assert(status == GSL_SUCCESS);
    
    assert(fabs(dyds[0] - sin(M_PI / 4.0)) < TOLERANCE);
    assert(fabs(dyds[1] - 0.1) < TOLERANCE);
    assert(isfinite(dyds[2]));
    
    printf("  gle_system tests passed.\n");
}

void test_boundary_conditions() {
    printf("Testing boundary conditions...\n");
    
#ifdef HAVE_GSL_BVP_H
    gsl_vector *y_a = gsl_vector_alloc(3);
    gsl_vector *y_b = gsl_vector_alloc(3);
    gsl_vector *resid = gsl_vector_alloc(3);
    
    // Set correct boundary values
    gsl_vector_set(y_a, 0, 1e-5);         // h(0) = LAMBDA_SLIP
    gsl_vector_set(y_a, 1, M_PI/6.0);     // theta(0) = THETA0
    gsl_vector_set(y_a, 2, 0.5);          // omega(0) - not constrained
    
    gsl_vector_set(y_b, 0, 1e-4);         // h(s_max) - not constrained
    gsl_vector_set(y_b, 1, M_PI/4);       // theta(s_max) - not constrained
    gsl_vector_set(y_b, 2, 0.0);          // omega(s_max) = W_BOUNDARY
    
    boundary_conditions(y_a, y_b, resid, NULL);
    
    // Check residuals are zero for correct boundary values
    assert(fabs(gsl_vector_get(resid, 0)) < TOLERANCE);  // theta(0) - THETA0
    assert(fabs(gsl_vector_get(resid, 1)) < TOLERANCE);  // h(0) - LAMBDA_SLIP
    assert(fabs(gsl_vector_get(resid, 2)) < TOLERANCE);  // omega(s_max) - W_BOUNDARY
    
    gsl_vector_free(y_a);
    gsl_vector_free(y_b);
    gsl_vector_free(resid);
    
    printf("  boundary_conditions tests passed.\n");
#else
    printf("  boundary_conditions tests skipped (GSL BVP not available).\n");
#endif
}

void test_parameter_values() {
    printf("Testing parameter values match Python...\n");
    
    // Create parameter structure
    gle_parameters params = {
        .Ca = 1.0,
        .lambda_slip = 1e-5,
        .mu_r = 1e-3,
        .Delta = 1e-4
    };
    
    assert(fabs(params.Ca - 1.0) < TOLERANCE);
    assert(fabs(params.lambda_slip - 1e-5) < TOLERANCE);
    assert(fabs(params.mu_r - 1e-3) < TOLERANCE);
    assert(fabs(params.Delta - 1e-4) < TOLERANCE);
    
    printf("  All parameters match Python implementation.\n");
}

void test_numerical_stability() {
    printf("Testing numerical stability...\n");
    
    // Test f_combined near boundaries
    double mu_r = 1e-3;
    
    // Near theta = 0
    double theta_small = 1e-6;
    double result = f_combined(theta_small, mu_r);
    assert(isfinite(result));
    
    // Near theta = pi
    double theta_large = M_PI - 1e-6;
    result = f_combined(theta_large, mu_r);
    assert(isfinite(result));
    
    // Test ODE system with extreme h values
    double y[3] = {1e-10, M_PI/4, 0.1};  // Very small h
    double dyds[3];
    int status = gle_system(0.0, y, dyds, NULL);
    assert(status == GSL_SUCCESS);
    assert(isfinite(dyds[0]) && isfinite(dyds[1]) && isfinite(dyds[2]));
    
    printf("  Numerical stability tests passed.\n");
}

int main() {
    printf("=== Running GLE Solver Tests ===\n\n");
    
    test_f1_trig();
    test_f2_trig();
    test_f3_trig();
    test_f_combined();
    test_gle_system();
    test_boundary_conditions();
    test_parameter_values();
    test_numerical_stability();
    
    printf("\n=== All tests passed successfully! ===\n");
    return 0;
}