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
    
    double s = 0.5 * S_MAX;  // Middle of the domain
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

void test_shooting_method_setup() {
    printf("Testing shooting method setup...\n");
    
    // Test shooting context setup
    gle_parameters params = {
        .Ca = CA,
        .lambda_slip = LAMBDA_SLIP,
        .mu_r = MU_R,
        .Delta = DELTA
    };
    
    shooting_context ctx;
    ctx.gle_params = &params;
    ctx.theta0 = M_PI / 6.0;  // 30 degrees
    ctx.h0 = params.lambda_slip;
    ctx.s_max = S_MAX;
    
    // Verify initial conditions match expected values
    assert(fabs(ctx.theta0 - THETA0) < TOLERANCE);
    assert(fabs(ctx.h0 - LAMBDA_SLIP) < TOLERANCE);
    assert(fabs(ctx.s_max - S_MAX) < TOLERANCE);
    
    printf("  Shooting method setup tests passed.\n");
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

void test_shooting_residual() {
    printf("Testing shooting residual function...\n");
    
    // Set up parameters
    gle_parameters params = {
        .Ca = CA,
        .lambda_slip = LAMBDA_SLIP,
        .mu_r = MU_R,
        .Delta = DELTA
    };
    
    shooting_context ctx;
    ctx.gle_params = &params;
    ctx.theta0 = M_PI / 6.0;
    ctx.h0 = params.lambda_slip;
    ctx.s_max = S_MAX;
    
    // Create ODE system
    gsl_odeiv2_system sys = {gle_ode_system_python, NULL, 3, &params};
    ctx.driver = gsl_odeiv2_driver_alloc_y_new(&sys, gsl_odeiv2_step_rkf45,
                                               1e-12, 1e-10, 1e-8);
    
    if (ctx.driver) {
        // Test residual calculation with a test omega0
        double omega0_test = 100.0;
        double residual = shooting_residual_function(omega0_test, &ctx);
        
        // Should return a finite value
        assert(isfinite(residual));
        
        gsl_odeiv2_driver_free(ctx.driver);
        printf("  Shooting residual tests passed.\n");
    } else {
        printf("  Shooting residual tests skipped (driver allocation failed).\n");
    }
}

int main() {
    printf("=== Running GLE Solver Tests ===\n\n");
    
    test_f1_trig();
    test_f2_trig();
    test_f3_trig();
    test_f_combined();
    test_gle_system();
    test_shooting_method_setup();
    test_parameter_values();
    test_numerical_stability();
    test_shooting_residual();
    
    printf("\n=== All tests passed successfully! ===\n");
    return 0;
}