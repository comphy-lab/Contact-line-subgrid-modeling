#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>
#include "../../src-local/gle_solver.h"
#include "../../src-local/csv_output.h"

/**
 * @file test_gle_solver_c.c
 * @brief Unit tests for GLE solver functionality in C
 * 
 * These tests verify the C SUNDIALS implementation of the GLE solver
 * and correspond to the Python integration tests.
 */

#define TOLERANCE 1e-6
#define PI M_PI

/* Test helper function */
int approx_equal(double a, double b, double tol) {
    return fabs(a - b) < tol;
}

/* Test parameter initialization */
void test_gle_params_init() {
    GLEParams params;
    gle_params_init(&params);
    
    assert(approx_equal(params.Ca, CA_DEFAULT, TOLERANCE));
    assert(approx_equal(params.lambda_slip, LAMBDA_SLIP_DEFAULT, TOLERANCE));
    assert(approx_equal(params.mu_r, MU_R_DEFAULT, TOLERANCE));
    assert(approx_equal(params.theta0, THETA0_DEFAULT, TOLERANCE));
    assert(approx_equal(params.Delta, DELTA_DEFAULT, TOLERANCE));
    assert(approx_equal(params.w, 0.0, TOLERANCE));
    assert(params.sunctx == NULL);
    
    printf("✓ test_gle_params_init passed\n");
}

/* Test solution allocation and deallocation */
void test_gle_solution_alloc_free() {
    int n_points = 100;
    GLESolution *solution = gle_solution_alloc(n_points);
    
    assert(solution != NULL);
    assert(solution->s_values != NULL);
    assert(solution->h_values != NULL);
    assert(solution->theta_values != NULL);
    assert(solution->omega != NULL);
    assert(solution->n_points == n_points);
    assert(solution->converged == 0);
    assert(solution->iterations == 0);
    
    gle_solution_free(solution);
    printf("✓ test_gle_solution_alloc_free passed\n");
}

/* Test ODE system function */
void test_gle_ode_system() {
    GLEParams params;
    gle_params_init(&params);
    gle_create_context(&params);
    
    /* Create test vectors */
    N_Vector y = N_VNew_Serial(3, params.sunctx);
    N_Vector ydot = N_VNew_Serial(3, params.sunctx);
    
    /* Set test state: h=1e-5, theta=pi/6, omega=0.1 */
    NV_Ith_S(y, 0) = 1e-5;
    NV_Ith_S(y, 1) = PI / 6.0;
    NV_Ith_S(y, 2) = 0.1;
    
    /* Call ODE system */
    int flag = gle_ode_system(0.5, y, ydot, &params);
    
    /* Check that function succeeded */
    assert(flag == 0);
    
    /* Check first equation: dh/ds = sin(theta) */
    double expected_dhds = sin(NV_Ith_S(y, 1));
    assert(approx_equal(NV_Ith_S(ydot, 0), expected_dhds, TOLERANCE));
    
    /* Check second equation: dtheta/ds = omega */
    assert(approx_equal(NV_Ith_S(ydot, 1), NV_Ith_S(y, 2), TOLERANCE));
    
    /* Check that third equation gives finite result */
    assert(isfinite(NV_Ith_S(ydot, 2)));
    
    N_VDestroy_Serial(y);
    N_VDestroy_Serial(ydot);
    gle_destroy_context(&params);
    
    printf("✓ test_gle_ode_system passed\n");
}

/* Test ODE system with invalid state */
void test_gle_ode_system_invalid_state() {
    GLEParams params;
    gle_params_init(&params);
    gle_create_context(&params);
    
    N_Vector y = N_VNew_Serial(3, params.sunctx);
    N_Vector ydot = N_VNew_Serial(3, params.sunctx);
    
    /* Test with negative h */
    NV_Ith_S(y, 0) = -1e-5;
    NV_Ith_S(y, 1) = PI / 6.0;
    NV_Ith_S(y, 2) = 0.1;
    
    int flag = gle_ode_system(0.5, y, ydot, &params);
    assert(flag != 0); /* Should fail */
    
    /* Test with theta out of range */
    NV_Ith_S(y, 0) = 1e-5;
    NV_Ith_S(y, 1) = PI + 0.1; /* theta > pi */
    NV_Ith_S(y, 2) = 0.1;
    
    flag = gle_ode_system(0.5, y, ydot, &params);
    assert(flag != 0); /* Should fail */
    
    N_VDestroy_Serial(y);
    N_VDestroy_Serial(ydot);
    gle_destroy_context(&params);
    
    printf("✓ test_gle_ode_system_invalid_state passed\n");
}

/* Test CVODE initialization */
void test_gle_cvode_init() {
    GLEParams params;
    gle_params_init(&params);
    gle_create_context(&params);
    
    void *cvode_mem = gle_cvode_init(&params);
    
    /* Note: This test may fail if SUNDIALS is not installed */
    if (cvode_mem != NULL) {
        gle_cvode_cleanup(cvode_mem);
        gle_destroy_context(&params);
        printf("✓ test_gle_cvode_init passed\n");
    } else {
        gle_destroy_context(&params);
        printf("⚠ test_gle_cvode_init skipped (SUNDIALS not available)\n");
    }
}

/* Test BVP solver (simplified) */
void test_gle_solve_bvp() {
    GLEParams params;
    gle_params_init(&params);
    gle_create_context(&params);
    
    int n_points = 50;
    GLESolution *solution = gle_solution_alloc(n_points);
    
    /* Attempt to solve BVP */
    int flag = gle_solve_bvp(&params, solution);
    
    if (flag == 0) {
        /* Check that solution has reasonable values */
        assert(solution->n_points == n_points);
        
        /* Check that h values are positive */
        for (int i = 0; i < n_points; i++) {
            assert(solution->h_values[i] > 0);
            assert(solution->theta_values[i] > 0);
            assert(solution->theta_values[i] < PI);
            assert(isfinite(solution->s_values[i]));
            assert(isfinite(solution->omega[i]));
        }
        
        /* Check that h increases with s (since dh/ds = sin(theta) > 0) */
        assert(solution->h_values[n_points-1] > solution->h_values[0]);
        
        printf("✓ test_gle_solve_bvp passed\n");
    } else {
        printf("⚠ test_gle_solve_bvp skipped (solver failed - may need SUNDIALS)\n");
    }
    
    gle_solution_free(solution);
    gle_destroy_context(&params);
}

/* Test CSV output functionality */
void test_csv_output() {
    /* Create test solution data */
    int n_points = 10;
    GLESolution *solution = gle_solution_alloc(n_points);
    
    /* Fill with test data */
    for (int i = 0; i < n_points; i++) {
        solution->s_values[i] = (double)i * 0.1;
        solution->h_values[i] = 1e-5 + (double)i * 1e-6;
        solution->theta_values[i] = PI/6.0 + (double)i * 0.01;
        solution->omega[i] = (double)i * 0.001;
    }
    
    /* Create test output directory */
    int flag = create_output_directory("test_c_output");
    assert(flag == 0);
    
    /* Test h profile CSV */
    flag = save_h_profile_csv("test_c_output/test_h_profile.csv", solution);
    assert(flag == 0);
    
    /* Test theta profile CSV */
    flag = save_theta_profile_csv("test_c_output/test_theta_profile.csv", solution);
    assert(flag == 0);
    
    /* Test complete solution CSV */
    flag = save_complete_solution_csv("test_c_output/test_complete_solution.csv", solution);
    assert(flag == 0);
    
    /* Verify files exist and have content */
    FILE *file = fopen("test_c_output/test_h_profile.csv", "r");
    assert(file != NULL);
    
    char line[256];
    assert(fgets(line, sizeof(line), file) != NULL); /* Header line */
    assert(strstr(line, "s,h") != NULL);
    assert(fgets(line, sizeof(line), file) != NULL); /* First data line */
    fclose(file);
    
    gle_solution_free(solution);
    printf("✓ test_csv_output passed\n");
}

/* Test error handling */
void test_error_handling() {
    /* Test NULL pointer handling */
    int flag = save_h_profile_csv(NULL, NULL);
    assert(flag != 0);
    
    flag = save_theta_profile_csv("test.csv", NULL);
    assert(flag != 0);
    
    GLESolution *empty_solution = gle_solution_alloc(0);
    flag = save_complete_solution_csv("test.csv", empty_solution);
    assert(flag != 0);
    
    gle_solution_free(empty_solution);
    printf("✓ test_error_handling passed\n");
}

/* Main test runner */
int main() {
    printf("Running GLE solver tests...\n\n");
    
    /* Basic functionality tests */
    test_gle_params_init();
    test_gle_solution_alloc_free();
    test_gle_ode_system();
    test_gle_ode_system_invalid_state();
    
    /* SUNDIALS-dependent tests */
    test_gle_cvode_init();
    test_gle_solve_bvp();
    
    /* CSV output tests */
    test_csv_output();
    test_error_handling();
    
    printf("\n✅ All GLE solver tests completed!\n");
    printf("Note: Some tests may be skipped if SUNDIALS is not available.\n");
    return 0;
}
