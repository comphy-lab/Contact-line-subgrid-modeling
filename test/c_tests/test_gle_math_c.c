#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "../../src-local/gle_math.h"

/**
 * @file test_gle_math_c.c
 * @brief Unit tests for GLE mathematical functions in C
 * 
 * These tests verify the C implementation against known values and
 * correspond to the Python tests in test_GLE_solver.py
 */

#define TOLERANCE 1e-12
#define PI M_PI

/* Test helper function */
int approx_equal(double a, double b, double tol) {
    return fabs(a - b) < tol;
}

/* Test f1 function */
void test_f1_at_zero() {
    double result = f1(0.0);
    assert(approx_equal(result, 0.0, TOLERANCE));
    printf("✓ test_f1_at_zero passed\n");
}

void test_f1_at_pi() {
    double result = f1(PI);
    double expected = PI * PI;
    assert(approx_equal(result, expected, TOLERANCE));
    printf("✓ test_f1_at_pi passed\n");
}

void test_f1_general() {
    double theta = PI / 4.0;
    double result = f1(theta);
    double expected = theta * theta - sin(theta) * sin(theta);
    assert(approx_equal(result, expected, TOLERANCE));
    printf("✓ test_f1_general passed\n");
}

/* Test f2 function */
void test_f2_at_zero() {
    double result = f2(0.0);
    assert(approx_equal(result, 0.0, TOLERANCE));
    printf("✓ test_f2_at_zero passed\n");
}

void test_f2_at_pi() {
    double result = f2(PI);
    double expected = PI;
    assert(approx_equal(result, expected, TOLERANCE));
    printf("✓ test_f2_at_pi passed\n");
}

void test_f2_general() {
    double theta = PI / 3.0;
    double result = f2(theta);
    double expected = theta - sin(theta) * cos(theta);
    assert(approx_equal(result, expected, TOLERANCE));
    printf("✓ test_f2_general passed\n");
}

/* Test f3 function */
void test_f3_at_zero() {
    double result = f3(0.0);
    assert(approx_equal(result, 0.0, TOLERANCE));
    printf("✓ test_f3_at_zero passed\n");
}

void test_f3_at_pi() {
    double result = f3(PI);
    assert(approx_equal(result, 0.0, TOLERANCE));
    printf("✓ test_f3_at_pi passed\n");
}

void test_f3_at_pi_half() {
    double theta = PI / 2.0;
    double result = f3(theta);
    double expected = theta * (PI - theta) + sin(theta) * sin(theta);
    assert(approx_equal(result, expected, TOLERANCE));
    printf("✓ test_f3_at_pi_half passed\n");
}

/* Test f_combined function */
void test_f_combined_near_pi_half() {
    double theta = PI / 2.0 - 0.01; /* Slightly off pi/2 to avoid singularity */
    double mu_r = 1.0;
    double result = f_combined(theta, mu_r);
    assert(is_finite_value(result));
    printf("✓ test_f_combined_near_pi_half passed\n");
}

void test_f_combined_with_small_mu_r() {
    double theta = PI / 3.0;
    double mu_r = 1e-3;
    double result = f_combined(theta, mu_r);
    assert(is_finite_value(result));
    printf("✓ test_f_combined_with_small_mu_r passed\n");
}

void test_f_combined_with_large_mu_r() {
    double theta = PI / 4.0;
    double mu_r = 1000.0;
    double result = f_combined(theta, mu_r);
    assert(is_finite_value(result));
    printf("✓ test_f_combined_with_large_mu_r passed\n");
}

/* Test parameter ranges */
void test_theta_range() {
    int n_points = 100;
    for (int i = 1; i < n_points; i++) { /* Skip i=0 and i=n_points to avoid boundaries */
        double theta = 0.01 + (PI - 0.02) * (double)i / (double)n_points;
        
        double result1 = f1(theta);
        double result2 = f2(theta);
        double result3 = f3(theta);
        
        assert(is_finite_value(result1));
        assert(is_finite_value(result2));
        assert(is_finite_value(result3));
    }
    printf("✓ test_theta_range passed\n");
}

void test_numerical_stability_near_boundaries() {
    double mu_r = 0.1;
    
    /* Near zero */
    double theta_small = 1e-6;
    double result_small = f_combined(theta_small, mu_r);
    assert(is_finite_value(result_small));
    
    /* Near pi */
    double theta_large = PI - 1e-6;
    double result_large = f_combined(theta_large, mu_r);
    assert(is_finite_value(result_large));
    
    printf("✓ test_numerical_stability_near_boundaries passed\n");
}

/* Test extreme parameter values */
void test_extreme_mu_r_values() {
    double theta = PI / 4.0;
    
    /* Very small mu_r */
    double result_small = f_combined(theta, 1e-10);
    assert(is_finite_value(result_small));
    
    /* Very large mu_r */
    double result_large = f_combined(theta, 1e10);
    assert(is_finite_value(result_large));
    
    printf("✓ test_extreme_mu_r_values passed\n");
}

/* Test is_finite_value function */
void test_is_finite_value_function() {
    assert(is_finite_value(1.0) == 1);
    assert(is_finite_value(0.0) == 1);
    assert(is_finite_value(-1.0) == 1);
    assert(is_finite_value(INFINITY) == 0);
    assert(is_finite_value(-INFINITY) == 0);
    assert(is_finite_value(NAN) == 0);
    printf("✓ test_is_finite_value_function passed\n");
}

/* Main test runner */
int main() {
    printf("Running GLE mathematical function tests...\n\n");
    
    /* Test f1 function */
    test_f1_at_zero();
    test_f1_at_pi();
    test_f1_general();
    
    /* Test f2 function */
    test_f2_at_zero();
    test_f2_at_pi();
    test_f2_general();
    
    /* Test f3 function */
    test_f3_at_zero();
    test_f3_at_pi();
    test_f3_at_pi_half();
    
    /* Test f_combined function */
    test_f_combined_near_pi_half();
    test_f_combined_with_small_mu_r();
    test_f_combined_with_large_mu_r();
    
    /* Test parameter ranges */
    test_theta_range();
    test_numerical_stability_near_boundaries();
    test_extreme_mu_r_values();
    
    /* Test utility functions */
    test_is_finite_value_function();
    
    printf("\n✅ All mathematical function tests passed!\n");
    return 0;
}
