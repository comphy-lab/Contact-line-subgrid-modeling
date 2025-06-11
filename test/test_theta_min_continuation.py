#!/usr/bin/env python3
"""
Test script to verify theta_min-based continuation is working correctly
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GLE_solver import solve_single_ca, find_critical_ca_lower_branch
from GLE_continuation_hybrid import solve_bvp_for_ca, solve_for_theta_min_newton

def test_theta_min_continuation():
    """Test that theta_min based continuation works"""
    print("Testing theta_min-based continuation...")
    
    # Parameters
    mu_r = 1e-6
    lambda_slip = 1e-4
    theta0 = np.pi/2
    w_bc = 0
    Delta = 10.0
    
    # Test 1: Verify we can solve for different Ca values and get theta_min
    print("\nTest 1: Solving for different Ca values...")
    Ca_values = [0.001, 0.01, 0.02]
    theta_min_values = []
    
    s_range = np.linspace(0, Delta, 5000)
    y_guess = np.zeros((3, len(s_range)))
    y_guess[0, :] = np.linspace(lambda_slip, Delta, len(s_range))
    y_guess[1, :] = theta0
    y_guess[2, :] = 0
    
    for Ca in Ca_values:
        result = solve_single_ca(Ca, mu_r, lambda_slip, theta0, w_bc, Delta, s_range, y_guess)
        if result.success:
            theta_min_values.append(result.theta_min)
            print(f"  Ca = {Ca:.3f}: theta_min = {result.theta_min*180/np.pi:.2f}°, x0 = {result.x0}")
            s_range = result.s_range
            y_guess = result.y_guess
        else:
            print(f"  Ca = {Ca:.3f}: Failed to solve")
    
    # Test 2: Verify theta_min decreases with increasing Ca
    print("\nTest 2: Checking theta_min trend...")
    if len(theta_min_values) >= 2:
        decreasing = all(theta_min_values[i] > theta_min_values[i+1] 
                        for i in range(len(theta_min_values)-1))
        if decreasing:
            print("  ✓ theta_min decreases with increasing Ca (as expected)")
        else:
            print("  ✗ theta_min does not decrease monotonically")
    
    # Test 3: Test Newton solver for theta_min
    print("\nTest 3: Testing Newton solver for theta_min...")
    if len(theta_min_values) >= 2:
        # Try to find Ca for a target theta_min between our computed values
        theta_min_target = (theta_min_values[0] + theta_min_values[1]) / 2
        Ca_guess = (Ca_values[0] + Ca_values[1]) / 2
        
        print(f"  Target theta_min = {theta_min_target*180/np.pi:.2f}°")
        print(f"  Initial Ca guess = {Ca_guess:.4f}")
        
        Ca_found, theta_min_found, solution = solve_for_theta_min_newton(
            theta_min_target, Ca_guess, mu_r, lambda_slip, theta0, w_bc, Delta
        )
        
        if Ca_found is not None:
            print(f"  ✓ Found Ca = {Ca_found:.4f} giving theta_min = {theta_min_found*180/np.pi:.2f}°")
            error = abs(theta_min_found - theta_min_target) / theta_min_target
            print(f"  Relative error: {error*100:.2f}%")
        else:
            print("  ✗ Newton solver failed to converge")
    
    # Test 4: Find critical Ca
    print("\nTest 4: Finding critical Ca...")
    try:
        Ca_cr, Ca_vals, x0_vals, theta_vals = find_critical_ca_lower_branch(
            mu_r, lambda_slip, theta0, w_bc, Delta, 
            nGridInit=5000, output_dir='test_output', tolerance=1e-3
        )
        print(f"  ✓ Found Ca_cr ≈ {Ca_cr:.4f}")
        
        # Check that theta_min approaches zero at Ca_cr
        if theta_vals:
            theta_min_at_cr = theta_vals[-1]
            print(f"  theta_min at Ca_cr = {theta_min_at_cr*180/np.pi:.2f}°")
            if theta_min_at_cr < 0.1:  # Less than ~5.7 degrees
                print("  ✓ theta_min approaches zero at Ca_cr")
            else:
                print("  ⚠ theta_min not very close to zero at Ca_cr")
    except Exception as e:
        print(f"  ✗ Error finding Ca_cr: {e}")
    
    print("\nAll tests completed!")

if __name__ == "__main__":
    # Create test output directory
    os.makedirs('test_output', exist_ok=True)
    
    try:
        test_theta_min_continuation()
    finally:
        # Cleanup
        import shutil
        if os.path.exists('test_output'):
            shutil.rmtree('test_output')