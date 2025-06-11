#!/usr/bin/env python3
"""
Test script to debug Phase 2 (pseudo-arclength continuation) issues
"""

import sys
import os
# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
# Add src-local to path
sys.path.append(os.path.join(parent_dir, 'src-local'))

import numpy as np
from GLE_continuation_hybrid import (
    PseudoArclengthContinuation,
    solve_bvp_for_ca,
    find_critical_ca_improved
)

def test_phase2_at_turning_point():
    """Test Phase 2 starting from a known turning point"""
    # Parameters
    mu_r = 1e-6
    lambda_slip = 1e-4
    theta0 = np.pi/6  # 30 degrees
    w_bc = 0
    Delta = 10.0
    
    print("Testing Phase 2 pseudo-arclength continuation...")
    print(f"Parameters: mu_r={mu_r}, lambda_slip={lambda_slip}, theta0={theta0*180/np.pi:.1f}°")
    
    # First find approximate Ca_cr using Phase 0
    print("\n1. Finding approximate Ca_cr...")
    Ca_cr, Ca_values, x0_values, theta_min_values = find_critical_ca_improved(
        mu_r, lambda_slip, theta0, w_bc, Delta,
        tolerance=1e-3
    )
    
    print(f"\nFound Ca_cr ≈ {Ca_cr:.6f}")
    
    # Get solution at turning point
    print("\n2. Getting solution at turning point...")
    s_range = np.linspace(0, Delta, 10000)
    y_guess = np.zeros((3, len(s_range)))
    y_guess[0, :] = np.linspace(lambda_slip, Delta, len(s_range))
    y_guess[1, :] = theta0
    y_guess[2, :] = 0
    
    x0_turn, theta_min_turn, solution_turn = solve_bvp_for_ca(
        Ca_cr, mu_r, lambda_slip, theta0, w_bc, Delta, s_range, y_guess
    )
    
    if solution_turn is None:
        print("ERROR: Could not get solution at turning point!")
        return
    
    print(f"Turning point: Ca={Ca_cr:.6f}, x0={x0_turn:.4f}, theta_min={theta_min_turn*180/np.pi:.2f}°")
    
    # Test Phase 2
    print("\n3. Testing pseudo-arclength continuation...")
    arc_cont = PseudoArclengthContinuation(mu_r, lambda_slip, theta0, w_bc, Delta)
    arc_cont.previous_Ca = [Ca_cr]
    
    # Try a single step
    ds = 0.001  # Very small step
    print(f"\nAttempting single step with ds={ds}...")
    
    result = arc_cont.predictor_corrector_step(
        Ca_cr, solution_turn, x0_turn, theta_min_turn, ds
    )
    
    if result is None:
        print("Step FAILED - this explains why Phase 2 gets stuck")
        
        # Try to understand why
        print("\n4. Debugging tangent computation...")
        tangent = arc_cont.compute_tangent(Ca_cr, solution_turn, dCa=1e-6)
        if tangent is None:
            print("Tangent computation failed - solutions near turning point are singular")
        else:
            dU_ds, dCa_ds = tangent
            print(f"Tangent computed successfully: dCa_ds = {dCa_ds:.6f}")
    else:
        Ca_new, solution_new, x0_new, theta_min_new = result
        print(f"Step SUCCEEDED!")
        print(f"  New point: Ca={Ca_new:.6f}, x0={x0_new:.4f}, theta_min={theta_min_new*180/np.pi:.2f}°")

if __name__ == "__main__":
    test_phase2_at_turning_point()