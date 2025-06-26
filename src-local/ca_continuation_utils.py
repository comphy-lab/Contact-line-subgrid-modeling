"""
Utility functions for Ca-based continuation methods in the GLE solver.
"""

import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import sys
import os
import time

# Import from local modules
from solution_types import SolutionResult, SolutionCache
from gle_utils import solve_single_ca, get_adaptive_max_nodes

def get_decimal_places_from_tolerance(tolerance: float) -> int:
    """
    Determine appropriate decimal places based on tolerance.
    
    Args:
        tolerance: The tolerance value
    
    Returns:
        Number of decimal places to use
    """
    if tolerance >= 1:
        return 0
    elif tolerance >= 0.1:
        return 1
    elif tolerance >= 0.01:
        return 2
    elif tolerance >= 0.001:
        return 3
    elif tolerance >= 0.0001:
        return 4
    elif tolerance >= 0.00001:
        return 5
    elif tolerance >= 0.000001:
        return 6
    elif tolerance >= 0.0000001:
        return 7
    elif tolerance >= 0.00000001:
        return 8
    elif tolerance >= 0.000000001:
        return 9
    else:
        # For very small tolerances, calculate based on log10
        import math
        return min(15, max(10, int(-math.log10(tolerance))))

def get_adaptive_tolerance(Ca_interval_width: float, base_tolerance: float = 1e-6) -> float:
    """
    Adaptive tolerance based on interval width.
    
    Args:
        Ca_interval_width: Width of the Ca interval
        base_tolerance: Base tolerance for convergence
    
    Returns:
        Adjusted tolerance
    """
    if Ca_interval_width > 1e-2:
        return 1e-3  # Coarse tolerance for wide intervals
    elif Ca_interval_width > 1e-4:
        return 1e-4  # Medium tolerance
    else:
        return base_tolerance  # Fine tolerance near convergence

def get_adaptive_max_nodes(Ca_interval_width: float, theta_min: Optional[float] = None) -> int:
    """
    Adaptive max nodes based on proximity to critical point.
    
    Args:
        Ca_interval_width: Width of the Ca interval
        theta_min: Minimum theta value (if available)
    
    Returns:
        Maximum number of nodes for BVP solver
    """
    if Ca_interval_width > 1e-2:
        return 50000
    elif theta_min and theta_min > 0.5:  # Far from critical
        return 100000
    else:
        return 500000  # Near critical point

def interpolate_solution(Ca_target: float, Ca1: float, sol1: SolutionResult, 
                        Ca2: float, sol2: SolutionResult) -> Tuple[np.ndarray, np.ndarray]:
    """
    Linearly interpolate solution between two known solutions.
    
    Args:
        Ca_target: Target Ca value
        Ca1, Ca2: Known Ca values
        sol1, sol2: Known solutions
    
    Returns:
        Interpolated (s_range, y_guess)
    """
    # Interpolation weight
    w = (Ca_target - Ca1) / (Ca2 - Ca1)
    
    # Use the solution data
    s1, y1 = sol1.s_range, sol1.y_guess
    s2, y2 = sol2.s_range, sol2.y_guess
    
    # Interpolate s_range (use finer of the two)
    if len(s1) >= len(s2):
        s_range = s1
        y2_interp = np.array([np.interp(s_range, s2, y2[i]) for i in range(3)])
        y_guess = (1-w) * y1 + w * y2_interp
    else:
        s_range = s2
        y1_interp = np.array([np.interp(s_range, s1, y1[i]) for i in range(3)])
        y_guess = w * y2 + (1-w) * y1_interp
    
    return s_range, y_guess

def inverse_quadratic_interpolation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Perform inverse quadratic interpolation to find x where y = 0.
    Uses the last 3 points in the arrays with improved numerical stability.
    """
    # Use last 3 points
    x0, x1, x2 = x[-3], x[-2], x[-1]
    y0, y1, y2 = y[-3], y[-2], y[-1]
    
    # Scale for numerical stability
    x_scale = max(abs(x0), abs(x1), abs(x2))
    y_scale = max(abs(y0), abs(y1), abs(y2))
    if x_scale < 1e-10 or y_scale < 1e-10:
        return 0.5 * (x[-1] + x[-2])
    
    # Normalize values
    x0_n, x1_n, x2_n = x0/x_scale, x1/x_scale, x2/x_scale
    y0_n, y1_n, y2_n = y0/y_scale, y1/y_scale, y2/y_scale
    
    # Check for duplicate or near-duplicate y values
    eps = 1e-10
    y_diffs = [abs(y0_n - y1_n), abs(y0_n - y2_n), abs(y1_n - y2_n)]
    if min(y_diffs) < eps:
        # Fall back to linear interpolation
        if abs(y1 - y2) > eps * y_scale and abs(x1 - x2) > eps * x_scale:
            return x2 - y2 * (x2 - x1) / (y2 - y1)
        elif abs(y0 - y2) > eps * y_scale and abs(x0 - x2) > eps * x_scale:
            return x2 - y2 * (x2 - x0) / (y2 - y0)
        else:
            return 0.5 * (x[-1] + x[-2])
    
    # Check condition number for stability
    # If points are too close in x-space relative to y-variation, IQI may be unstable
    x_range = max(abs(x0_n - x1_n), abs(x0_n - x2_n), abs(x1_n - x2_n))
    y_range = max(y_diffs)
    if x_range < eps or y_range / x_range > 1e6:
        # Points too close or badly conditioned
        # Use linear extrapolation from last two points
        if abs(y1 - y2) > eps * y_scale:
            return x2 - y2 * (x2 - x1) / (y2 - y1)
        else:
            return 0.5 * (x[-1] + x[-2])
    
    # IQI formula with scaled values
    denom1 = (y0_n - y1_n) * (y0_n - y2_n)
    denom2 = (y1_n - y0_n) * (y1_n - y2_n)
    denom3 = (y2_n - y0_n) * (y2_n - y1_n)
    
    # Additional stability check
    if abs(denom1) < eps or abs(denom2) < eps or abs(denom3) < eps:
        # Use linear extrapolation
        if abs(y1 - y2) > eps * y_scale:
            return x2 - y2 * (x2 - x1) / (y2 - y1)
        else:
            return 0.5 * (x[-1] + x[-2])
    
    term1 = x0_n * y1_n * y2_n / denom1
    term2 = x1_n * y0_n * y2_n / denom2
    term3 = x2_n * y0_n * y1_n / denom3
    
    x_interp = (term1 + term2 + term3) * x_scale
    
    # Sanity check: IQI should not extrapolate too far
    x_min, x_max = min(x), max(x)
    x_range_full = x_max - x_min
    if x_interp < x_min - 0.5 * x_range_full or x_interp > x_max + 0.5 * x_range_full:
        # IQI suggesting unreasonable extrapolation, use linear instead
        if abs(y1 - y2) > eps * y_scale:
            return x2 - y2 * (x2 - x1) / (y2 - y1)
        else:
            return 0.5 * (x[-1] + x[-2])
    
    return x_interp

def hybrid_iqi_newton_refinement(Ca_good: float, Ca_fail: float, mu_r: float,
                                lambda_slip: float, theta0: float, w_bc: float, Delta: float,
                                s_range: np.ndarray, y_guess: np.ndarray, tolerance: float = 1e-6,
                                max_iter: int = 30) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Refine Ca_cr using hybrid Inverse Quadratic Interpolation (IQI) and Newton-Raphson method.
    
    This method combines:
    - IQI for fast convergence when we have 3+ points
    - Newton-Raphson when close to critical point (theta_min < 0.1)
    - Bisection as fallback for robustness
    
    Args:
        Ca_good: Last known good Ca
        Ca_fail: First known failing Ca
        tolerance: Convergence tolerance (default 1e-6)
        max_iter: Maximum iterations
    
    Returns:
        (Ca_critical, s_range_final, y_guess_final) where the final values are from the last successful solution
    """
    decimal_places = get_decimal_places_from_tolerance(tolerance)
    print(f"\nRefining Ca_cr between {Ca_good:.{decimal_places}f} and {Ca_fail:.{decimal_places}f}")
    print(f"Target tolerance: {tolerance}")
    
    # Store history of (Ca, theta_min) pairs for IQI
    history_Ca = []
    history_theta_min = []
    failed_Ca = set()  # Track failed attempts to avoid repeating them
    consecutive_bisection_failures = 0  # Track consecutive bisection failures
    
    # Add initial good point
    result = solve_single_ca(Ca_good, mu_r, lambda_slip, theta0, w_bc, Delta, s_range, y_guess, tol=tolerance)
    if result.success and result.theta_min is not None:
        history_Ca.append(Ca_good)
        history_theta_min.append(result.theta_min)
        s_range = result.s_range
        y_guess = result.y_guess
    
    for iter_count in range(max_iter):
        interval_width = Ca_fail - Ca_good
        
        # Check convergence
        if interval_width < tolerance:
            print(f"Converged to Ca_cr = {Ca_good:.{decimal_places}f} after {iter_count} iterations")
            break
            
        # Special handling when interval is very small but not converged
        # This happens when Ca_critical is extremely close to Ca_good
        if interval_width < 100 * tolerance and consecutive_bisection_failures > 5:
            print(f"\nInterval very small ({interval_width:.2e}) with many failures.")
            print(f"Switching to logarithmic search in tiny interval...")
            # Try logarithmically spaced points in the tiny interval
            n_points = 10
            Ca_test_points = np.logspace(np.log10(Ca_good + tolerance), 
                                        np.log10(Ca_fail), 
                                        n_points)
            for Ca_test in Ca_test_points:
                if Ca_test <= Ca_good or Ca_test >= Ca_fail:
                    continue
                result_test = solve_single_ca(
                    Ca_test, mu_r, lambda_slip, theta0, w_bc, Delta, s_range, y_guess, tol=tolerance
                )
                if result_test.success and result_test.theta_min is not None:
                    Ca_good = Ca_test
                    s_range = result_test.s_range
                    y_guess = result_test.y_guess
                    print(f"  Found better Ca: {Ca_test:.{decimal_places}f}, θ_min = {result_test.theta_min*180/np.pi:.2f}°")
                    history_Ca.append(Ca_test)
                    history_theta_min.append(result_test.theta_min)
                    consecutive_bisection_failures = 0
                    break
                else:
                    Ca_fail = Ca_test
                    failed_Ca.add(Ca_test)
        
        # Choose method based on available data and proximity to critical point
        if len(history_Ca) >= 3 and min(history_theta_min) > 0.001:
            # Use Inverse Quadratic Interpolation
            # Select best 3 points for IQI: spread out and with small theta_min
            if len(history_Ca) > 3:
                # Score points based on: small theta_min and good spacing
                scores = []
                for i in range(len(history_Ca)):
                    # Prefer points with smaller theta_min
                    theta_score = 1.0 / (1.0 + history_theta_min[i])
                    
                    # Prefer points that are well-spaced from others
                    spacing_score = 0.0
                    for j in range(len(history_Ca)):
                        if i != j:
                            spacing_score += abs(history_Ca[i] - history_Ca[j])
                    spacing_score /= len(history_Ca) - 1
                    
                    scores.append(theta_score + 0.5 * spacing_score)
                
                # Select top 3 points by score
                indices = np.argsort(scores)[-3:]
                indices.sort()  # Keep chronological order
                Ca_vals = np.array([history_Ca[i] for i in indices])
                theta_vals = np.array([history_theta_min[i] for i in indices])
            else:
                Ca_vals = np.array(history_Ca[-3:])
                theta_vals = np.array(history_theta_min[-3:])
            
            # IQI formula to find root (where theta_min = 0)
            Ca_iqi = inverse_quadratic_interpolation(Ca_vals, theta_vals)
            
            # Apply relaxation factor when close to critical point
            if min(theta_vals) < 0.1:
                # Use weighted average with current best estimate
                relax_factor = 0.7  # Take 70% of IQI suggestion
                Ca_new = relax_factor * Ca_iqi + (1 - relax_factor) * Ca_good
            else:
                Ca_new = Ca_iqi
            
            # Ensure Ca_new is within bounds
            Ca_new = max(Ca_good, min(Ca_fail, Ca_new))
            
            # Check if IQI is making progress
            if len(history_Ca) > 0:
                # Check if stuck (too close to previous attempts)
                min_dist = min(abs(Ca_new - ca) for ca in history_Ca[-3:])
                if min_dist < tolerance * 0.1:
                    # Use adaptive bisection step
                    if consecutive_bisection_failures >= 3:
                        Ca_new = Ca_good + 0.618 * (Ca_fail - Ca_good)
                        method = "Bisection (IQI stuck, golden)"
                    else:
                        Ca_new = 0.5 * (Ca_good + Ca_fail)
                        method = "Bisection (IQI stuck)"
                # Check if oscillating (going back and forth)
                elif len(history_Ca) >= 2 and abs(Ca_new - history_Ca[-2]) < tolerance:
                    if consecutive_bisection_failures >= 3:
                        Ca_new = Ca_good + 0.618 * (Ca_fail - Ca_good)
                        method = "Bisection (IQI oscillating, golden)"
                    else:
                        Ca_new = 0.5 * (Ca_good + Ca_fail)
                        method = "Bisection (IQI oscillating)"
                # Check if suggesting a value that already failed
                # Use tighter tolerance since we're working with 6 decimal places
                elif any(abs(Ca_new - ca) < tolerance for ca in failed_Ca):
                    if consecutive_bisection_failures >= 3:
                        Ca_new = Ca_good + 0.618 * (Ca_fail - Ca_good)
                        method = "Bisection (IQI suggesting failed, golden)"
                    else:
                        Ca_new = 0.5 * (Ca_good + Ca_fail)
                        method = "Bisection (IQI suggesting failed)"
                else:
                    method = "IQI"
            else:
                method = "IQI"
            
        elif len(history_Ca) >= 1 and min(history_theta_min) < 0.1:
            # Use Newton-Raphson when close to critical point
            # Estimate derivative using last two points if available
            if len(history_Ca) >= 2:
                dtheta_dCa = (history_theta_min[-1] - history_theta_min[-2]) / (history_Ca[-1] - history_Ca[-2])
            else:
                # Use finite difference
                h = 0.01 * (Ca_fail - Ca_good)
                Ca_test = Ca_good + h
                result_test = solve_single_ca(
                    Ca_test, mu_r, lambda_slip, theta0, w_bc, Delta, s_range, y_guess, tol=tolerance
                )
                if result_test.success and result_test.theta_min is not None:
                    dtheta_dCa = (result_test.theta_min - history_theta_min[-1]) / h
                else:
                    dtheta_dCa = None
            
            if dtheta_dCa is not None and abs(dtheta_dCa) > 1e-10:
                # Newton step: Ca_new = Ca - theta_min / (dtheta_min/dCa)
                Ca_new = history_Ca[-1] - history_theta_min[-1] / dtheta_dCa
                # Ensure within bounds
                Ca_new = max(Ca_good, min(Ca_fail, Ca_new))
                method = "Newton"
            else:
                # Fallback to bisection
                Ca_new = 0.5 * (Ca_good + Ca_fail)
                method = "Bisection"
        else:
            # Default to bisection with adaptive step size
            if consecutive_bisection_failures >= 5:
                # After 5 failures, use a larger step (0.8 of interval)
                Ca_new = Ca_good + 0.8 * (Ca_fail - Ca_good)
                method = "Bisection (aggressive)"
            elif consecutive_bisection_failures >= 3:
                # After 3 failures, use golden ratio instead of 0.5
                # This explores the interval more efficiently
                golden_ratio = 0.618
                Ca_new = Ca_good + golden_ratio * (Ca_fail - Ca_good)
                method = "Bisection (golden)"
            else:
                Ca_new = 0.5 * (Ca_good + Ca_fail)
                method = "Bisection"
        
        # Test the new Ca value
        result_new = solve_single_ca(
            Ca_new, mu_r, lambda_slip, theta0, w_bc, Delta, s_range, y_guess, tol=tolerance
        )
        
        if result_new.success and result_new.theta_min is not None:
            # Success - update bounds and history
            Ca_good = Ca_new
            s_range = result_new.s_range
            y_guess = result_new.y_guess
            
            history_Ca.append(Ca_new)
            history_theta_min.append(result_new.theta_min)
            
            # Keep only recent history for IQI
            if len(history_Ca) > 5:
                history_Ca = history_Ca[-5:]
                history_theta_min = history_theta_min[-5:]
            
            print(f"  Iteration {iter_count+1} ({method}): Ca = {Ca_new:.{decimal_places}f}, θ_min = {result_new.theta_min*180/np.pi:.2f}°")
            
            # Reset consecutive bisection failures on success
            consecutive_bisection_failures = 0
            
            # Early termination if theta_min is very small
            if result_new.theta_min < 0.001:
                print(f"θ_min < 0.001 rad, terminating early at Ca = {Ca_new:.{decimal_places}f}")
                Ca_good = Ca_new
                break
        else:
            # Failed - update upper bound
            Ca_fail = Ca_new
            failed_Ca.add(Ca_new)  # Track this failed attempt
            print(f"  Iteration {iter_count+1} ({method}): Ca = {Ca_new:.{decimal_places}f} failed")
            
            # Track consecutive bisection failures
            if "Bisection" in method:
                consecutive_bisection_failures += 1
            else:
                consecutive_bisection_failures = 0
                
            # If we've had many consecutive failures, update search strategy
            if consecutive_bisection_failures >= 10:
                # The critical point is very close to Ca_good
                # Instead of continuing failed bisection, restart from Ca_good
                print(f"\n  Many consecutive failures detected. Critical Ca is very close to {Ca_good:.{decimal_places}f}")
                print(f"  Switching to fine-grained search from last successful value...")
                
                # Do a very fine search starting from Ca_good
                step_size = (Ca_fail - Ca_good) * 0.001  # Start with 0.1% of interval
                Ca_test = Ca_good + step_size
                
                while Ca_test < Ca_fail and iter_count < max_iter - 1:
                    iter_count += 1
                    result_test = solve_single_ca(
                        Ca_test, mu_r, lambda_slip, theta0, w_bc, Delta, s_range, y_guess, tol=tolerance
                    )
                    if result_test.success and result_test.theta_min is not None:
                        # Success - update Ca_good
                        Ca_good = Ca_test
                        s_range = result_test.s_range
                        y_guess = result_test.y_guess
                        history_Ca.append(Ca_test)
                        history_theta_min.append(result_test.theta_min)
                        print(f"  Iteration {iter_count+1} (fine search): Ca = {Ca_test:.{decimal_places}f}, θ_min = {result_test.theta_min*180/np.pi:.2f}°")
                        # Try a slightly larger step
                        step_size *= 1.5
                    else:
                        # Failed - this is our new upper bound
                        Ca_fail = Ca_test
                        failed_Ca.add(Ca_test)
                        print(f"  Iteration {iter_count+1} (fine search): Ca = {Ca_test:.{decimal_places}f} failed")
                        # Reduce step size
                        step_size *= 0.5
                        
                    # Check if we've converged
                    if Ca_fail - Ca_good < tolerance:
                        print(f"  Converged to Ca_cr = {Ca_good:.{decimal_places}f}")
                        # Report final theta_min
                        if history_theta_min:
                            print(f"  Final θ_min = {history_theta_min[-1]*180/np.pi:.2f}°")
                        return Ca_good, s_range, y_guess
                        
                    Ca_test = Ca_good + step_size
                    
                # After fine search, reset failure counter and continue
                consecutive_bisection_failures = 0
                continue  # Skip the rest of the iteration
    
    # Diagnostic message if we exhausted iterations
    if iter_count == max_iter - 1:
        print(f"\nWARNING: Reached maximum iterations ({max_iter})")
        print(f"  Last successful Ca: {Ca_good:.{decimal_places}f}")
        print(f"  First failing Ca: {Ca_fail:.{decimal_places}f}")
        print(f"  Interval width: {Ca_fail - Ca_good:.2e}")
        print(f"  The critical Ca is likely very close to {Ca_good:.{decimal_places}f}")
    
    return Ca_good, s_range, y_guess

def find_critical_ca_lower_branch(mu_r: float, lambda_slip: float, theta0: float, 
                                 w_bc: float, Delta: float, nGridInit: int = 10000,
                                 output_dir: str = 'output', Ca_requested: float = 1.0,
                                 tolerance: float = 1e-6) -> Tuple[float, List[float], List[float], List[float], np.ndarray, np.ndarray]:
    """
    Find the critical Ca using a two-stage approach with improved refinement.
    
    Stage 1: Coarse logarithmic search
    Stage 2: Hybrid IQI + Newton-Raphson refinement
    
    Returns:
        (Ca_critical, Ca_values, x0_values, theta_min_values, s_range_final, y_guess_final)
        where s_range_final and y_guess_final are from the last successful solution
    """
    print("\nFinding critical Ca using a two-stage method...")
    timing_info = {}
    total_start = time.time()
    
    # Stage 1: Coarse logarithmic search
    stage1_start = time.time()
    Ca_coarse = np.logspace(-4, np.log10(max(Ca_requested, 1e-3)), 30)
    
    # Initial guess with better h profile
    s_range = np.linspace(0, Delta, nGridInit)
    y_guess = np.zeros((3, s_range.size))
    # Use a more physical initial guess for h
    # h grows from lambda_slip following approximately h ~ s for small s
    y_guess[0, :] = lambda_slip + s_range * np.sin(theta0)
    y_guess[1, :] = theta0
    y_guess[2, :] = 0
    
    Ca_values = []
    x0_values = []
    theta_min_values = []
    Ca_critical = 0
    Ca_fail = None
    
    print("Stage 1: Coarse search...")
    for i, Ca in enumerate(Ca_coarse):
        if i % 5 == 0:
            print(f"  Testing Ca = {Ca:.{get_decimal_places_from_tolerance(tolerance)}f}")
        
        result = solve_single_ca(
            Ca, mu_r, lambda_slip, theta0, w_bc, Delta, s_range, y_guess
        )
        
        if result.success:
            Ca_critical = Ca
            Ca_values.append(Ca)
            theta_min_values.append(result.theta_min)
            s_range = result.s_range
            y_guess = result.y_guess
            
            # x0 always exists with the new definition (position where theta is minimum)
            x0_values.append(result.x0)
            
            # Print progress with x0 information
            if result.theta_min < 0.1:
                print(f"  theta_min approaching zero at Ca = {Ca:.{get_decimal_places_from_tolerance(tolerance)}f} " +
                      f"(θ_min = {result.theta_min*180/np.pi:.2f}°, x0 = {result.x0:.4f})")
        else:
            if hasattr(result, 'message') and result.message:
                print(f"  Solution failed at Ca = {Ca:.{get_decimal_places_from_tolerance(tolerance)}f}: {result.message}")
            else:
                print(f"  Solution failed at Ca = {Ca:.{get_decimal_places_from_tolerance(tolerance)}f}")
            Ca_fail = Ca
            break
    
    timing_info['stage1_time'] = time.time() - stage1_start
    
    # Stage 1.5: If the interval is too large, do a finer search
    stage15_start = time.time()
    if Ca_critical > 0 and Ca_fail is not None:
        interval_ratio = Ca_fail / Ca_critical
        if interval_ratio > 1.1:  # Interval is more than 10% wide
            print(f"\nStage 1.5: Refining coarse interval [{Ca_critical:.6f}, {Ca_fail:.6f}]...")
            # Do a finer search in this interval
            n_refine = 10
            Ca_refine = np.linspace(Ca_critical, Ca_fail, n_refine + 2)[1:-1]  # Exclude endpoints
            for Ca in Ca_refine:
                result = solve_single_ca(
                    Ca, mu_r, lambda_slip, theta0, w_bc, Delta, s_range, y_guess
                )
                if result.success:
                    Ca_critical = Ca
                    s_range = result.s_range
                    y_guess = result.y_guess
                    if result.theta_min < 0.1:
                        print(f"  Better estimate: Ca = {Ca:.{get_decimal_places_from_tolerance(tolerance)}f}, " +
                              f"θ_min = {result.theta_min*180/np.pi:.2f}°")
                else:
                    Ca_fail = Ca
                    break
            print(f"  Refined interval: [{Ca_critical:.{get_decimal_places_from_tolerance(tolerance)}f}, " +
                  f"{Ca_fail:.{get_decimal_places_from_tolerance(tolerance)}f}]")
    
    timing_info['stage15_time'] = time.time() - stage15_start
    
    # Stage 2: Hybrid IQI + Newton refinement
    stage2_start = time.time()
    if Ca_critical > 0 and Ca_fail is not None:
        print("\nStage 2: Hybrid IQI + Newton-Raphson refinement...")
        Ca_critical_refined, s_range, y_guess = hybrid_iqi_newton_refinement(
            Ca_critical, Ca_fail, mu_r, lambda_slip, theta0, w_bc, Delta,
            s_range, y_guess, tolerance=tolerance
        )
        
        if Ca_values:
            Ca_values[-1] = Ca_critical_refined
        Ca_critical = Ca_critical_refined
    elif Ca_fail is None:
        print(f"\nNo failure found up to Ca = {Ca_coarse[-1]:.{get_decimal_places_from_tolerance(tolerance)}f}")
    
    timing_info['stage2_time'] = time.time() - stage2_start
    
    # Final extrapolation if we have enough data points near critical Ca
    if len(Ca_values) >= 3 and len(theta_min_values) >= 3:
        # Check if we're close to critical point but haven't reached θ_min = 0
        if min(theta_min_values) > 0.01:  # θ_min still > ~0.5°
            print(f"\nθ_min at Ca_cr = {Ca_critical:.{get_decimal_places_from_tolerance(tolerance)}f} is {min(theta_min_values)*180/np.pi:.2f}° (not at critical point)")
            print("Attempting extrapolation to estimate true Ca_critical...")
            
            # Use the last few points for extrapolation
            n_extrap = min(5, len(Ca_values))
            Ca_extrap = np.array(Ca_values[-n_extrap:])
            theta_extrap = np.array(theta_min_values[-n_extrap:])
            
            # Fit a polynomial to extrapolate where θ_min = 0
            # Use quadratic fit for robustness
            try:
                coeffs = np.polyfit(theta_extrap, Ca_extrap, 2)
                Ca_critical_extrap = np.polyval(coeffs, 0.0)
                
                # Sanity check - extrapolation shouldn't be too far
                if Ca_critical < Ca_critical_extrap < Ca_critical * 1.1:
                    print(f"Extrapolated Ca_critical ≈ {Ca_critical_extrap:.{get_decimal_places_from_tolerance(tolerance)}f}")
                    print(f"(Note: This is an estimate based on extrapolation)")
                else:
                    print(f"Extrapolation gave unrealistic value ({Ca_critical_extrap:.6f}), using best found value")
            except:
                print("Extrapolation failed, using best found value")
    
    print(f"\nFinal critical Ca: Ca_cr = {Ca_critical:.{get_decimal_places_from_tolerance(tolerance)}f}")
    
    # Add diagnostic about theta_min at critical Ca
    if theta_min_values:
        final_theta_min = theta_min_values[-1] if Ca_values[-1] == Ca_critical else min(theta_min_values)
        print(f"θ_min at Ca_cr: {final_theta_min*180/np.pi:.2f}°")
        if final_theta_min > 0.1:  # > ~5.7°
            print("WARNING: θ_min is not close to 0°. True critical Ca may be slightly higher.")
            print("Consider using GLE_criticalCa_advanced.py for more accurate results.")
    
    # Print timing summary for this function
    timing_info['total_time'] = time.time() - total_start
    print(f"\nCritical Ca finding stages:")
    print(f"  Stage 1 (coarse search): {timing_info['stage1_time']:.2f}s")
    print(f"  Stage 1.5 (refinement): {timing_info['stage15_time']:.2f}s")
    print(f"  Stage 2 (IQI+Newton): {timing_info['stage2_time']:.2f}s")
    print(f"  Total: {timing_info['total_time']:.2f}s")
    
    return Ca_critical, Ca_values, x0_values, theta_min_values, s_range, y_guess 