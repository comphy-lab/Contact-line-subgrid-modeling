"""
Generalized Lubrication Equations (GLE) Solver
==============================================

This module solves the coupled ODEs for contact line dynamics using the
Generalized Lubrication Equations with optimized parallel bisection refinement.

Author: Vatsal
Created: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import os
import sys
from functools import partial
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any

# Import x0 finding utilities from src-local
import sys
sys.path.append('src-local')
from find_x0_utils import find_x0_and_theta_min

# Default parameters (can be overridden via command line)
DEFAULT_DELTA = 1e1  # Maximum s-value for the solver
DEFAULT_CA = 0.0246  # Capillary number
DEFAULT_LAMBDA_SLIP = 1e-4  # Slip length
DEFAULT_MU_R = 1e-6  # μ_g/μ_l
DEFAULT_THETA0 = np.pi/2  # theta at s = 0
DEFAULT_W = 0  # curvature boundary condition at s = Δ
DEFAULT_NGRID_INIT = 10000  # Initial number of grid points
DEFAULT_TOLERANCE = 1e-6  # Default tolerance for Ca_cr refinement

# ============================================================================
# Mathematical Functions for GLE
# ============================================================================

def f1(theta: np.ndarray) -> np.ndarray:
    """First helper function for GLE formulation."""
    return theta**2 - np.sin(theta)**2

def f2(theta: np.ndarray) -> np.ndarray:
    """Second helper function for GLE formulation."""
    return theta - np.sin(theta) * np.cos(theta)

def f3(theta: np.ndarray) -> np.ndarray:
    """Third helper function for GLE formulation."""
    return theta * (np.pi - theta) + np.sin(theta)**2

def f(theta: np.ndarray, mu_r: float) -> np.ndarray:
    """Combined function for GLE with viscosity ratio."""
    numerator = 2 * np.sin(theta)**3 * (mu_r**2 * f1(theta) + 2 * mu_r * f3(theta) + f1(np.pi - theta))
    denominator = 3 * (mu_r * f1(theta) * f2(np.pi - theta) - f1(np.pi - theta) * f2(theta))
    
    # Add small epsilon to prevent division by zero
    epsilon = 1e-10
    denominator = np.where(np.abs(denominator) < epsilon, 
                          np.sign(denominator) * epsilon + (denominator == 0) * epsilon, 
                          denominator)
    return numerator / denominator

# ============================================================================
# ODE System and Boundary Conditions
# ============================================================================

def GLE(s: float, y: np.ndarray, Ca: float, mu_r: float, lambda_slip: float) -> List[float]:
    """
    System of ODEs defining the contact line shape evolution.
    
    Args:
        s: Arc length coordinate
        y: State vector [h, theta, omega]
        Ca: Capillary number
        mu_r: Viscosity ratio
        lambda_slip: Slip length
    
    Returns:
        Derivatives [dh/ds, dtheta/ds, domega/ds]
    """
    h, theta, omega = y
    
    # Ensure h stays positive (physical constraint)
    # If h becomes too small, clamp it to a small positive value
    h_min = lambda_slip * 1e-6  # Minimum h as a fraction of slip length
    h_safe = np.maximum(h, h_min)
    
    # Ensure theta stays in physical range (0, pi)
    theta_safe = np.clip(theta, 1e-10, np.pi - 1e-10)
    
    dh_ds = np.sin(theta_safe)  # dh/ds = sin(theta)
    dt_ds = omega  # omega = dtheta/ds
    
    # Use safe values to prevent division by zero
    dw_ds = -3 * Ca * f(theta_safe, mu_r) / (h_safe * (h_safe + 3 * lambda_slip)) - np.cos(theta_safe)
    
    return [dh_ds, dt_ds, dw_ds]

def boundary_conditions(ya: np.ndarray, yb: np.ndarray, w_bc: float, 
                       theta0: float, lambda_slip: float) -> List[float]:
    """
    Boundary conditions for the BVP.
    
    Args:
        ya: State at s = 0
        yb: State at s = Delta
        w_bc: Curvature BC at s = Delta
        theta0: Initial contact angle
        lambda_slip: Slip length
    
    Returns:
        Residuals for boundary conditions
    """
    h_a, theta_a, w_a = ya  # boundary conditions at s = 0
    h_b, theta_b, w_b = yb  # boundary conditions at s = Delta
    return [
        theta_a - theta0,      # theta(0) = theta0
        h_a - lambda_slip,     # h(0) = lambda_slip
        w_b - w_bc            # w(Delta) = w_bc (curvature at s=Delta)
    ]

# ============================================================================
# Utility Functions
# ============================================================================

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

# ============================================================================
# Solution Data Structures
# ============================================================================

@dataclass
class SolutionResult:
    """Container for solution results."""
    success: bool
    solution: Optional[Any] = None
    theta_min: Optional[float] = None
    x0: Optional[float] = None
    s_range: Optional[np.ndarray] = None
    y_guess: Optional[np.ndarray] = None
    Ca: Optional[float] = None
    message: Optional[str] = None

class SolutionCache:
    """Cache for storing and interpolating solutions."""
    
    def __init__(self, max_size: int = 20):
        self.cache: Dict[float, SolutionResult] = {}
        self.max_size = max_size
    
    def add(self, result: SolutionResult) -> None:
        """Add a solution to the cache."""
        if result.Ca is not None:
            self.cache[result.Ca] = result
            # Keep only most recent entries
            if len(self.cache) > self.max_size:
                oldest_ca = min(self.cache.keys())
                del self.cache[oldest_ca]
    
    def get_nearest_two(self, Ca: float) -> Optional[Tuple[float, float, SolutionResult, SolutionResult]]:
        """Get two nearest Ca values for interpolation."""
        cas = sorted(self.cache.keys())
        if len(cas) < 2:
            return None
            
        if Ca <= cas[0]:
            return cas[0], cas[1], self.cache[cas[0]], self.cache[cas[1]]
        elif Ca >= cas[-1]:
            return cas[-2], cas[-1], self.cache[cas[-2]], self.cache[cas[-1]]
        else:
            # Find bracketing values
            for i in range(len(cas)-1):
                if cas[i] <= Ca <= cas[i+1]:
                    return cas[i], cas[i+1], self.cache[cas[i]], self.cache[cas[i+1]]
        return None

# ============================================================================
# Adaptive Optimization Functions
# ============================================================================

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

# ============================================================================
# Core Solver Functions
# ============================================================================

def solve_single_ca(Ca: float, mu_r: float, lambda_slip: float, theta0: float, 
                   w_bc: float, Delta: float, s_range: np.ndarray, y_guess: np.ndarray, 
                   tol: float = 1e-6, max_nodes: Optional[int] = None) -> SolutionResult:
    """
    Solve for a single Ca value with adaptive features.
    
    Returns:
        SolutionResult object with solution data
    """
    if max_nodes is None:
        # Estimate based on current solution
        if hasattr(y_guess, 'shape') and y_guess.shape[0] >= 3:
            theta_vals = y_guess[1, :]
            theta_min_est = np.min(theta_vals)
            max_nodes = get_adaptive_max_nodes(0.01, theta_min_est)
        else:
            max_nodes = 100000
    
    GLE_with_params = partial(GLE, Ca=Ca, mu_r=mu_r, lambda_slip=lambda_slip)
    bc_with_params = partial(boundary_conditions, w_bc=w_bc, theta0=theta0, lambda_slip=lambda_slip)
    
    try:
        solution = solve_bvp(GLE_with_params, bc_with_params, s_range, y_guess, 
                           max_nodes=max_nodes, tol=tol, verbose=0)
        
        if solution.success:
            # Calculate metrics
            h_vals, theta_vals, w_vals = solution.y
            
            # Validate solution - check for physical constraints
            if np.any(h_vals <= 0):
                # Solution has negative film thickness - unphysical
                return SolutionResult(
                    success=False,
                    solution=None,
                    theta_min=None,
                    x0=None,
                    s_range=s_range,
                    y_guess=y_guess,
                    Ca=Ca,
                    message="Solution has negative film thickness"
                )
            
            if np.any(theta_vals <= 0) or np.any(theta_vals >= np.pi):
                # Solution has theta outside physical range
                return SolutionResult(
                    success=False,
                    solution=None,
                    theta_min=None,
                    x0=None,
                    s_range=s_range,
                    y_guess=y_guess,
                    Ca=Ca,
                    message="Solution has theta outside physical range [0, π]"
                )
            
            # Find x0 (position where theta reaches its minimum) and theta_min
            x0, theta_min, x0_idx = find_x0_and_theta_min(solution.x, theta_vals)
            
            # With the new definition, x0 always exists since theta_min always exists
            # When theta_min approaches 0, we are at the critical capillary number (Ca_cr)
            
            return SolutionResult(
                success=True,
                solution=solution,
                theta_min=theta_min,
                x0=x0,
                s_range=solution.x,
                y_guess=solution.y,
                Ca=Ca
            )
        else:
            return SolutionResult(
                success=False,
                s_range=s_range,
                y_guess=y_guess,
                Ca=Ca,
                message=f"solve_bvp failed: {solution.message if hasattr(solution, 'message') else 'Unknown error'}"
            )
            
    except Exception as e:
        return SolutionResult(
            success=False,
            s_range=s_range,
            y_guess=y_guess,
            Ca=Ca,
            message=f"Exception: {str(e)}"
        )

def parallel_solve_multiple_ca(Ca_list: List[float], mu_r: float, lambda_slip: float, 
                              theta0: float, w_bc: float, Delta: float,
                              s_range_list: List[np.ndarray], y_guess_list: List[np.ndarray],
                              tol_list: List[float], max_nodes_list: List[int],
                              max_workers: int = 4) -> List[SolutionResult]:
    """
    Solve for multiple Ca values in parallel using threading.
    
    Returns:
        List of SolutionResult objects sorted by Ca
    """
    with ThreadPoolExecutor(max_workers=min(len(Ca_list), max_workers)) as executor:
        # Submit all tasks
        futures = {}
        for Ca, s_range, y_guess, tol, max_nodes in zip(
            Ca_list, s_range_list, y_guess_list, tol_list, max_nodes_list
        ):
            future = executor.submit(
                solve_single_ca, Ca, mu_r, lambda_slip, theta0, w_bc, Delta, 
                s_range, y_guess, tol, max_nodes
            )
            futures[future] = Ca
        
        # Collect results
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    return sorted(results, key=lambda x: x.Ca)

# ============================================================================
# Critical Ca Finding Functions
# ============================================================================

def inverse_quadratic_interpolation(x: np.ndarray, y: np.ndarray) -> float:
    """
    Perform inverse quadratic interpolation to find x where y = 0.
    Uses the last 3 points in the arrays.
    """
    # Use last 3 points
    x0, x1, x2 = x[-3], x[-2], x[-1]
    y0, y1, y2 = y[-3], y[-2], y[-1]
    
    # Check for duplicate y values (would cause division by zero)
    eps = 1e-15
    if abs(y0 - y1) < eps or abs(y0 - y2) < eps or abs(y1 - y2) < eps:
        # Fall back to linear interpolation between last two distinct points
        if abs(y1 - y2) > eps and abs(x1 - x2) > eps:
            # Linear interpolation to find where y = 0
            return x2 - y2 * (x2 - x1) / (y2 - y1)
        elif abs(y0 - y2) > eps and abs(x0 - x2) > eps:
            return x2 - y2 * (x2 - x0) / (y2 - y0)
        else:
            # All points too similar, return midpoint
            return 0.5 * (x[-1] + x[-2])
    
    # IQI formula
    term1 = x0 * y1 * y2 / ((y0 - y1) * (y0 - y2))
    term2 = x1 * y0 * y2 / ((y1 - y0) * (y1 - y2))
    term3 = x2 * y0 * y1 / ((y2 - y0) * (y2 - y1))
    
    return term1 + term2 + term3

def hybrid_iqi_newton_refinement(Ca_good: float, Ca_fail: float, mu_r: float,
                                lambda_slip: float, theta0: float, w_bc: float, Delta: float,
                                s_range: np.ndarray, y_guess: np.ndarray, tolerance: float = 1e-6,
                                max_iter: int = 30) -> float:
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
        Refined critical Ca value
    """
    decimal_places = get_decimal_places_from_tolerance(tolerance)
    print(f"\nRefining Ca_cr between {Ca_good:.{decimal_places}f} and {Ca_fail:.{decimal_places}f}")
    print(f"Target tolerance: {tolerance}")
    
    # Store history of (Ca, theta_min) pairs for IQI
    history_Ca = []
    history_theta_min = []
    
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
        
        # Choose method based on available data and proximity to critical point
        if len(history_Ca) >= 3 and min(history_theta_min) > 0.001:
            # Use Inverse Quadratic Interpolation
            # We want to find Ca where theta_min = 0
            # Fit quadratic through last 3 points
            Ca_vals = np.array(history_Ca[-3:])
            theta_vals = np.array(history_theta_min[-3:])
            
            # IQI formula to find root (where theta_min = 0)
            Ca_new = inverse_quadratic_interpolation(Ca_vals, theta_vals)
            
            # Ensure Ca_new is within bounds
            Ca_new = max(Ca_good, min(Ca_fail, Ca_new))
            
            # If IQI suggests same point as last attempt, use bisection
            if len(history_Ca) > 0 and abs(Ca_new - history_Ca[-1]) < tolerance * 0.1:
                Ca_new = 0.5 * (Ca_good + Ca_fail)
                method = "Bisection (IQI stuck)"
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
            # Default to bisection
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
            
            # Early termination if theta_min is very small
            if result_new.theta_min < 0.001:
                print(f"θ_min < 0.001 rad, terminating early at Ca = {Ca_new:.{decimal_places}f}")
                Ca_good = Ca_new
                break
        else:
            # Failed - update upper bound
            Ca_fail = Ca_new
            print(f"  Iteration {iter_count+1} ({method}): Ca = {Ca_new:.{decimal_places}f} failed")
    
    return Ca_good

def find_critical_ca_lower_branch(mu_r: float, lambda_slip: float, theta0: float, 
                                 w_bc: float, Delta: float, nGridInit: int = DEFAULT_NGRID_INIT,
                                 output_dir: str = 'output', Ca_requested: float = 1.0,
                                 tolerance: float = 1e-6) -> Tuple[float, List[float], List[float], List[float]]:
    """
    Find the critical Ca using a two-stage approach with improved refinement.
    
    Stage 1: Coarse logarithmic search
    Stage 2: Hybrid IQI + Newton-Raphson refinement
    
    Returns:
        (Ca_critical, Ca_values, x0_values, theta_min_values)
    """
    print("\nFinding critical Ca using improved two-stage method...")
    
    # Stage 1: Coarse logarithmic search
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
    
    # Stage 2: Hybrid IQI + Newton refinement
    if Ca_critical > 0 and Ca_fail is not None:
        print("\nStage 2: Hybrid IQI + Newton-Raphson refinement...")
        Ca_critical_refined = hybrid_iqi_newton_refinement(
            Ca_critical, Ca_fail, mu_r, lambda_slip, theta0, w_bc, Delta,
            s_range, y_guess, tolerance=tolerance
        )
        
        if Ca_values:
            Ca_values[-1] = Ca_critical_refined
        Ca_critical = Ca_critical_refined
    elif Ca_fail is None:
        print(f"\nNo failure found up to Ca = {Ca_coarse[-1]:.{get_decimal_places_from_tolerance(tolerance)}f}")
    
    print(f"\nFinal critical Ca: Ca_cr = {Ca_critical:.{get_decimal_places_from_tolerance(tolerance)}f}")
    
    return Ca_critical, Ca_values, x0_values, theta_min_values

# ============================================================================
# Plotting Functions
# ============================================================================

def plot_continuation_results(Ca_values: List[float], x0_values: List[float], 
                            theta_min_values: List[float], output_dir: str = 'output',
                            tolerance: float = 1e-6) -> None:
    """Create plots showing theta_min vs Ca and x_0 vs Ca (x0 is position where theta is minimum)."""
    if not Ca_values:
        return
        
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # If x0_values is empty or all None, only plot theta_min
    has_x0_data = x0_values and any(x is not None for x in x0_values)
    
    if has_x0_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax2 = None
    
    # Convert theta_min to degrees
    theta_min_deg = np.array(theta_min_values) * 180 / np.pi
    
    # Plot 1: theta_min vs Ca
    ax1.scatter(Ca_values, theta_min_deg, s=50, alpha=0.7, color='darkred', 
                edgecolors='black', linewidth=1)
    ax1.plot(Ca_values, theta_min_deg, '-', color='crimson', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Ca (Capillary Number)', fontsize=12)
    ax1.set_ylabel('$\\theta_{min}$ [degrees]', fontsize=12)
    ax1.set_title('Minimum Contact Angle vs Capillary Number', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    if max(Ca_values) / min(Ca_values) > 100:
        ax1.set_xscale('log')
        ax1.set_xlabel('Ca (Capillary Number) [log scale]', fontsize=12)
    
    # Mark critical Ca on theta_min plot
    Ca_cr = Ca_values[-1]
    decimal_places = get_decimal_places_from_tolerance(tolerance)
    ax1.axvline(x=Ca_cr, color='green', linestyle='--', linewidth=2, alpha=0.7, 
                label=f'Ca_cr ≈ {Ca_cr:.{decimal_places}f} ± {tolerance}')
    ax1.legend(fontsize=14)
    
    # Plot 2: x_0 vs Ca (only if x0 data exists)
    if has_x0_data and ax2 is not None:
        # Filter out None values for plotting
        Ca_with_x0 = [Ca for Ca, x0 in zip(Ca_values, x0_values) if x0 is not None]
        x0_filtered = [x0 for x0 in x0_values if x0 is not None]
        
        if Ca_with_x0 and x0_filtered:
            ax2.scatter(Ca_with_x0, x0_filtered, s=50, alpha=0.7, color='darkblue', 
                        edgecolors='black', linewidth=1)
            ax2.plot(Ca_with_x0, x0_filtered, '-', color='blue', linewidth=2, alpha=0.8)
            ax2.set_xlabel('Ca (Capillary Number)', fontsize=12)
            ax2.set_ylabel('$x_0$ (Position at $\\theta_{min}$)', fontsize=12)
            ax2.set_title('Position at Minimum Contact Angle vs Capillary Number', 
                          fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            if max(Ca_with_x0) / min(Ca_with_x0) > 100:
                ax2.set_xscale('log')
                ax2.set_xlabel('Ca (Capillary Number) [log scale]', fontsize=12)
            
            ax2.axvline(x=Ca_cr, color='green', linestyle='--', linewidth=2, alpha=0.7,
                        label=f'Ca_cr ≈ {Ca_cr:.{decimal_places}f} ± {tolerance}')
            ax2.legend(fontsize=14)
    
    plt.suptitle('Lower Branch Continuation Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'continuation_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Continuation results plot saved to: {plot_path}")
    
    # Save data to CSV
    if has_x0_data:
        # Include x0 values, replacing None with NaN
        x0_for_csv = [x if x is not None else np.nan for x in x0_values]
        csv_data = np.column_stack((Ca_values, theta_min_values, theta_min_deg, x0_for_csv))
        csv_path = os.path.join(output_dir, 'continuation_data.csv')
        np.savetxt(csv_path, csv_data, delimiter=',', 
                   header='Ca,theta_min_rad,theta_min_deg,x0', comments='')
    else:
        # No x0 data
        csv_data = np.column_stack((Ca_values, theta_min_values, theta_min_deg))
        csv_path = os.path.join(output_dir, 'continuation_data.csv')
        np.savetxt(csv_path, csv_data, delimiter=',', 
                   header='Ca,theta_min_rad,theta_min_deg', comments='')
    print(f"Continuation data saved to: {csv_path}")

def run_solver_and_plot(Delta: float, Ca: float, lambda_slip: float, mu_r: float, 
                       theta0: float, w: float, nGridInit: int = DEFAULT_NGRID_INIT, 
                       GUI: bool = False, output_dir: str = 'output',
                       tolerance: float = 1e-6) -> Tuple:
    """
    Run the solver and create plots.
    
    Returns:
        (solution, s_values, h_values, theta_values, w_values)
    """
    # Set matplotlib backend
    if not GUI:
        import matplotlib
        matplotlib.use('Agg')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Initial guess with better h profile
    s_range_local = np.linspace(0, Delta, nGridInit)
    y_guess_local = np.zeros((3, s_range_local.size))
    # Use a more physical initial guess for h
    # h grows from lambda_slip following approximately h ~ s for small s
    y_guess_local[0, :] = lambda_slip + s_range_local * np.sin(theta0)
    y_guess_local[1, :] = theta0
    y_guess_local[2, :] = 0

    # Try direct solve first
    print(f"\nAttempting direct solve with Ca = {Ca}, mu_r = {mu_r}...")
    result = solve_single_ca(Ca, mu_r, lambda_slip, theta0, w, Delta, 
                           s_range_local, y_guess_local)
    
    Ca_actual = Ca
    Ca_cr = None
    Ca_values = []
    x0_values = []
    theta_min_values = []
    
    if not result.success:
        print("Direct solve failed. Finding critical Ca...")
        # Find critical Ca
        Ca_cr, Ca_values, x0_values, theta_min_values = find_critical_ca_lower_branch(
            mu_r, lambda_slip, theta0, w, Delta, nGridInit, output_dir, 
            Ca_requested=Ca, tolerance=tolerance
        )
        
        # Solve at critical Ca
        if Ca_cr > 0:
            Ca_actual = Ca_cr
            print(f"\nSolving at critical Ca = {Ca_cr:.{get_decimal_places_from_tolerance(tolerance)}f}")
            
            result = solve_single_ca(Ca_actual, mu_r, lambda_slip, theta0, w, Delta,
                                   s_range_local, y_guess_local)
        else:
            print("Failed to find critical Ca")
            return None, None, None, None, None

    if not result.success:
        print("Failed to obtain solution")
        return None, None, None, None, None

    # Extract solution
    solution = result.solution
    s_values_local = solution.x
    h_values_local, theta_values_local, w_values_local = solution.y
    theta_values_deg = theta_values_local * 180 / np.pi
    
    # Calculate x(s)
    x_values_local = np.zeros_like(s_values_local)
    x_values_local[1:] = np.cumsum(np.diff(s_values_local) * np.cos(theta_values_local[:-1]))

    # Create plots
    create_solution_plots(s_values_local, h_values_local, theta_values_local, 
                         w_values_local, x_values_local, Ca_actual, Ca, Ca_cr,
                         lambda_slip, mu_r, theta0, Delta, GUI, output_dir, tolerance)

    # Plot continuation results if available
    if Ca_values:
        plot_continuation_results(Ca_values, x0_values, theta_min_values, output_dir, tolerance)

    # Save data
    csv_data = np.column_stack((s_values_local, h_values_local, theta_values_local, 
                               w_values_local, x_values_local))
    csv_path = os.path.join(output_dir, 'data-python.csv')
    np.savetxt(csv_path, csv_data, delimiter=',', header='s,h,theta,w,x', comments='')
    print(f"Data saved to: {csv_path}")

    return solution, s_values_local, h_values_local, theta_values_local, w_values_local

def create_solution_plots(s_values: np.ndarray, h_values: np.ndarray, 
                         theta_values: np.ndarray, w_values: np.ndarray,
                         x_values: np.ndarray, Ca_actual: float, Ca_requested: float,
                         Ca_cr: Optional[float], lambda_slip: float, mu_r: float,
                         theta0: float, Delta: float, GUI: bool, output_dir: str,
                         tolerance: float = 1e-6) -> None:
    """Create the 2x2 subplot grid for solution visualization."""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    theta_values_deg = theta_values * 180 / np.pi
    solver_color = '#1f77b4'
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # Title
    if Ca_cr is not None:
        decimal_places = get_decimal_places_from_tolerance(tolerance) if tolerance else 6
        fig.suptitle(f'GLE Solution at Critical Ca = {Ca_actual:.{decimal_places}f} (requested Ca = {Ca_requested:.{decimal_places}f})', 
                     fontsize=16, fontweight='bold', color='darkred')
    else:
        decimal_places = get_decimal_places_from_tolerance(tolerance) if tolerance else 6
        fig.suptitle(f'GLE Solution at Ca = {Ca_actual:.{decimal_places}f}', 
                     fontsize=16, fontweight='bold')
    
    # Plot 1: h(s) vs s
    ax1.plot(s_values, h_values, '-', color=solver_color, linewidth=2.5)
    ax1.set_xlabel('s', fontsize=14)
    ax1.set_ylabel('h(s)', fontsize=14)
    ax1.set_title('Film Thickness Profile', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, Delta)
    ax1.set_ylim(lambda_slip, 1.01*max(h_values))
    
    # Parameter box
    if Ca_cr is not None:
        decimal_places = get_decimal_places_from_tolerance(tolerance)
        textstr = f'Ca = {Ca_actual:.{decimal_places}f} (Ca_cr)\n$\\lambda_\\text{{slip}}$ = {lambda_slip:.0e}\n$\\mu_r$ = {mu_r:.0e}'
        facecolor = 'salmon'
    else:
        decimal_places = get_decimal_places_from_tolerance(tolerance)
        textstr = f'Ca = {Ca_actual:.{decimal_places}f}\n$\\lambda_\\text{{slip}}$ = {lambda_slip:.0e}\n$\\mu_r$ = {mu_r:.0e}'
        facecolor = 'wheat'
    props = dict(boxstyle='round', facecolor=facecolor, alpha=0.5)
    ax1.text(0.02, 0.95, textstr, transform=ax1.transAxes, fontsize=16,
             verticalalignment='top', bbox=props)
    
    # Plot 2: theta(s) vs s
    ax2.plot(s_values, theta_values_deg, '-', color=solver_color, linewidth=2.5)
    ax2.set_xlabel('s', fontsize=14)
    ax2.set_ylabel('$\\theta(s)$ [degrees]', fontsize=14)
    ax2.set_title('Contact Angle Profile', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, Delta)
    ax2.set_ylim(0.99*min(theta_values_deg), 1.01*max(theta_values_deg))
    ax2.plot([], [], ' ', label=f'θ(0) = {theta0*180/np.pi:.0f}°')
    ax2.legend(loc='best', fontsize=12, fancybox=True, framealpha=0.8, 
               facecolor='lightblue', edgecolor='none')
    
    # Plot 3: theta(s) vs h(s)
    ax3.plot(h_values, theta_values_deg, '-', color=solver_color, linewidth=2.5)
    ax3.set_xlabel('h(s)', fontsize=14)
    ax3.set_ylabel('$\\theta(s)$ [degrees]', fontsize=14)
    ax3.set_title('Contact Angle vs Film Thickness', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(lambda_slip, 1.01*max(h_values))
    ax3.set_ylim(0.99*min(theta_values_deg), 1.01*max(theta_values_deg))
    
    # Plot 4: x(s) vs h(s)
    ax4.plot(h_values, x_values, '-', color=solver_color, linewidth=2.5)
    ax4.set_xlabel('h(s)', fontsize=14)
    ax4.set_ylabel('x(s)', fontsize=14)
    ax4.set_title('Horizontal Position vs Film Thickness', fontsize=16, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(lambda_slip, 1.01*max(h_values))
    ax4.set_ylim(0, 1.01*max(x_values))
    
    # Mark x0 position (where theta is minimum)
    x0, _, x0_idx = find_x0_and_theta_min(s_values, theta_values)
    if x0 is not None and x0_idx is not None:
        ax4.axvline(x=h_values[x0_idx], color='red', linestyle='--', alpha=0.5, 
                    label=f'$x_0$ = {x0:.3f} (at $\\theta_{min}$)')
        ax4.legend()
    
    plt.tight_layout()
    
    if GUI:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, 'GLE_profiles.png'), dpi=300, bbox_inches='tight')
        plt.close()

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Solve Generalized Lubrication Equations (GLE) with optimized parallel bisection'
    )
    
    # Add arguments
    parser.add_argument('--delta', type=float, default=DEFAULT_DELTA,
                        help=f'Maximum s-value for the solver (default: {DEFAULT_DELTA})')
    parser.add_argument('--ca', type=float, default=DEFAULT_CA,
                        help=f'Capillary number (default: {DEFAULT_CA})')
    parser.add_argument('--lambda-slip', type=float, default=DEFAULT_LAMBDA_SLIP,
                        help=f'Slip length (default: {DEFAULT_LAMBDA_SLIP})')
    parser.add_argument('--mu-r', type=float, default=DEFAULT_MU_R,
                        help=f'Viscosity ratio mu_g/mu_l (default: {DEFAULT_MU_R})')
    parser.add_argument('--theta0', type=float, default=DEFAULT_THETA0*180/np.pi,
                        help=f'Initial contact angle in degrees (default: {DEFAULT_THETA0*180/np.pi:.0f})')
    parser.add_argument('--w', type=float, default=DEFAULT_W,
                        help=f'Curvature boundary condition at s=Delta (default: {DEFAULT_W})')
    parser.add_argument('--ngrid-init', type=int, default=DEFAULT_NGRID_INIT,
                        help=f'Initial number of grid points (default: {DEFAULT_NGRID_INIT})')
    parser.add_argument('--tolerance', type=float, default=DEFAULT_TOLERANCE,
                        help=f'Tolerance for Ca_cr refinement (default: {DEFAULT_TOLERANCE})')
    parser.add_argument('--gui', action='store_true',
                        help='Display plots in GUI mode')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory for plots and data (default: output)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert theta0 from degrees to radians
    theta0_rad = args.theta0 * np.pi / 180
    
    # Run solver
    solution, s_values, h_values, theta_values, w_values = run_solver_and_plot(
        Delta=args.delta,
        Ca=args.ca,
        lambda_slip=args.lambda_slip,
        mu_r=args.mu_r,
        theta0=theta0_rad,
        w=args.w,
        nGridInit=args.ngrid_init,
        GUI=args.gui,
        output_dir=args.output_dir,
        tolerance=args.tolerance
    )

    if solution is not None:
        print(f"Solution converged: {solution.success}")
        print(f"Number of iterations: {solution.niter}")
    else:
        print("No solution found.")
    
    # Print parameters used
    print("\nParameters used:")
    print(f"  Delta: {args.delta}")
    print(f"  Ca: {args.ca}")
    print(f"  lambda_slip: {args.lambda_slip}")
    print(f"  mu_r: {args.mu_r}")
    print(f"  theta0: {args.theta0}° ({theta0_rad:.4f} rad)")
    print(f"  w: {args.w}")
    print(f"  tolerance: {args.tolerance}")

    if not args.gui:
        print(f"\nPlots saved to: {args.output_dir}/")