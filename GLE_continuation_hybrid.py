import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import root_scalar
from scipy.linalg import solve as lin_solve
import os
from functools import partial
import argparse
from multiprocessing import Pool, cpu_count
import time

# Import the critical Ca finding function from GLE_solver
from GLE_solver import find_critical_ca_lower_branch, solve_single_ca
# Import x0 finding utilities from src-local
import sys
sys.path.append('src-local')
from find_x0_utils import find_x0_and_theta_min

# Default parameters
DEFAULT_DELTA = 10.0  # Large domain for continuation
DEFAULT_LAMBDA_SLIP = 1e-4  # Slip length
DEFAULT_MU_R = 1e-6  # \mu_g/\mu_l
DEFAULT_THETA0 = np.pi/6  # theta at s = 0
DEFAULT_W = 0  # curvature boundary condition at s = \Delta
DEFAULT_NGRID = 10000  # Number of grid points
DEFAULT_WORKERS = min(4, cpu_count())  # Number of parallel workers

# Define f1, f2, and f3 functions needed for the GLE
def f1(theta):
    return theta**2 - np.sin(theta)**2

def f2(theta):
    return theta - np.sin(theta) * np.cos(theta)

def f3(theta):
    return theta * (np.pi - theta) + np.sin(theta)**2

# Define f(theta, mu_r) function
def f(theta, mu_r):
    numerator = 2 * np.sin(theta)**3 * (mu_r**2 * f1(theta) + 2 * mu_r * f3(theta) + f1(np.pi - theta))
    denominator = 3 * (mu_r * f1(theta) * f2(np.pi - theta) - f1(np.pi - theta) * f2(theta))
    epsilon = 1e-10
    denominator = np.where(np.abs(denominator) < epsilon, 
                          np.sign(denominator) * epsilon + (denominator == 0) * epsilon, 
                          denominator)
    return numerator / denominator

# Define the coupled ODEs system
def GLE(s, y, Ca, mu_r, lambda_slip):
    h, theta, omega = y
    dh_ds = np.sin(theta)
    dt_ds = omega
    dw_ds = - 3 * Ca * f(theta, mu_r) / (h * (h + 3 * lambda_slip)) - np.cos(theta)
    return [dh_ds, dt_ds, dw_ds]


# Boundary conditions
def boundary_conditions(ya, yb, w_bc, theta0, lambda_slip):
    h_a, theta_a, w_a = ya
    h_b, theta_b, w_b = yb
    return [
        theta_a - theta0,
        h_a - lambda_slip,
        w_b - w_bc
    ]

def interpolate_solution_to_mesh(solution_from, s_to):
    """
    Interpolate a solution from one mesh to another.
    
    Args:
        solution_from: Solution object with .x and .y attributes
        s_to: Target mesh points
        
    Returns:
        Interpolated y values on the new mesh
    """
    y_interp = np.zeros((3, len(s_to)))
    for i in range(3):
        # Use cubic interpolation for smooth results
        f_interp = interp1d(solution_from.x, solution_from.y[i, :], 
                          kind='cubic', bounds_error=False, 
                          fill_value='extrapolate')
        y_interp[i, :] = f_interp(s_to)
    return y_interp

def solve_bvp_for_ca(Ca, mu_r, lambda_slip, theta0, w_bc, Delta, s_range=None, y_guess=None, tol=1e-6):
    """
    Solve BVP for given Ca and return x0, theta_min, and solution
    """
    if s_range is None:
        s_range = np.linspace(0, Delta, DEFAULT_NGRID)
    
    if y_guess is None:
        y_guess = np.zeros((3, len(s_range)))
        y_guess[0, :] = np.linspace(lambda_slip, Delta, len(s_range))
        y_guess[1, :] = theta0
        y_guess[2, :] = 0
    
    GLE_with_params = partial(GLE, Ca=Ca, mu_r=mu_r, lambda_slip=lambda_slip)
    bc_with_params = partial(boundary_conditions, w_bc=w_bc, theta0=theta0, lambda_slip=lambda_slip)
    
    try:
        solution = solve_bvp(GLE_with_params, bc_with_params, s_range, y_guess,
                           max_nodes=1000000, tol=tol, verbose=0)
        
        if solution.success:
            h_vals, theta_vals, w_vals = solution.y
            
            # Check for valid solution
            if np.any(theta_vals < 0) or np.any(theta_vals > np.pi):
                return None, None, None
            
            # Find x0 (position where theta is minimum) and theta_min
            x0, theta_min, x0_idx = find_x0_and_theta_min(solution.x, theta_vals)
            
            return x0, theta_min, solution
        else:
            return None, None, None
    except Exception as e:
        return None, None, None

def hybrid_iqi_newton_refinement(Ca_good, Ca_fail, mu_r, lambda_slip, theta0, w_bc, Delta,
                                s_range, y_guess, tolerance=1e-6, max_iter=30):
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
    print(f"\nRefining Ca_cr between {Ca_good:.6f} and {Ca_fail:.6f}")
    print(f"Target tolerance: {tolerance}")
    
    # Store history of (Ca, theta_min) pairs for IQI
    history_Ca = []
    history_theta_min = []
    
    # Add initial good point
    x0, theta_min, solution = solve_bvp_for_ca(Ca_good, mu_r, lambda_slip, theta0, w_bc, Delta, s_range, y_guess)
    if theta_min is not None:
        history_Ca.append(Ca_good)
        history_theta_min.append(theta_min)
        s_range = solution.x
        y_guess = solution.y
    
    for iter_count in range(max_iter):
        interval_width = Ca_fail - Ca_good
        
        # Check convergence
        if interval_width < tolerance:
            print(f"Converged to Ca_cr = {Ca_good:.6f} after {iter_count} iterations")
            break
        
        # Choose method based on available data and proximity to critical point
        if len(history_Ca) >= 3 and min(history_theta_min) > 0.001:
            # Use Inverse Quadratic Interpolation
            # We want to find Ca where theta_min = 0
            # Fit quadratic through last 3 points
            n = len(history_Ca)
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
                x0_test, theta_min_test, _ = solve_bvp_for_ca(
                    Ca_test, mu_r, lambda_slip, theta0, w_bc, Delta, s_range, y_guess
                )
                if theta_min_test is not None:
                    dtheta_dCa = (theta_min_test - history_theta_min[-1]) / h
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
        x0_new, theta_min_new, solution_new = solve_bvp_for_ca(
            Ca_new, mu_r, lambda_slip, theta0, w_bc, Delta, s_range, y_guess
        )
        
        if theta_min_new is not None:
            # Success - update bounds and history
            Ca_good = Ca_new
            s_range = solution_new.x
            y_guess = solution_new.y
            
            history_Ca.append(Ca_new)
            history_theta_min.append(theta_min_new)
            
            # Keep only recent history for IQI
            if len(history_Ca) > 5:
                history_Ca = history_Ca[-5:]
                history_theta_min = history_theta_min[-5:]
            
            print(f"  Iteration {iter_count+1} ({method}): Ca = {Ca_new:.6f}, θ_min = {theta_min_new*180/np.pi:.2f}°")
            
            # Early termination if theta_min is very small
            if theta_min_new < 0.001:
                print(f"θ_min < 0.001 rad, terminating early at Ca = {Ca_new:.6f}")
                Ca_good = Ca_new
                break
        else:
            # Failed - update upper bound
            Ca_fail = Ca_new
            print(f"  Iteration {iter_count+1} ({method}): Ca = {Ca_new:.6f} failed")
    
    return Ca_good


def inverse_quadratic_interpolation(x, y):
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


def solve_for_theta_min_newton(theta_min_target, Ca_guess, mu_r, lambda_slip, theta0, w_bc, Delta, 
                              Ca_spline=None, solution_cache=None):
    """
    Find Ca for target theta_min using Newton's method with smart initial guess
    """
    # Get initial Ca guess from spline if available
    if Ca_spline is not None:
        try:
            Ca_init = float(Ca_spline(theta_min_target))
            Ca_init = max(1e-6, min(1.0, Ca_init))
        except:
            Ca_init = Ca_guess
    else:
        Ca_init = Ca_guess
    
    # Get initial solution guess from cache if available
    s_range = None
    y_guess = None
    if solution_cache is not None and len(solution_cache) > 0:
        # Find closest theta_min in cache
        theta_min_cache = [item['theta_min'] for item in solution_cache]
        closest_idx = np.argmin(np.abs(np.array(theta_min_cache) - theta_min_target))
        s_range = solution_cache[closest_idx]['s_range']
        y_guess = solution_cache[closest_idx]['y_solution']
    
    def objective(Ca):
        """Return theta_min(Ca) - theta_min_target"""
        if Ca <= 0 or Ca > 1.0:
            return 1e10
        
        x0_computed, theta_min, solution = solve_bvp_for_ca(
            Ca, mu_r, lambda_slip, theta0, w_bc, Delta, s_range, y_guess
        )
        
        if theta_min is not None:
            return theta_min - theta_min_target
        else:
            return 1e10
    
    # Newton iterations with finite differences for derivative
    Ca = Ca_init
    h = 1e-4
    max_iter = 10
    tol = 1e-8
    
    for i in range(max_iter):
        f_val = objective(Ca)
        if abs(f_val) < tol:
            break
        
        f_plus = objective(Ca * (1 + h))
        df_dCa = (f_plus - f_val) / (Ca * h)
        
        if abs(df_dCa) > 1e-10:
            delta_Ca = -f_val / df_dCa
            damping = 0.7 if abs(delta_Ca/Ca) > 0.5 else 1.0
            Ca_new = Ca + damping * delta_Ca
            Ca_new = max(1e-6, min(1.0, Ca_new))
            
            if abs(Ca_new - Ca) / Ca < 1e-6:
                break
            Ca = Ca_new
        else:
            break
    
    # Get final solution
    x0_final, theta_min_final, solution = solve_bvp_for_ca(
        Ca, mu_r, lambda_slip, theta0, w_bc, Delta, s_range, y_guess, tol=1e-6
    )
    
    if theta_min_final is not None and abs(theta_min_final - theta_min_target) < 0.001:
        return Ca, theta_min_final, solution
    else:
        return None, None, None

def worker_solve_theta_min(args):
    """Worker function for parallel processing with improved error handling"""
    theta_min_target, Ca_guess, mu_r, lambda_slip, theta0, w_bc, Delta, Ca_spline_data = args
    
    # Reconstruct spline in worker
    if Ca_spline_data is not None:
        theta_min_data, Ca_data = Ca_spline_data
        try:
            Ca_spline = UnivariateSpline(theta_min_data, Ca_data, k=min(3, len(theta_min_data)-1), s=0)
        except:
            Ca_spline = None
    else:
        Ca_spline = None
    
    # Try with Newton method first
    Ca, theta_min, solution = solve_for_theta_min_newton(
        theta_min_target, Ca_guess, mu_r, lambda_slip, theta0, w_bc, Delta, Ca_spline
    )
    
    # If Newton fails, try bracketing method
    if Ca is None and Ca_spline is not None:
        try:
            # Bracket search around initial guess
            Ca_lower = Ca_guess * 0.1
            Ca_upper = Ca_guess * 10.0
            
            def objective(Ca):
                _, theta_min_comp, _ = solve_bvp_for_ca(
                    Ca, mu_r, lambda_slip, theta0, w_bc, Delta
                )
                return (theta_min_comp - theta_min_target) if theta_min_comp is not None else 1e10
            
            # Check if we can bracket
            f_lower = objective(Ca_lower)
            f_upper = objective(Ca_upper)
            
            if f_lower * f_upper < 0:
                from scipy.optimize import brentq
                Ca = brentq(objective, Ca_lower, Ca_upper, xtol=1e-8)
                _, theta_min, solution = solve_bvp_for_ca(
                    Ca, mu_r, lambda_slip, theta0, w_bc, Delta
                )
        except:
            pass
    
    if Ca is not None and solution is not None:
        x0_computed, _, _ = solve_bvp_for_ca(
            Ca, mu_r, lambda_slip, theta0, w_bc, Delta, solution.x, solution.y
        )
        return {
            'x0': x0_computed,  # Track x0 (position at theta_min) for plotting
            'Ca': Ca,
            'theta_min': theta_min,
            's_range': solution.x,
            'y_solution': solution.y
        }
    else:
        return None


def find_critical_ca_improved(mu_r, lambda_slip, theta0, w_bc, Delta, nGridInit=DEFAULT_NGRID, 
                             output_dir='output', Ca_requested=1.0, tolerance=1e-6):
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
    
    # Initial guess
    s_range = np.linspace(0, Delta, nGridInit)
    y_guess = np.zeros((3, s_range.size))
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
            print(f"  Testing Ca = {Ca:.6f}")
        
        x0, theta_min, solution = solve_bvp_for_ca(
            Ca, mu_r, lambda_slip, theta0, w_bc, Delta, s_range, y_guess
        )
        
        if solution is not None:
            Ca_critical = Ca
            Ca_values.append(Ca)
            theta_min_values.append(theta_min)
            s_range = solution.x
            y_guess = solution.y
            x0_values.append(x0)
            
            if theta_min < 0.1:
                print(f"  theta_min approaching zero at Ca = {Ca:.6f} " +
                      f"(θ_min = {theta_min*180/np.pi:.2f}°, x0 = {x0:.4f})")
        else:
            print(f"  Solution failed at Ca = {Ca:.6f}")
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
        print(f"\nNo failure found up to Ca = {Ca_coarse[-1]:.6f}")
    
    print(f"\nFinal critical Ca: Ca_cr = {Ca_critical:.6f}")
    
    return Ca_critical, Ca_values, x0_values, theta_min_values


class PseudoArclengthContinuation:
    """
    Pseudo-arclength continuation for tracing through turning points
    """
    def __init__(self, mu_r, lambda_slip, theta0, w_bc, Delta):
        self.mu_r = mu_r
        self.lambda_slip = lambda_slip
        self.theta0 = theta0
        self.w_bc = w_bc
        self.Delta = Delta
        self.previous_Ca = []  # Track Ca history for direction choice
    
    def compute_tangent(self, Ca, solution, dCa=1e-5):
        """
        Compute tangent vector (dU/ds, dCa/ds) using finite differences
        """
        # Current solution
        x0_current, theta_min_current, _ = solve_bvp_for_ca(
            Ca, self.mu_r, self.lambda_slip, self.theta0, self.w_bc, self.Delta,
            solution.x, solution.y, tol=1e-6
        )
        
        x0_plus, theta_min_plus, solution_plus = solve_bvp_for_ca(
            Ca + dCa, self.mu_r, self.lambda_slip, self.theta0, self.w_bc, self.Delta,
            solution.x, solution.y, tol=1e-6
        )
        
        if solution_plus is None:
            dCa = dCa / 10
            x0_plus, theta_min_plus, solution_plus = solve_bvp_for_ca(
                Ca + dCa, self.mu_r, self.lambda_slip, self.theta0, self.w_bc, self.Delta,
                solution.x, solution.y, tol=1e-6
            )
            if solution_plus is None:
                return None
        
        # Interpolate solution_plus to the same mesh as solution if needed
        if len(solution_plus.x) != len(solution.x):
            y_plus_interp = interpolate_solution_to_mesh(solution_plus, solution.x)
            dU_dCa = (y_plus_interp - solution.y) / dCa
        else:
            # Compute derivative of full solution vector
            dU_dCa = (solution_plus.y - solution.y) / dCa
        
        # Normalize to get tangent vector
        dU_dCa_norm = np.linalg.norm(dU_dCa.flatten())
        norm = np.sqrt(dU_dCa_norm**2 + 1)  # Include Ca component
        
        dU_ds = dU_dCa / norm
        dCa_ds = 1.0 / norm
        
        # Choose sign to maintain continuity (trace upward on upper branch)
        if hasattr(self, 'previous_Ca') and len(self.previous_Ca) > 1:
            Ca_diff = Ca - self.previous_Ca[-2]
            if Ca_diff * dCa_ds < 0:  # Sign change at turning point
                dU_ds = -dU_ds
                dCa_ds = -dCa_ds
        
        return dU_ds, dCa_ds
    
    def predictor_corrector_step(self, Ca_old, solution_old, x0_old, theta_min_old, ds):
        """
        Take one predictor-corrector step using pseudo-arclength continuation
        """
        # Compute tangent
        tangent = self.compute_tangent(Ca_old, solution_old)
        if tangent is None:
            return None
        
        dU_ds, dCa_ds = tangent
        
        # Predictor step
        Ca_pred = Ca_old + ds * dCa_ds
        y_pred = solution_old.y + ds * dU_ds
        
        # Corrector step: Newton iterations to satisfy perpendicular constraint
        Ca = Ca_pred
        s_range = solution_old.x
        y_guess = y_pred
        
        max_iter = 10
        for iter_count in range(max_iter):
            # Solve BVP
            x0_new, theta_min_new, solution_new = solve_bvp_for_ca(
                Ca, self.mu_r, self.lambda_slip, self.theta0, self.w_bc, self.Delta,
                s_range, y_guess, tol=1e-6
            )
            
            if solution_new is None:
                return None
            
            # Check perpendicular constraint
            # N = (U - U_old)·(dU/ds)_old + (Ca - Ca_old)·(dCa/ds)_old - ds = 0
            # Interpolate if meshes are different
            if len(solution_new.x) != len(solution_old.x):
                y_new_interp = interpolate_solution_to_mesh(solution_new, solution_old.x)
                U_diff = (y_new_interp - solution_old.y).flatten()
            else:
                U_diff = (solution_new.y - solution_old.y).flatten()
            dU_ds_flat = dU_ds.flatten()
            
            N = np.dot(U_diff, dU_ds_flat) + (Ca - Ca_old) * dCa_ds - ds
            
            if abs(N) < 1e-8:
                self.previous_Ca.append(Ca)
                # Update s_range for next iteration if mesh changed
                s_range = solution_new.x
                return Ca, solution_new, x0_new, theta_min_new
            
            # Newton update for Ca
            dCa_newton = 1e-5
            Ca_pert = Ca + dCa_newton
            
            x0_pert, theta_min_pert, solution_pert = solve_bvp_for_ca(
                Ca_pert, self.mu_r, self.lambda_slip, self.theta0, self.w_bc, self.Delta,
                solution_new.x, solution_new.y, tol=1e-6
            )
            
            if solution_pert is None:
                dCa_newton = 1e-6
                Ca_pert = Ca + dCa_newton
                x0_pert, theta_min_pert, solution_pert = solve_bvp_for_ca(
                    Ca_pert, self.mu_r, self.lambda_slip, self.theta0, self.w_bc, self.Delta,
                    solution_new.x, solution_new.y, tol=1e-6
                )
                if solution_pert is None:
                    return None
            
            # Compute dN/dCa
            # Interpolate if meshes are different
            if len(solution_pert.x) != len(solution_old.x):
                y_pert_interp = interpolate_solution_to_mesh(solution_pert, solution_old.x)
                U_diff_pert = (y_pert_interp - solution_old.y).flatten()
            else:
                U_diff_pert = (solution_pert.y - solution_old.y).flatten()
            N_pert = np.dot(U_diff_pert, dU_ds_flat) + (Ca_pert - Ca_old) * dCa_ds - ds
            dN_dCa = (N_pert - N) / dCa_newton
            
            if abs(dN_dCa) > 1e-10:
                # Newton correction
                delta_Ca = -N / dN_dCa
                
                # Damping for stability
                if abs(delta_Ca) > 0.1 * abs(Ca):
                    delta_Ca = 0.1 * abs(Ca) * np.sign(delta_Ca)
                
                Ca = Ca + delta_Ca
                Ca = max(1e-6, min(1.0, Ca))  # Keep in bounds
                
                # Update guess and mesh
                y_guess = solution_new.y
                s_range = solution_new.x
            else:
                break
        
        return None

def trace_both_branches_hybrid(mu_r, lambda_slip, theta0=DEFAULT_THETA0, w_bc=DEFAULT_W,
                              Delta=DEFAULT_DELTA, output_dir='output', n_workers=DEFAULT_WORKERS):
    """
    Three-phase hybrid approach:
    Phase 0: Use GLE_solver to find approximate Ca_cr efficiently
    Phase 1: x0 parameterization for detailed lower branch
    Phase 2: pseudo-arclength for upper branch
    """
    print(f"\nTHREE-PHASE HYBRID SOLVER: Optimized approach")
    print(f"Parameters: mu_r = {mu_r}, lambda_slip = {lambda_slip}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Phase 0: Find approximate Ca_cr using improved method
    print("\nPhase 0: Finding approximate Ca_cr using improved IQI + Newton method...")
    start_time_phase0 = time.time()
    
    # First, solve at Ca=0 to get baseline x0 (position at theta_min)
    print("  Solving at Ca=0 to establish baseline...")
    s_range_ca0 = np.linspace(0, Delta, DEFAULT_NGRID)
    y_guess_ca0 = np.zeros((3, len(s_range_ca0)))
    y_guess_ca0[0, :] = np.linspace(lambda_slip, Delta, len(s_range_ca0))
    y_guess_ca0[1, :] = theta0
    y_guess_ca0[2, :] = 0
    
    result_ca0 = solve_single_ca(
        Ca=0.0, mu_r=mu_r, lambda_slip=lambda_slip, theta0=theta0, 
        w_bc=w_bc, Delta=Delta, s_range=s_range_ca0, y_guess=y_guess_ca0
    )
    
    x0_ca_zero = None
    if result_ca0.success:
        x0_ca_zero = result_ca0.x0
        if x0_ca_zero is not None:
            print(f"  Ca=0 solution: x0 = {x0_ca_zero:.4f} (at theta_min = {result_ca0.theta_min*180/np.pi:.2f}°)")
    
    # Use our improved method with IQI + Newton refinement
    Ca_cr_approx, Ca_values_phase0, x0_values_phase0, theta_min_values_phase0 = find_critical_ca_improved(
        mu_r, lambda_slip, theta0, w_bc, Delta, 
        nGridInit=DEFAULT_NGRID, 
        output_dir=output_dir,
        Ca_requested=1.0,  # High value to ensure we find Ca_cr
        tolerance=1e-3  # Coarse tolerance for speed
    )
    
    # Add Ca=0 result to phase 0 data if successful and x0 exists
    if result_ca0.success and x0_ca_zero is not None:
        Ca_values_phase0 = [0.0] + Ca_values_phase0
        x0_values_phase0 = [x0_ca_zero] + x0_values_phase0
        theta_min_values_phase0 = [result_ca0.theta_min] + theta_min_values_phase0
    
    phase0_time = time.time() - start_time_phase0
    print(f"Phase 0 completed in {phase0_time:.1f} seconds")
    print(f"Approximate Ca_cr = {Ca_cr_approx:.4f}")
    
    # Get the last successful solution from Phase 0 for initial guess
    if Ca_values_phase0:
        # Find the solution closest to Ca_cr_approx
        idx_closest = np.argmin(np.abs(np.array(Ca_values_phase0) - Ca_cr_approx))
        Ca_init = Ca_values_phase0[idx_closest]
        x0_init = x0_values_phase0[idx_closest]
        theta_min_init = theta_min_values_phase0[idx_closest]
        
        # Get the actual solution for initial guess
        s_range_init = np.linspace(0, Delta, DEFAULT_NGRID)
        y_guess_init = np.zeros((3, len(s_range_init)))
        y_guess_init[0, :] = np.linspace(lambda_slip, Delta, len(s_range_init))
        y_guess_init[1, :] = theta0
        y_guess_init[2, :] = 0
        
        # Solve at Ca_init to get proper initial guess
        x0_temp, theta_min_temp, solution_init = solve_bvp_for_ca(
            Ca_init, mu_r, lambda_slip, theta0, w_bc, Delta, s_range_init, y_guess_init
        )
        if solution_init:
            s_range_init = solution_init.x
            y_guess_init = solution_init.y
    else:
        print("Warning: Phase 0 did not find any solutions. Starting from scratch.")
        Ca_cr_approx = 0.1  # Fallback
        x0_init = 1.0
        s_range_init = np.linspace(0, Delta, DEFAULT_NGRID)
        y_guess_init = np.zeros((3, len(s_range_init)))
        y_guess_init[0, :] = np.linspace(lambda_slip, Delta, len(s_range_init))
        y_guess_init[1, :] = theta0
        y_guess_init[2, :] = 0
    
    # Phase 1: Use theta_min parameterization to capture detailed lower branch
    print("\nPhase 1: Capturing detailed lower branch with theta_min parameterization...")
    start_time = time.time()
    
    # Initial exploration focused around Ca_cr_approx
    # Create a denser grid around the approximate critical Ca
    Ca_min = Ca_cr_approx * 0.01  # Start well below
    Ca_max = Ca_cr_approx * 1.2   # Go slightly above
    
    # Logarithmic spacing below Ca_cr and linear near Ca_cr
    Ca_log = np.logspace(np.log10(Ca_min), np.log10(Ca_cr_approx * 0.8), 15)
    Ca_lin = np.linspace(Ca_cr_approx * 0.8, Ca_max, 15)
    Ca_initial = np.unique(np.concatenate([Ca_log, Ca_lin]))
    
    x0_initial = []
    Ca_found_initial = []
    theta_min_initial = []
    solutions_initial = []
    
    # Use the initial guess from Phase 0
    s_range = s_range_init
    y_guess = y_guess_init
    
    for i, Ca in enumerate(Ca_initial):
        x0, theta_min, solution = solve_bvp_for_ca(Ca, mu_r, lambda_slip, theta0, w_bc, Delta, s_range, y_guess)
        
        if x0 is not None:
            # Only include solutions where theta reaches 90°
            x0_initial.append(x0)
            Ca_found_initial.append(Ca)
            theta_min_initial.append(theta_min)
            solutions_initial.append({
                'x0': x0,
                'Ca': Ca,
                's_range': solution.x,
                'y_solution': solution.y
            })
            
            s_range = solution.x
            y_guess = solution.y
            
            if len(x0_initial) % 5 == 0:
                if x0 is not None:
                    print(f"Ca = {Ca:.6f}, x0 = {x0:.4f}, theta_min = {theta_min*180/np.pi:.2f}°")
                else:
                    print(f"Ca = {Ca:.6f}, theta_max = {theta_min*180/np.pi:.2f}° < 90° (no x0)")
        else:
            # We've likely passed Ca_cr
            if Ca > Ca_cr_approx * 0.9:
                print(f"Solution failed at Ca = {Ca:.6f} (near expected Ca_cr)")
                break
    
    if len(x0_initial) < 3:
        print("Failed to find enough initial solutions!")
        return None, None, None
    
    # Build interpolating spline
    x0_arr = np.array(x0_initial)
    Ca_arr = np.array(Ca_found_initial)
    sort_idx = np.argsort(x0_arr)
    x0_sorted = x0_arr[sort_idx]
    Ca_sorted = Ca_arr[sort_idx]
    Ca_spline_data = (x0_sorted, Ca_sorted)
    
    # Since we already have Ca_cr_approx from Phase 0, we can identify the turning point more accurately
    # Find the solution closest to Ca_cr_approx
    Ca_diffs = np.abs(Ca_sorted - Ca_cr_approx)
    turning_region_idx = np.argmin(Ca_diffs)
    
    x0_turning_approx = x0_sorted[turning_region_idx]
    Ca_turning_approx = Ca_sorted[turning_region_idx]
    
    print(f"\nTurning point from Phase 0 knowledge:")
    print(f"  Ca ≈ {Ca_turning_approx:.6f}, x0 ≈ {x0_turning_approx:.4f}")
    
    # Do a quick refinement if we're not very close
    if abs(Ca_turning_approx - Ca_cr_approx) > 0.001:
        print(f"\nRefining turning point estimate...")
        Ca_test_max = Ca_turning_approx
        Ca_test_fail = min(Ca_turning_approx * 1.1, Ca_cr_approx * 1.5)
        
        # Quick bisection (fewer iterations needed since we start close)
        for _ in range(3):
            Ca_mid = (Ca_test_max + Ca_test_fail) / 2
            x0_mid, _, sol_mid = solve_bvp_for_ca(
                Ca_mid, mu_r, lambda_slip, theta0, w_bc, Delta,
                solutions_initial[-1]['s_range'], solutions_initial[-1]['y_solution']
            )
            if sol_mid is not None:
                Ca_test_max = Ca_mid
                x0_turning_approx = x0_mid
                Ca_turning_approx = Ca_mid
            else:
                Ca_test_fail = Ca_mid
        
        print(f"  Refined turning point: Ca ≈ {Ca_turning_approx:.6f}, x0 ≈ {x0_turning_approx:.4f}")
    
    # Build theta_min interpolating spline from initial exploration
    theta_min_arr = np.array(theta_min_initial)
    Ca_arr = np.array(Ca_found_initial)
    sort_idx = np.argsort(theta_min_arr)
    theta_min_sorted = theta_min_arr[sort_idx]
    Ca_sorted_by_theta = Ca_arr[sort_idx]
    
    # Find turning point based on theta_min approaching zero
    theta_min_turning_approx = theta_min_sorted[0]  # Smallest theta_min
    Ca_turning_approx_theta = Ca_sorted_by_theta[0]
    
    print(f"\nTurning point from theta_min analysis:")
    print(f"  Ca ≈ {Ca_turning_approx_theta:.6f}, theta_min ≈ {theta_min_turning_approx*180/np.pi:.2f}°")
    
    # Dense theta_min sweep focused on the region near turning point
    # For Ca=0, theta_min should be close to theta0
    theta_min_ca_zero = theta0 if result_ca0.success else theta0
    if result_ca0.success:
        theta_min_ca_zero = result_ca0.theta_min
        print(f"\nUsing theta_min from Ca=0 solution: {theta_min_ca_zero*180/np.pi:.2f}°")
    
    # Create theta_min range from near theta0 down to near zero
    theta_min_max = min(theta_min_ca_zero * 0.95, max(theta_min_initial))
    theta_min_min = max(theta_min_turning_approx * 0.1, 0.001)  # Don't go exactly to zero
    
    print(f"Phase 1 theta_min range: [{theta_min_min*180/np.pi:.2f}°, {theta_min_max*180/np.pi:.2f}°]")
    
    # Create adaptive grid with most points near turning point (small theta_min)
    theta_min_grid1 = np.linspace(theta_min_max, theta_min_turning_approx * 2, 20)
    theta_min_grid2 = np.linspace(theta_min_turning_approx * 2, theta_min_turning_approx * 0.5, 30)  # Dense near turning
    theta_min_grid3 = np.linspace(theta_min_turning_approx * 0.5, theta_min_min, 10)
    theta_min_test_lower = np.unique(np.concatenate([theta_min_grid1, theta_min_grid2, theta_min_grid3]))
    
    print(f"\nParallel theta_min sweep for lower branch ({len(theta_min_test_lower)} points)...")
    print(f"  theta_min range: [{theta_min_min*180/np.pi:.2f}°, {theta_min_max*180/np.pi:.2f}°]")
    
    # Build spline for initial guesses
    Ca_spline_data = (theta_min_sorted, Ca_sorted_by_theta)
    Ca_spline = UnivariateSpline(theta_min_sorted, Ca_sorted_by_theta, k=min(3, len(theta_min_sorted)-1), s=0)
    
    args_list = []
    for theta_min_target in theta_min_test_lower:
        try:
            Ca_guess = float(Ca_spline(theta_min_target))
            Ca_guess = max(1e-6, min(1.0, Ca_guess))
        except:
            # Extrapolation fallback
            if theta_min_target > theta_min_sorted[-1]:
                Ca_guess = Ca_sorted_by_theta[-1] * (theta_min_target / theta_min_sorted[-1])**2
            else:
                Ca_guess = Ca_sorted_by_theta[0]
        
        args_list.append((theta_min_target, Ca_guess, mu_r, lambda_slip, theta0, w_bc, Delta, Ca_spline_data))
    
    # Parallel execution with progress tracking
    lower_branch_results = []
    failed_count = 0
    
    with Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_solve_theta_min, args_list)):
            if result is not None:
                lower_branch_results.append(result)
            else:
                failed_count += 1
            
            # Progress update
            if (i + 1) % 20 == 0:
                success_rate = len(lower_branch_results) / (i + 1) * 100
                print(f"  Progress: {i+1}/{len(theta_min_test_lower)} points, "
                      f"success rate: {success_rate:.1f}%")
    
    phase1_time = time.time() - start_time
    print(f"Phase 1 completed in {phase1_time:.1f} seconds")
    print(f"Found {len(lower_branch_results)} points on lower branch")
    
    # Find the point closest to turning point
    all_Ca = [r['Ca'] for r in lower_branch_results]
    max_Ca_idx = np.argmax(all_Ca)
    turning_point_approx = lower_branch_results[max_Ca_idx]
    
    Ca_turn = turning_point_approx['Ca']
    x0_turn = turning_point_approx['x0']
    theta_min_turn = turning_point_approx['theta_min']
    solution_turn = solve_bvp_for_ca(Ca_turn, mu_r, lambda_slip, theta0, w_bc, Delta,
                                     turning_point_approx['s_range'], 
                                     turning_point_approx['y_solution'])[2]
    
    print(f"\nTurning point found:")
    print(f"  Ca_turn = {Ca_turn:.6f}")
    print(f"  x0_turn = {x0_turn:.4f}")
    print(f"  theta_min_turn = {theta_min_turn*180/np.pi:.2f}°")
    
    # Validate turning point
    if len(lower_branch_results) < 10:
        print("\nWarning: Few points found on lower branch. Results may be incomplete.")
    
    # Check if we actually found the turning point
    Ca_values_lower = [r['Ca'] for r in lower_branch_results]
    if max(Ca_values_lower) < 0.9 * Ca_turn:
        print("\nWarning: Turning point may not be accurate. Consider increasing x0 range.")
    
    # Phase 2: Use pseudo-arclength to trace through turning point and capture upper branch
    print("\nPhase 2: Tracing through turning point and capturing upper branch with pseudo-arclength...")
    start_time = time.time()
    
    arc_cont = PseudoArclengthContinuation(mu_r, lambda_slip, theta0, w_bc, Delta)
    
    # Store upper branch points
    upper_branch_Ca = [Ca_turn]
    upper_branch_x0 = [x0_turn]
    upper_branch_theta_min = [theta_min_turn]
    
    # Initialize arc-length continuation tracking
    arc_cont.previous_Ca = [Ca_turn]  # Initialize with turning point
    
    # Trace with pseudo-arclength
    Ca_current = Ca_turn
    solution_current = solution_turn
    x0_current = x0_turn
    theta_min_current = theta_min_turn
    
    ds = 0.01  # Initial step size
    ds_min = 1e-4
    ds_max = 0.1
    max_steps = 150
    failed_steps = 0
    max_failed_steps = 10
    
    for step in range(max_steps):
        result = arc_cont.predictor_corrector_step(
            Ca_current, solution_current, x0_current, theta_min_current, ds
        )
        
        if result is None:
            # Reduce step size and try again
            failed_steps += 1
            ds = ds * 0.5
            if abs(ds) < ds_min or failed_steps > max_failed_steps:
                print(f"\nStopping: {'step size too small' if abs(ds) < ds_min else 'too many failed steps'}")
                print(f"  Final step: {step}, ds = {ds:.6f}")
                break
            continue
        
        Ca_new, solution_new, x0_new, theta_min_new = result
        
        # Reset failed steps counter on success
        failed_steps = 0
        
        # Check stopping criteria
        if Ca_new < Ca_turn * 0.05:  # Traced to very low Ca
            print(f"\nReached Ca < 5% of turning point at step {step}")
            break
        
        if x0_new > x0_turn * 5:  # Very large x0
            print(f"\nReached x0 > 5× turning point value at step {step}")
            break
        
        # Store new point
        upper_branch_Ca.append(Ca_new)
        upper_branch_x0.append(x0_new)
        upper_branch_theta_min.append(theta_min_new)
        
        # Calculate step quality metrics
        step_distance = np.sqrt((Ca_new - Ca_current)**2 + (x0_new - x0_current)**2)
        
        # Update for next step
        Ca_current = Ca_new
        solution_current = solution_new
        x0_current = x0_new
        theta_min_current = theta_min_new
        
        # Adaptive step size based on success
        if step_distance > 0.8 * abs(ds) and step_distance < 1.2 * abs(ds):
            # Good step - increase step size
            ds = min(ds * 1.5, ds_max)
        elif step_distance < 0.5 * abs(ds):
            # Step too small - increase step size
            ds = min(ds * 2.0, ds_max)
        # Keep current ds if step_distance is reasonable
        
        if (step + 1) % 10 == 0:
            print(f"Step {step+1}: Ca = {Ca_new:.6f}, x0 = {x0_new:.4f}, "
                  f"θ_min = {theta_min_new*180/np.pi:.2f}°, ds = {ds:.5f}")
    
    phase2_time = time.time() - start_time
    print(f"\nPhase 2 completed in {phase2_time:.1f} seconds")
    print(f"Found {len(upper_branch_Ca)-1} additional points on upper branch")
    
    # Combine results from all phases with branch tracking
    all_results = []
    branch_labels = []
    phase_labels = []  # Track which phase found each point
    
    # Add Phase 0 results (coarse exploration)
    if Ca_values_phase0:
        for i in range(len(Ca_values_phase0)):
            all_results.append({
                'Ca': Ca_values_phase0[i],
                'x0': x0_values_phase0[i],
                'theta_min': theta_min_values_phase0[i]
            })
            branch_labels.append('lower')
            phase_labels.append('phase0')
    
    # Add Phase 1 results (detailed lower branch)
    for r in lower_branch_results:
        # Check if this point is already in results (avoid duplicates)
        is_duplicate = False
        for existing in all_results:
            if abs(existing['Ca'] - r['Ca']) < 1e-8:
                is_duplicate = True
                break
        
        if not is_duplicate:
            all_results.append(r)
            branch_labels.append('lower')
            phase_labels.append('phase1')
    
    # Add Phase 2 results (upper branch points, excluding the turning point)
    for i in range(1, len(upper_branch_Ca)):
        all_results.append({
            'Ca': upper_branch_Ca[i],
            'x0': upper_branch_x0[i],
            'theta_min': upper_branch_theta_min[i]
        })
        branch_labels.append('upper')
        phase_labels.append('phase2')
    
    # Extract arrays
    Ca_all = np.array([r['Ca'] for r in all_results])
    x0_all = np.array([r['x0'] for r in all_results])
    theta_min_all = np.array([r['theta_min'] for r in all_results])
    branch_all = np.array(branch_labels)
    phase_all = np.array(phase_labels)
    
    # Sort by Ca while keeping track of branches and phases
    sort_idx = np.argsort(Ca_all)
    Ca_sorted = Ca_all[sort_idx]
    x0_sorted = x0_all[sort_idx]
    theta_min_sorted = theta_min_all[sort_idx]
    branch_sorted = branch_all[sort_idx]
    phase_sorted = phase_all[sort_idx]
    
    # Find critical point (maximum Ca)
    Ca_max_idx = np.argmax(Ca_sorted)
    Ca_critical = Ca_sorted[Ca_max_idx]
    x0_critical = x0_sorted[Ca_max_idx]
    theta_min_critical = theta_min_sorted[Ca_max_idx]
    
    # Verify we have both branches
    lower_branch_count = np.sum(branch_sorted == 'lower')
    upper_branch_count = np.sum(branch_sorted == 'upper')
    
    print(f"\nBranch statistics:")
    print(f"  Lower branch points: {lower_branch_count}")
    print(f"  Upper branch points: {upper_branch_count}")
    
    # Phase statistics
    phase0_count = np.sum(phase_sorted == 'phase0')
    phase1_count = np.sum(phase_sorted == 'phase1')
    phase2_count = np.sum(phase_sorted == 'phase2')
    print(f"\nPhase contributions:")
    print(f"  Phase 0 points: {phase0_count}")
    print(f"  Phase 1 points: {phase1_count}")
    print(f"  Phase 2 points: {phase2_count}")
    
    if upper_branch_count < 5:
        print("\nWarning: Few points on upper branch. Consider adjusting continuation parameters.")
    
    print(f"\nCritical point:")
    print(f"  Ca_cr = {Ca_critical:.6f}")
    print(f"  x0_cr = {x0_critical:.4f}")
    print(f"  theta_min_cr = {theta_min_critical*180/np.pi:.2f}°")
    
    # Performance summary
    total_time = phase0_time + phase1_time + phase2_time
    print(f"\nPerformance Summary:")
    print(f"  Phase 0 (Ca_cr approximation):  {phase0_time:.1f}s")
    print(f"  Phase 1 (x0 parameterization):  {phase1_time:.1f}s")
    print(f"  Phase 2 (pseudo-arclength):     {phase2_time:.1f}s")
    print(f"  Total time:                     {total_time:.1f}s")
    print(f"\nEfficiency gain: Phase 0 provided Ca_cr ≈ {Ca_cr_approx:.4f}, " +
          f"allowing focused exploration")
    
    # Create plots
    plot_bifurcation_hybrid(Ca_sorted, x0_sorted, theta_min_sorted, branch_sorted, phase_sorted,
                           Ca_critical, x0_critical, theta_min_critical, mu_r, lambda_slip, output_dir)
    
    # Save data with branch and phase information
    # Convert labels to numeric
    branch_numeric = np.array([0 if b == 'lower' else 1 for b in branch_sorted])
    phase_numeric = np.array([0 if p == 'phase0' else 1 if p == 'phase1' else 2 for p in phase_sorted])
    csv_data = np.column_stack((Ca_sorted, x0_sorted, theta_min_sorted, theta_min_sorted*180/np.pi, 
                               branch_numeric, phase_numeric))
    csv_path = os.path.join(output_dir, 'both_branches_hybrid.csv')
    np.savetxt(csv_path, csv_data, delimiter=',',
               header='Ca,x0,theta_min_rad,theta_min_deg,branch(0=lower,1=upper),phase(0,1,2)', 
               comments='')
    print(f"\nData saved to: {csv_path}")
    
    return Ca_sorted, x0_sorted, theta_min_sorted

def plot_bifurcation_hybrid(Ca_sorted, x0_sorted, theta_min_sorted, branch_sorted, phase_sorted,
                           Ca_critical, x0_critical, theta_min_critical, mu_r, lambda_slip, output_dir):
    """Create bifurcation diagram plots with phase information"""
    # Separate branches based on labels
    lower_mask = branch_sorted == 'lower'
    upper_mask = branch_sorted == 'upper'
    
    # Separate phases for different markers
    phase0_mask = phase_sorted == 'phase0'
    phase1_mask = phase_sorted == 'phase1'
    phase2_mask = phase_sorted == 'phase2'
    
    # Check if we have valid x0 data
    has_x0_data = x0_sorted is not None and any(x is not None for x in x0_sorted)
    
    # Create plots
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: theta_min vs Ca with phase distinction (PRIMARY)
    theta_min_deg = theta_min_sorted * 180 / np.pi
    
    # Plot lines for branches
    ax1.plot(Ca_sorted[lower_mask], theta_min_deg[lower_mask], 'b-', linewidth=2.5, 
             label='Lower branch (stable)', alpha=0.7)
    if np.sum(upper_mask) > 1:
        ax1.plot(Ca_sorted[upper_mask], theta_min_deg[upper_mask], 'r--', linewidth=2.5, 
                 label='Upper branch (unstable)', alpha=0.7)
    
    # Add scatter points with different markers for each phase
    if np.sum(phase0_mask & lower_mask) > 0:
        ax1.scatter(Ca_sorted[phase0_mask & lower_mask], theta_min_deg[phase0_mask & lower_mask], 
                   color='lightblue', s=20, alpha=0.6, marker='o', label='Phase 0 (coarse)')
    if np.sum(phase1_mask & lower_mask) > 0:
        ax1.scatter(Ca_sorted[phase1_mask & lower_mask], theta_min_deg[phase1_mask & lower_mask], 
                   color='blue', s=40, alpha=0.8, marker='s', label='Phase 1 (detailed)')
    if np.sum(phase2_mask) > 0:
        ax1.scatter(Ca_sorted[phase2_mask], theta_min_deg[phase2_mask], 
                   color='red', s=40, alpha=0.8, marker='^', label='Phase 2 (arclength)')
    
    ax1.scatter(Ca_critical, theta_min_critical*180/np.pi, color='green', s=200, marker='*',
                zorder=5, label=f'Turning point\n(Ca={Ca_critical:.5f})')
    
    ax1.set_xlabel('Ca (Capillary Number)', fontsize=14)
    ax1.set_ylabel('$\\theta_{min}$ [degrees]', fontsize=14)
    ax1.set_title('Bifurcation Diagram: $\\theta_{min}$ vs Ca (PRIMARY)', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    ax1.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    if Ca_sorted.max() / Ca_sorted.min() > 100:
        ax1.set_xscale('log')
        ax1.set_xlabel('Ca (Capillary Number) [log scale]', fontsize=14)
    
    # Plot 2: x0 vs Ca with phase distinction (SECONDARY - if data exists)
    if has_x0_data:
        # Filter out None values
        valid_x0_mask = np.array([x is not None for x in x0_sorted])
        
        # Plot lines for branches (only where x0 exists)
        lower_valid = lower_mask & valid_x0_mask
        upper_valid = upper_mask & valid_x0_mask
        
        if np.sum(lower_valid) > 1:
            ax2.plot(Ca_sorted[lower_valid], x0_sorted[lower_valid], 'b-', linewidth=2.5, 
                     label='Lower branch (stable)', alpha=0.7)
        if np.sum(upper_valid) > 1:
            ax2.plot(Ca_sorted[upper_valid], x0_sorted[upper_valid], 'r--', linewidth=2.5, 
                     label='Upper branch (unstable)', alpha=0.7)
        
        # Add scatter points with different markers for each phase
        if np.sum(phase0_mask & lower_valid) > 0:
            ax2.scatter(Ca_sorted[phase0_mask & lower_valid], x0_sorted[phase0_mask & lower_valid], 
                       color='lightblue', s=20, alpha=0.6, marker='o')
        if np.sum(phase1_mask & lower_valid) > 0:
            ax2.scatter(Ca_sorted[phase1_mask & lower_valid], x0_sorted[phase1_mask & lower_valid], 
                       color='blue', s=40, alpha=0.8, marker='s')
        if np.sum(phase2_mask & valid_x0_mask) > 0:
            ax2.scatter(Ca_sorted[phase2_mask & valid_x0_mask], x0_sorted[phase2_mask & valid_x0_mask], 
                       color='red', s=40, alpha=0.8, marker='^')
        
        if x0_critical is not None:
            ax2.scatter(Ca_critical, x0_critical, color='green', s=200, marker='*',
                        zorder=5, label=f'Turning point')
        
        ax2.set_xlabel('Ca (Capillary Number)', fontsize=14)
        ax2.set_ylabel('$x_0$ (Position at $\\theta_{min}$)', fontsize=14)
        ax2.set_title('Bifurcation Diagram: $x_0$ vs Ca', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=12)
        
        if Ca_sorted.max() / Ca_sorted.min() > 100:
            ax2.set_xscale('log')
            ax2.set_xlabel('Ca (Capillary Number) [log scale]', fontsize=14)
    else:
        # No x0 data - this should not happen with the new definition
        ax2.text(0.5, 0.5, 'No x0 data available\n(Error: x0 should always exist)', 
                 transform=ax2.transAxes, ha='center', va='center', fontsize=16)
        ax2.set_xlabel('Ca (Capillary Number)', fontsize=14)
        ax2.set_ylabel('$x_0$', fontsize=14)
        ax2.set_title('Bifurcation Diagram: $x_0$ vs Ca', fontsize=16, fontweight='bold')
    
    plt.suptitle(f'Both Solution Branches - HYBRID METHOD ($\\mu_r$ = {mu_r:.0e}, $\\lambda_{{slip}}$ = {lambda_slip:.0e})',
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'bifurcation_diagram_hybrid.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nBifurcation diagram saved to: {plot_path}")

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Hybrid solver: x0 for lower branch, arclength for upper')
    
    parser.add_argument('--mu-r', type=float, default=DEFAULT_MU_R,
                        help=f'Viscosity ratio mu_g/mu_l (default: {DEFAULT_MU_R})')
    parser.add_argument('--lambda-slip', type=float, default=DEFAULT_LAMBDA_SLIP,
                        help=f'Slip length (default: {DEFAULT_LAMBDA_SLIP})')
    parser.add_argument('--theta0', type=float, default=DEFAULT_THETA0*180/np.pi,
                        help=f'Initial contact angle in degrees (default: {DEFAULT_THETA0*180/np.pi:.0f})')
    parser.add_argument('--w', type=float, default=DEFAULT_W,
                        help=f'Curvature boundary condition at s=Delta (default: {DEFAULT_W})')
    parser.add_argument('--delta', type=float, default=DEFAULT_DELTA,
                        help=f'Domain size (default: {DEFAULT_DELTA})')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory for plots and data (default: output)')
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS,
                        help=f'Number of parallel workers (default: {DEFAULT_WORKERS})')
    
    args = parser.parse_args()
    
    # Convert theta0 to radians
    theta0_rad = args.theta0 * np.pi / 180
    
    # Run the hybrid continuation
    trace_both_branches_hybrid(
        mu_r=args.mu_r,
        lambda_slip=args.lambda_slip,
        theta0=theta0_rad,
        w_bc=args.w,
        Delta=args.delta,
        output_dir=args.output_dir,
        n_workers=args.workers
    )