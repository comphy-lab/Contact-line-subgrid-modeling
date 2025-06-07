import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.interpolate import UnivariateSpline
from scipy.optimize import root_scalar
from scipy.linalg import solve as lin_solve
import os
from functools import partial
import argparse
from multiprocessing import Pool, cpu_count
import time

# Import the critical Ca finding function from GLE_solver
from GLE_solver import find_critical_ca_lower_branch

# Default parameters
DEFAULT_DELTA = 10.0  # Large domain for continuation
DEFAULT_LAMBDA_SLIP = 1e-4  # Slip length
DEFAULT_MU_R = 1e-6  # \mu_g/\mu_l
DEFAULT_THETA0 = np.pi/2  # theta at s = 0
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
            
            min_theta_idx = np.argmin(theta_vals)
            theta_min = theta_vals[min_theta_idx]
            
            # Calculate x values
            x_vals = np.zeros_like(solution.x)
            x_vals[1:] = np.cumsum(np.diff(solution.x) * np.cos(theta_vals[:-1]))
            x0 = x_vals[min_theta_idx]
            
            return x0, theta_min, solution
        else:
            return None, None, None
    except Exception as e:
        return None, None, None

def solve_for_x0_newton(x0_target, Ca_guess, mu_r, lambda_slip, theta0, w_bc, Delta, 
                       Ca_spline=None, solution_cache=None):
    """
    Find Ca for target x0 using Newton's method with smart initial guess
    """
    # Get initial Ca guess from spline if available
    if Ca_spline is not None:
        try:
            Ca_init = float(Ca_spline(x0_target))
            Ca_init = max(1e-6, min(1.0, Ca_init))
        except:
            Ca_init = Ca_guess
    else:
        Ca_init = Ca_guess
    
    # Get initial solution guess from cache if available
    s_range = None
    y_guess = None
    if solution_cache is not None and len(solution_cache) > 0:
        # Find closest x0 in cache
        x0_cache = [item['x0'] for item in solution_cache]
        closest_idx = np.argmin(np.abs(np.array(x0_cache) - x0_target))
        s_range = solution_cache[closest_idx]['s_range']
        y_guess = solution_cache[closest_idx]['y_solution']
    
    def objective(Ca):
        """Return x0(Ca) - x0_target"""
        if Ca <= 0 or Ca > 1.0:
            return 1e10
        
        x0_computed, theta_min, solution = solve_bvp_for_ca(
            Ca, mu_r, lambda_slip, theta0, w_bc, Delta, s_range, y_guess
        )
        
        if x0_computed is not None:
            return x0_computed - x0_target
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
    x0_final, theta_min, solution = solve_bvp_for_ca(
        Ca, mu_r, lambda_slip, theta0, w_bc, Delta, s_range, y_guess, tol=1e-6
    )
    
    if x0_final is not None and abs(x0_final - x0_target) < 0.01:
        return Ca, theta_min, solution
    else:
        return None, None, None

def worker_solve_x0(args):
    """Worker function for parallel processing with improved error handling"""
    x0_target, Ca_guess, mu_r, lambda_slip, theta0, w_bc, Delta, Ca_spline_data = args
    
    # Reconstruct spline in worker
    if Ca_spline_data is not None:
        x0_data, Ca_data = Ca_spline_data
        try:
            Ca_spline = UnivariateSpline(x0_data, Ca_data, k=min(3, len(x0_data)-1), s=0)
        except:
            Ca_spline = None
    else:
        Ca_spline = None
    
    # Try with Newton method first
    Ca, theta_min, solution = solve_for_x0_newton(
        x0_target, Ca_guess, mu_r, lambda_slip, theta0, w_bc, Delta, Ca_spline
    )
    
    # If Newton fails, try bracketing method
    if Ca is None and Ca_spline is not None:
        try:
            # Bracket search around initial guess
            Ca_lower = Ca_guess * 0.1
            Ca_upper = Ca_guess * 10.0
            
            def objective(Ca):
                x0_comp, _, _ = solve_bvp_for_ca(
                    Ca, mu_r, lambda_slip, theta0, w_bc, Delta
                )
                return (x0_comp - x0_target) if x0_comp is not None else 1e10
            
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
        return {
            'x0': x0_target,
            'Ca': Ca,
            'theta_min': theta_min,
            's_range': solution.x,
            'y_solution': solution.y
        }
    else:
        return None

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
        # Perturb Ca
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
            U_diff = (solution_new.y - solution_old.y).flatten()
            dU_ds_flat = dU_ds.flatten()
            
            N = np.dot(U_diff, dU_ds_flat) + (Ca - Ca_old) * dCa_ds - ds
            
            if abs(N) < 1e-8:
                self.previous_Ca.append(Ca)
                return Ca, solution_new, x0_new, theta_min_new
            
            # Newton update for Ca
            dCa_newton = 1e-5
            Ca_pert = Ca + dCa_newton
            
            x0_pert, theta_min_pert, solution_pert = solve_bvp_for_ca(
                Ca_pert, self.mu_r, self.lambda_slip, self.theta0, self.w_bc, self.Delta,
                s_range, solution_new.y, tol=1e-6
            )
            
            if solution_pert is None:
                dCa_newton = 1e-6
                Ca_pert = Ca + dCa_newton
                x0_pert, theta_min_pert, solution_pert = solve_bvp_for_ca(
                    Ca_pert, self.mu_r, self.lambda_slip, self.theta0, self.w_bc, self.Delta,
                    s_range, solution_new.y, tol=1e-6
                )
                if solution_pert is None:
                    return None
            
            # Compute dN/dCa
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
                
                # Update guess
                y_guess = solution_new.y
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
    
    # Phase 0: Find approximate Ca_cr using GLE_solver's efficient method
    print("\nPhase 0: Finding approximate Ca_cr using efficient bisection...")
    start_time_phase0 = time.time()
    
    # Use a coarse tolerance for the initial estimate
    Ca_cr_approx, Ca_values_phase0, x0_values_phase0, theta_min_values_phase0 = find_critical_ca_lower_branch(
        mu_r, lambda_slip, theta0, w_bc, Delta, 
        nGridInit=DEFAULT_NGRID, 
        output_dir=output_dir,
        Ca_requested=1.0,  # High value to ensure we find Ca_cr
        tolerance=1e-3  # Coarse tolerance for speed
    )
    
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
    
    # Phase 1: Use x0 parameterization to capture detailed lower branch
    print("\nPhase 1: Capturing detailed lower branch with x0 parameterization...")
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
                print(f"Ca = {Ca:.6f}, x0 = {x0:.4f}, theta_min = {theta_min*180/np.pi:.2f}°")
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
    
    # Dense x0 sweep focused on the region near turning point
    # Since we know Ca_cr approximately, we can focus our efforts
    x0_min = min(x0_initial) * 0.5
    x0_max_lower = x0_turning_approx * 1.3  # Don't extend too far past turning point
    
    # Create adaptive grid with most points near turning point
    x0_grid1 = np.linspace(x0_min, x0_turning_approx * 0.9, 20)
    x0_grid2 = np.linspace(x0_turning_approx * 0.9, x0_turning_approx * 1.1, 30)  # Dense near turning
    x0_grid3 = np.linspace(x0_turning_approx * 1.1, x0_max_lower, 10)
    x0_test_lower = np.unique(np.concatenate([x0_grid1, x0_grid2, x0_grid3]))
    
    print(f"\nParallel x0 sweep for lower branch ({len(x0_test_lower)} points)...")
    print(f"  x0 range: [{x0_min:.4f}, {x0_max_lower:.4f}]")
    
    # Build spline for initial guesses
    Ca_spline = UnivariateSpline(x0_sorted, Ca_sorted, k=min(3, len(x0_sorted)-1), s=0)
    
    args_list = []
    for x0_target in x0_test_lower:
        try:
            Ca_guess = float(Ca_spline(x0_target))
            Ca_guess = max(1e-6, min(1.0, Ca_guess))
        except:
            # Extrapolation fallback
            if x0_target < x0_sorted[0]:
                Ca_guess = Ca_sorted[0] * (x0_target / x0_sorted[0])**0.5
            else:
                Ca_guess = Ca_sorted[-1]
        
        args_list.append((x0_target, Ca_guess, mu_r, lambda_slip, theta0, w_bc, Delta, Ca_spline_data))
    
    # Parallel execution with progress tracking
    lower_branch_results = []
    failed_count = 0
    
    with Pool(n_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(worker_solve_x0, args_list)):
            if result is not None:
                lower_branch_results.append(result)
            else:
                failed_count += 1
            
            # Progress update
            if (i + 1) % 20 == 0:
                success_rate = len(lower_branch_results) / (i + 1) * 100
                print(f"  Progress: {i+1}/{len(x0_test_lower)} points, "
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
    
    # Combine results with branch tracking
    all_results = []
    branch_labels = []
    
    # Add lower branch results
    for r in lower_branch_results:
        all_results.append(r)
        branch_labels.append('lower')
    
    # Add upper branch points (excluding the turning point which is already included)
    for i in range(1, len(upper_branch_Ca)):
        all_results.append({
            'Ca': upper_branch_Ca[i],
            'x0': upper_branch_x0[i],
            'theta_min': upper_branch_theta_min[i]
        })
        branch_labels.append('upper')
    
    # Extract arrays
    Ca_all = np.array([r['Ca'] for r in all_results])
    x0_all = np.array([r['x0'] for r in all_results])
    theta_min_all = np.array([r['theta_min'] for r in all_results])
    branch_all = np.array(branch_labels)
    
    # Sort by Ca while keeping track of branches
    sort_idx = np.argsort(Ca_all)
    Ca_sorted = Ca_all[sort_idx]
    x0_sorted = x0_all[sort_idx]
    theta_min_sorted = theta_min_all[sort_idx]
    branch_sorted = branch_all[sort_idx]
    
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
    plot_bifurcation_hybrid(Ca_sorted, x0_sorted, theta_min_sorted, branch_sorted, Ca_critical,
                           x0_critical, theta_min_critical, mu_r, lambda_slip, output_dir)
    
    # Save data with branch information
    # Convert branch labels to numeric (0=lower, 1=upper)
    branch_numeric = np.array([0 if b == 'lower' else 1 for b in branch_sorted])
    csv_data = np.column_stack((Ca_sorted, x0_sorted, theta_min_sorted, theta_min_sorted*180/np.pi, branch_numeric))
    csv_path = os.path.join(output_dir, 'both_branches_hybrid.csv')
    np.savetxt(csv_path, csv_data, delimiter=',',
               header='Ca,x0,theta_min_rad,theta_min_deg,branch(0=lower,1=upper)', comments='')
    print(f"\nData saved to: {csv_path}")
    
    return Ca_sorted, x0_sorted, theta_min_sorted

def plot_bifurcation_hybrid(Ca_sorted, x0_sorted, theta_min_sorted, branch_sorted, Ca_critical,
                           x0_critical, theta_min_critical, mu_r, lambda_slip, output_dir):
    """Create bifurcation diagram plots"""
    # Separate branches based on labels
    lower_mask = branch_sorted == 'lower'
    upper_mask = branch_sorted == 'upper'
    
    # Create plots
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: x0 vs Ca
    ax1.plot(Ca_sorted[lower_mask], x0_sorted[lower_mask], 'b-', linewidth=3, label='Lower branch (stable)')
    ax1.scatter(Ca_sorted[lower_mask], x0_sorted[lower_mask], color='blue', s=30, alpha=0.6)
    
    if np.sum(upper_mask) > 1:
        ax1.plot(Ca_sorted[upper_mask], x0_sorted[upper_mask], 'r--', linewidth=3, label='Upper branch (unstable)')
        ax1.scatter(Ca_sorted[upper_mask], x0_sorted[upper_mask], color='red', s=30, alpha=0.6)
    
    ax1.scatter(Ca_critical, x0_critical, color='green', s=200, marker='*',
                zorder=5, label=f'Turning point\n(Ca={Ca_critical:.5f})')
    
    ax1.set_xlabel('Ca (Capillary Number)', fontsize=14)
    ax1.set_ylabel('$x_0$ (Position at $\\theta_{min}$)', fontsize=14)
    ax1.set_title('Bifurcation Diagram: $x_0$ vs Ca', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    if Ca_sorted.max() / Ca_sorted.min() > 100:
        ax1.set_xscale('log')
        ax1.set_xlabel('Ca (Capillary Number) [log scale]', fontsize=14)
    
    # Plot 2: theta_min vs Ca
    theta_min_deg = theta_min_sorted * 180 / np.pi
    
    ax2.plot(Ca_sorted[lower_mask], theta_min_deg[lower_mask], 'b-', linewidth=3, label='Lower branch (stable)')
    ax2.scatter(Ca_sorted[lower_mask], theta_min_deg[lower_mask], color='blue', s=30, alpha=0.6)
    
    if np.sum(upper_mask) > 1:
        ax2.plot(Ca_sorted[upper_mask], theta_min_deg[upper_mask], 'r--', linewidth=3, label='Upper branch (unstable)')
        ax2.scatter(Ca_sorted[upper_mask], theta_min_deg[upper_mask], color='red', s=30, alpha=0.6)
    
    ax2.scatter(Ca_critical, theta_min_critical*180/np.pi, color='green', s=200, marker='*',
                zorder=5, label=f'Turning point')
    
    ax2.set_xlabel('Ca (Capillary Number)', fontsize=14)
    ax2.set_ylabel('$\\theta_{min}$ [degrees]', fontsize=14)
    ax2.set_title('Bifurcation Diagram: $\\theta_{min}$ vs Ca', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=12)
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    if Ca_sorted.max() / Ca_sorted.min() > 100:
        ax2.set_xscale('log')
        ax2.set_xlabel('Ca (Capillary Number) [log scale]', fontsize=14)
    
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