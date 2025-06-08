import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import root_scalar, brentq
from scipy.linalg import solve as lin_solve
import os
from functools import partial
import argparse
from multiprocessing import Pool, cpu_count
import time

# Import utilities from src-local
import sys
sys.path.append('src-local')
from find_x0_utils import find_x0_and_theta_min
from gle_utils import (
    f1, f2, f3, f, GLE, boundary_conditions
)

# Import critical Ca finding from GLE_solver
from GLE_solver import find_critical_ca_lower_branch

# Default parameters
DEFAULT_DELTA = 10.0  # Large domain for continuation
DEFAULT_LAMBDA_SLIP = 1e-4  # Slip length
DEFAULT_MU_R = 1e-6  # \mu_g/\mu_l
DEFAULT_THETA0 = np.pi/6  # theta at s = 0
DEFAULT_W = 0  # curvature boundary condition at s = \Delta
DEFAULT_NGRID = 10000  # Number of grid points

# Note: Mathematical functions f1, f2, f3, f and ODE system GLE, boundary_conditions
# are now imported from gle_utils to avoid duplication

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

def analyze_solution_properties(solution, Ca, verbose=False):
    """
    Analyze key properties of a BVP solution for debugging and convergence monitoring
    """
    h_vals, theta_vals, omega_vals = solution.y
    s_vals = solution.x
    
    # Find critical points
    x0, theta_min, x0_idx = find_x0_and_theta_min(s_vals, theta_vals)
    
    # Compute gradients and curvature metrics
    max_theta_gradient = np.max(np.abs(np.gradient(theta_vals, s_vals)))
    max_h_gradient = np.max(np.abs(np.gradient(h_vals, s_vals)))
    max_omega = np.max(np.abs(omega_vals))
    
    # Estimate solution "stiffness" - how rapidly it changes
    theta_variation = np.std(theta_vals)
    h_variation = np.std(h_vals)
    
    properties = {
        'Ca': Ca,
        'theta_min': theta_min,
        'theta_min_deg': theta_min * 180 / np.pi,
        'x0': x0,
        'max_theta_grad': max_theta_gradient,
        'max_h_grad': max_h_gradient,
        'max_omega': max_omega,
        'theta_variation': theta_variation,
        'h_variation': h_variation,
        'mesh_points': len(s_vals),
        'domain_size': s_vals[-1] - s_vals[0]
    }
    
    if verbose:
        print(f"    Solution analysis at Ca = {Ca:.6f}:")
        print(f"      θ_min = {theta_min*180/np.pi:.3f}°, x0 = {x0:.6f}")
        print(f"      Max gradients: |∇θ| = {max_theta_gradient:.3e}, |∇h| = {max_h_gradient:.3e}")
        print(f"      Max curvature: |ω| = {max_omega:.3e}")
        print(f"      Variations: σ_θ = {theta_variation:.3e}, σ_h = {h_variation:.3e}")
        print(f"      Mesh: {len(s_vals)} points over domain [{s_vals[0]:.1f}, {s_vals[-1]:.1f}]")
    
    return properties

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


class AdaptiveSolutionTracker:
    """
    Advanced solution tracking with pseudo-arclength-like stepping
    Tracks entire BVP solution U = [h₁, θ₁, ω₁, h₂, θ₂, ω₂, ..., hₙ, θₙ, ωₙ]
    Purpose: Robustly approach critical Ca where θ_min → 0
    """
    def __init__(self, mu_r, lambda_slip, theta0, w_bc, Delta):
        self.mu_r = mu_r
        self.lambda_slip = lambda_slip
        self.theta0 = theta0
        self.w_bc = w_bc
        self.Delta = Delta
        self.tangent_cache = None  # Cache for previous tangent
    
    def solution_to_vector(self, solution):
        """
        Convert BVP solution to flat vector U = [h₁, θ₁, ω₁, h₂, θ₂, ω₂, ...]
        """
        return solution.y.flatten(order='F')  # Fortran order: all h, then all theta, then all omega
    
    def vector_to_solution_shape(self, U_flat, s_points):
        """
        Convert flat vector back to (3, N) solution shape
        """
        N = len(s_points)
        return U_flat.reshape((3, N), order='F')
    
    def compute_tangent(self, Ca, solution, dCa=None):
        """
        Compute tangent vector (dU/ds, dCa/ds) in full solution space
        Uses finite differences to approximate the Jacobian of F(U, Ca) = 0
        """
        U_current = self.solution_to_vector(solution)
        s_points = solution.x
        
        # Adaptive dCa based on Ca magnitude
        if dCa is None:
            dCa = max(1e-5, Ca * 0.001)  # 0.1% of Ca, but at least 1e-5
        
        # Try forward difference in Ca
        x0_plus, theta_min_plus, solution_plus = solve_bvp_for_ca(
            Ca + dCa, self.mu_r, self.lambda_slip, self.theta0, self.w_bc, self.Delta,
            s_points, solution.y, tol=1e-6
        )
        
        if solution_plus is None:
            # Try smaller step
            dCa = dCa / 10
            x0_plus, theta_min_plus, solution_plus = solve_bvp_for_ca(
                Ca + dCa, self.mu_r, self.lambda_slip, self.theta0, self.w_bc, self.Delta,
                s_points, solution.y, tol=1e-6
            )
            if solution_plus is None:
                # Try backward difference
                dCa = -dCa
                x0_plus, theta_min_plus, solution_plus = solve_bvp_for_ca(
                    Ca + dCa, self.mu_r, self.lambda_slip, self.theta0, self.w_bc, self.Delta,
                    s_points, solution.y, tol=1e-6
                )
                if solution_plus is None:
                    return None
        
        U_plus = self.solution_to_vector(solution_plus)
        
        # Compute derivatives: ∂F/∂Ca ≈ (F(U, Ca+dCa) - F(U, Ca)) / dCa
        # Here F(U, Ca) = 0, so ∂F/∂Ca ≈ (U_plus - U_current) / dCa
        dU_dCa = (U_plus - U_current) / dCa
        
        # Normalize to get unit tangent vector
        # ||dU/ds||² + ||dCa/ds||² = 1
        norm_squared = np.dot(dU_dCa, dU_dCa) + 1.0
        norm = np.sqrt(norm_squared)
        
        dU_ds = dU_dCa / norm
        dCa_ds = 1.0 / norm
        
        # Ensure consistent orientation (moving toward critical Ca)
        if dCa_ds < 0:
            dU_ds = -dU_ds
            dCa_ds = -dCa_ds
        
        # Check consistency with previous tangent if available
        if self.tangent_cache is not None:
            prev_dU_ds, prev_dCa_ds = self.tangent_cache
            # Dot product to check orientation
            dot_product = np.dot(dU_ds, prev_dU_ds) + dCa_ds * prev_dCa_ds
            if dot_product < 0:  # Opposite orientation
                dU_ds = -dU_ds
                dCa_ds = -dCa_ds
        
        tangent = (dU_ds, dCa_ds)
        self.tangent_cache = tangent
        return tangent
    
    def predictor_corrector_step(self, Ca_old, solution_old, ds, verbose=True):
        """
        Take one predictor-corrector step toward critical Ca
        Uses arc-length constraint for robust convergence near θ_min = 0
        """
        U_old = self.solution_to_vector(solution_old)
        s_points = solution_old.x
        
        # Compute tangent vector
        tangent = self.compute_tangent(Ca_old, solution_old)
        if tangent is None:
            if verbose:
                print(f"      Failed to compute tangent")
            return None
        
        dU_ds, dCa_ds = tangent
        if verbose:
            print(f"      Tangent: dCa/ds = {dCa_ds:.6f}, ||dU/ds|| = {np.linalg.norm(dU_ds):.6f}")
        
        # PREDICTOR STEP: Linear extrapolation along tangent
        U_pred = U_old + ds * dU_ds
        Ca_pred = Ca_old + ds * dCa_ds
        
        if verbose:
            print(f"      Predictor: Ca = {Ca_pred:.6f}")
        
        # Convert predicted U back to solution shape for BVP solve
        y_pred = self.vector_to_solution_shape(U_pred, s_points)
        
        # CORRECTOR STEP: Newton iterations with arc-length constraint
        Ca = Ca_pred
        y_guess = y_pred
        
        max_iter = 10
        for iter_count in range(max_iter):
            # Solve BVP at current Ca guess
            x0_new, theta_min_new, solution_new = solve_bvp_for_ca(
                Ca, self.mu_r, self.lambda_slip, self.theta0, self.w_bc, self.Delta,
                s_points, y_guess, tol=1e-6
            )
            
            if solution_new is None:
                if verbose and iter_count == 0:
                    print(f"      Corrector: BVP failed at Ca = {Ca:.6f}")
                return None
            
            U_new = self.solution_to_vector(solution_new)
            
            # Check arc-length constraint
            # G = (U - U_old)·(dU/ds) + (Ca - Ca_old)·(dCa/ds) - ds = 0
            G = np.dot(U_new - U_old, dU_ds) + (Ca - Ca_old) * dCa_ds - ds
            
            if verbose and iter_count == 0:
                print(f"      Corrector iter 1: Ca = {Ca:.6f}, θ_min = {theta_min_new*180/np.pi:.3f}°, G = {G:.3e}")
            
            if abs(G) < 1e-8:
                if verbose:
                    print(f"      Converged after {iter_count+1} iterations")
                return Ca, theta_min_new, solution_new
            
            # Newton correction for Ca
            # Need ∂G/∂Ca = dCa_ds + (dU/dCa)·(dU/ds)
            dCa_newton = 1e-6
            Ca_pert = Ca + dCa_newton
            
            x0_pert, theta_min_pert, solution_pert = solve_bvp_for_ca(
                Ca_pert, self.mu_r, self.lambda_slip, self.theta0, self.w_bc, self.Delta,
                s_points, solution_new.y, tol=1e-6
            )
            
            if solution_pert is None:
                # Try smaller perturbation
                dCa_newton = 1e-7
                Ca_pert = Ca + dCa_newton
                x0_pert, theta_min_pert, solution_pert = solve_bvp_for_ca(
                    Ca_pert, self.mu_r, self.lambda_slip, self.theta0, self.w_bc, self.Delta,
                    s_points, solution_new.y, tol=1e-6
                )
                if solution_pert is None:
                    return None
            
            U_pert = self.solution_to_vector(solution_pert)
            
            # Compute ∂G/∂Ca
            dU_dCa_local = (U_pert - U_new) / dCa_newton
            dG_dCa = dCa_ds + np.dot(dU_dCa_local, dU_ds)
            
            if abs(dG_dCa) > 1e-12:
                # Newton update
                delta_Ca = -G / dG_dCa
                
                # Damping for stability
                max_step = 0.1 * abs(Ca)
                if abs(delta_Ca) > max_step:
                    delta_Ca = max_step * np.sign(delta_Ca)
                
                Ca_new = Ca + delta_Ca
                Ca_new = max(1e-8, min(1.0, Ca_new))  # Keep in reasonable bounds
                
                if abs(Ca_new - Ca) < 1e-12:  # No progress
                    break
                
                Ca = Ca_new
                # Update guess using current solution
                y_guess = solution_new.y
            else:
                # Singular Jacobian, can't continue
                if verbose:
                    print(f"      Warning: Singular Jacobian (dG/dCa ≈ 0)")
                break
        
        return None


def find_critical_ca_advanced(mu_r, lambda_slip, theta0=DEFAULT_THETA0, w_bc=DEFAULT_W,
                            Delta=DEFAULT_DELTA, output_dir='output',
                            tolerance_init_ca=1e-6, tolerance_s=1e-3, tolerance_th=1e-6):
    """
    Advanced three-phase approach to find critical Ca where theta_min → 0
    
    Phase 0: Initial Ca_cr estimate using coarse search + IQI refinement
    Phase 1: Adaptive solution tracking to approach θ_min ≈ 0
    Phase 2: Final high-precision refinement using root finding
    
    This is NOT a continuation method - it specifically finds Ca_critical
    """
    print(f"\nADVANCED CRITICAL Ca FINDER")
    print(f"Parameters: theta_0 = {theta0/np.pi*180:.1f}°, mu_r = {mu_r}, lambda_slip = {lambda_slip}")
    print(f"Tolerances: init_Ca={tolerance_init_ca}, s={tolerance_s}, th={tolerance_th}")
    print(f"Method: Three-phase approach with adaptive solution tracking")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Phase 0: Initial Ca_cr estimate
    print("\n" + "="*60)
    print("PHASE 0: Initial Ca_cr estimate using GLE_solver method")
    print("="*60)
    start_time_phase0 = time.time()
    
    Ca_cr_approx, Ca_values_phase0, x0_values_phase0, theta_min_values_phase0, s_range_final, y_guess_final = find_critical_ca_lower_branch(
        mu_r, lambda_slip, theta0, w_bc, Delta, 
        nGridInit=DEFAULT_NGRID, 
        output_dir=output_dir,
        Ca_requested=1.0,  # Large value to trigger search
        tolerance=tolerance_init_ca
    )
    
    phase0_time = time.time() - start_time_phase0
    print(f"\nPhase 0 completed in {phase0_time:.1f} seconds")
    print(f"Initial estimate: Ca_cr ≈ {Ca_cr_approx:.6f}")
    
    # Get solution at approximate critical Ca
    # Use the final solution from Phase 0 as a starting point
    x0_cr_approx, theta_min_cr_approx, solution_cr_approx = solve_bvp_for_ca(
        Ca_cr_approx, mu_r, lambda_slip, theta0, w_bc, Delta, s_range_final, y_guess_final
    )
    
    if solution_cr_approx is None:
        print("Error: Could not solve at approximate critical Ca")
        return None
    
    print(f"θ_min at Ca_cr_approx: {theta_min_cr_approx*180/np.pi:.2f}°")
    
    # Phase 1: Focused refinement near critical Ca
    print("\n" + "="*60)
    print("PHASE 1: Focused search near critical Ca")
    print("="*60)
    start_time_phase1 = time.time()
    
    # Do a focused search in a small interval around Ca_cr_approx
    # This is more reliable than trying to trace from far away
    Ca_lower = Ca_cr_approx * 0.95
    Ca_upper = Ca_cr_approx * 1.05
    
    print(f"Searching in interval [{Ca_lower:.6f}, {Ca_upper:.6f}]")
    
    # Find where the solver starts to fail in this interval
    Ca_good = Ca_lower
    Ca_fail = None
    
    # First, verify our bounds
    x0_lower, theta_min_lower, sol_lower = solve_bvp_for_ca(
        Ca_lower, mu_r, lambda_slip, theta0, w_bc, Delta
    )
    
    if sol_lower is None:
        print(f"Warning: Solution failed at Ca_lower = {Ca_lower:.6f}")
        print("Using Ca_cr_approx from Phase 0 as final result")
        Ca_values = [Ca_cr_approx]
        theta_min_values = [theta_min_cr_approx]
        x0_values = [x0_cr_approx]
    else:
        print(f"Ca_lower = {Ca_lower:.6f}: θ_min = {theta_min_lower*180/np.pi:.2f}°")
    
        # Test upper bound
        x0_upper, theta_min_upper, sol_upper = solve_bvp_for_ca(
            Ca_upper, mu_r, lambda_slip, theta0, w_bc, Delta
        )
        
        if sol_upper is not None:
            print(f"Ca_upper = {Ca_upper:.6f}: θ_min = {theta_min_upper*180/np.pi:.2f}°")
            print("Warning: Solution still exists at Ca_upper. Critical Ca may be higher.")
            # Extend search range
            Ca_upper = Ca_cr_approx * 1.2
        
        # Do a bisection search in this interval
        Ca_values = []
        theta_min_values = []
        x0_values = []
        
        # Binary search with finer resolution near critical
        max_iter = 20
        prev_Ca_mid = None
        stuck_count = 0
        
        for i in range(max_iter):
            Ca_mid = (Ca_good + Ca_upper) / 2
            
            # Check if we're stuck at the same value
            if prev_Ca_mid is not None and abs(Ca_mid - prev_Ca_mid) < 1e-10:
                stuck_count += 1
                if stuck_count >= 3:
                    print(f"  Search stuck at Ca = {Ca_mid:.6f} for {stuck_count} iterations.")
                    print("  Moving to Phase 2...")
                    break
            else:
                stuck_count = 0
            prev_Ca_mid = Ca_mid
            
            x0_mid, theta_min_mid, sol_mid = solve_bvp_for_ca(
                Ca_mid, mu_r, lambda_slip, theta0, w_bc, Delta
            )
            
            if sol_mid is not None:
                Ca_good = Ca_mid
                Ca_values.append(Ca_mid)
                theta_min_values.append(theta_min_mid)
                x0_values.append(x0_mid)
                print(f"  Iter {i+1}: Ca = {Ca_mid:.6f} SUCCESS, θ_min = {theta_min_mid*180/np.pi:.3f}°")
                
                # Check if we're close enough to critical
                if theta_min_mid < tolerance_s:
                    print(f"  Found θ_min < tolerance at Ca = {Ca_mid:.6f}")
                    break
            else:
                Ca_fail = Ca_mid
                print(f"  Iter {i+1}: Ca = {Ca_mid:.6f} FAILED")
            
            # Check convergence
            if Ca_fail is not None and (Ca_fail - Ca_good) < tolerance_init_ca:
                print(f"  Converged: interval width = {Ca_fail - Ca_good:.2e}")
                break
            
            # Additional check: if we're making very small progress
            if i > 5 and Ca_fail is not None:
                progress = (Ca_fail - Ca_good) / Ca_upper
                if progress < 0.001:  # Less than 0.1% of the search range
                    print(f"  Very slow progress detected (interval = {Ca_fail - Ca_good:.2e})")
                    print("  Moving to Phase 2...")
                    break
    
    phase1_time = time.time() - start_time_phase1
    print(f"\nPhase 1 completed in {phase1_time:.1f} seconds")
    if Ca_values:
        print(f"Found {len(Ca_values)} successful solutions")
        print(f"Best Ca = {Ca_values[-1]:.6f}, θ_min = {theta_min_values[-1]*180/np.pi:.3f}°")
    else:
        print("No additional solutions found in Phase 1")
    
    # Phase 2: Parameter tracking and extrapolation
    print("\n" + "="*60)
    print("PHASE 2: Parameter tracking in (Ca, θ_min) space")
    print("="*60)
    start_time_phase2 = time.time()
    
    # Combine data from all phases
    all_Ca_phase2 = []
    all_theta_min_phase2 = []
    
    # Add Phase 0 data
    for i in range(len(Ca_values_phase0)):
        if theta_min_values_phase0[i] < 0.5:  # Only include points with θ_min < ~28°
            all_Ca_phase2.append(Ca_values_phase0[i])
            all_theta_min_phase2.append(theta_min_values_phase0[i])
    
    # Add Phase 1 data
    for i in range(len(Ca_values)):
        # Check for duplicates
        is_duplicate = False
        for ca in all_Ca_phase2:
            if abs(ca - Ca_values[i]) < 1e-8:
                is_duplicate = True
                break
        if not is_duplicate:
            all_Ca_phase2.append(Ca_values[i])
            all_theta_min_phase2.append(theta_min_values[i])
    
    # Sort by Ca
    if all_Ca_phase2:
        sorted_indices = np.argsort(all_Ca_phase2)
        Ca_sorted = np.array([all_Ca_phase2[i] for i in sorted_indices])
        theta_min_sorted = np.array([all_theta_min_phase2[i] for i in sorted_indices])
    else:
        # Fallback to Phase 0 data only
        Ca_sorted = np.array(Ca_values_phase0)
        theta_min_sorted = np.array(theta_min_values_phase0)
    
    print(f"Using {len(Ca_sorted)} data points for parameter tracking")
    
    # Step 1: Track the curve in (Ca, θ_min) space to get more points near θ_min = 0
    Ca_last = Ca_sorted[-1]
    theta_min_last = theta_min_sorted[-1]
    
    # Adaptive step size based on how close we are to θ_min = 0
    if theta_min_last > 0.3:  # Very far from critical (> ~17°)
        step_factor = 1.1   # 10% steps - be aggressive
        print(f"\nθ_min = {theta_min_last*180/np.pi:.1f}° is far from critical, using aggressive stepping")
    elif theta_min_last > 0.1:  # Far from critical
        step_factor = 1.05  # 5% steps
    elif theta_min_last > 0.01:  # Getting closer
        step_factor = 1.02  # 2% steps
    else:  # Very close
        step_factor = 1.01  # 1% steps
    
    print(f"Tracking parameter curve from Ca = {Ca_last:.6f}, θ_min = {theta_min_last*180/np.pi:.3f}°")
    
    # Track the curve until we hit a failure or θ_min gets very small
    max_tracking_steps = 20
    tracking_Ca = [Ca_last]
    tracking_theta_min = [theta_min_last]
    
    for i in range(max_tracking_steps):
        # Predict next Ca using local derivative
        if len(tracking_Ca) >= 2:
            # Estimate dθ_min/dCa from recent points
            dtheta_dCa = (tracking_theta_min[-1] - tracking_theta_min[-2]) / (tracking_Ca[-1] - tracking_Ca[-2])
            
            # Use adaptive step size based on curvature
            if len(tracking_Ca) >= 3:
                # Estimate second derivative to detect curvature
                dtheta_dCa_prev = (tracking_theta_min[-2] - tracking_theta_min[-3]) / (tracking_Ca[-2] - tracking_Ca[-3])
                curvature = abs(dtheta_dCa - dtheta_dCa_prev) / (tracking_Ca[-1] - tracking_Ca[-3])
                
                # Smaller steps when curvature is high
                if curvature > 10:
                    step_factor = 1.005
                elif curvature > 5:
                    step_factor = 1.01
                else:
                    step_factor = 1.02
        
        Ca_next = tracking_Ca[-1] * step_factor
        
        # Try to solve at the predicted Ca
        x0_next, theta_min_next, sol_next = solve_bvp_for_ca(
            Ca_next, mu_r, lambda_slip, theta0, w_bc, Delta
        )
        
        if sol_next is not None:
            tracking_Ca.append(Ca_next)
            tracking_theta_min.append(theta_min_next)
            print(f"  Step {i+1}: Ca = {Ca_next:.6f}, θ_min = {theta_min_next*180/np.pi:.3f}°")
            
            # Stop if θ_min is very small
            if theta_min_next < 0.001:
                print(f"  Reached θ_min < 0.001 rad (~0.057°)")
                break
                
            # Stop if we're making very small progress
            if i > 5 and abs(theta_min_next - tracking_theta_min[-2]) < 1e-6:
                print(f"  θ_min change < 1e-6, stopping tracking")
                break
        else:
            print(f"  Step {i+1}: Ca = {Ca_next:.6f} FAILED - reached critical region")
            # If this is the first step and we're far from critical, try smaller steps
            if i == 0 and theta_min_last > 0.1:
                print("  First step failed, trying smaller increments...")
                smaller_factors = [1.02, 1.01, 1.005, 1.002]
                for sf in smaller_factors:
                    Ca_try = tracking_Ca[-1] * sf
                    x0_try, theta_min_try, sol_try = solve_bvp_for_ca(
                        Ca_try, mu_r, lambda_slip, theta0, w_bc, Delta
                    )
                    if sol_try is not None:
                        tracking_Ca.append(Ca_try)
                        tracking_theta_min.append(theta_min_try)
                        print(f"    Success with factor {sf}: Ca = {Ca_try:.6f}, θ_min = {theta_min_try*180/np.pi:.3f}°")
                        break
                else:
                    # None of the smaller steps worked
                    break
            else:
                break
    
    # Step 2: Fit a model to extrapolate to θ_min = 0
    # Combine all available data
    all_Ca_final = list(Ca_sorted) + tracking_Ca[1:]  # Avoid duplicating last point
    all_theta_min_final = list(theta_min_sorted) + tracking_theta_min[1:]
    
    # Sort and remove duplicates
    combined = sorted(zip(all_Ca_final, all_theta_min_final))
    Ca_final = np.array([c[0] for c in combined])
    theta_min_final_arr = np.array([c[1] for c in combined])
    
    # Use only points with small θ_min for fitting
    mask = theta_min_final_arr < 0.5  # Use points with θ_min < ~28.6°
    Ca_fit = Ca_final[mask]
    theta_fit = theta_min_final_arr[mask]
    
    # If we don't have enough close points, use all available points
    if len(Ca_fit) < 3:
        print(f"  Only {len(Ca_fit)} points with θ_min < 0.5 rad, using all {len(Ca_final)} points")
        Ca_fit = Ca_final
        theta_fit = theta_min_final_arr
    
    print(f"\nExtrapolating to θ_min = 0 using {len(Ca_fit)} points")
    
    if len(Ca_fit) >= 3:
        # Try different models and choose the best
        models = []
        
        # Model 1: Quadratic in θ_min
        try:
            p2 = np.polyfit(theta_fit, Ca_fit, 2)
            Ca_quad = np.polyval(p2, 0)
            # Check if reasonable
            if Ca_fit[-1] < Ca_quad < Ca_fit[-1] * 1.2:
                models.append(('Quadratic', Ca_quad))
        except:
            pass
        
        # Model 2: Linear in log(θ_min) - often works well near critical points
        try:
            # Only use very small theta values to avoid log of negative
            mask_log = theta_fit > 1e-4
            if np.sum(mask_log) >= 2:
                log_theta = np.log(theta_fit[mask_log])
                Ca_log = Ca_fit[mask_log]
                p_log = np.polyfit(log_theta, Ca_log, 1)
                # Extrapolate carefully
                log_theta_target = np.log(1e-6)  # Very small but not zero
                Ca_log_extrap = np.polyval(p_log, log_theta_target)
                if Ca_fit[-1] < Ca_log_extrap < Ca_fit[-1] * 1.2:
                    models.append(('Log-linear', Ca_log_extrap))
        except:
            pass
        
        # Model 3: Power law: θ_min = A * (Ca_cr - Ca)^β
        # Rearranged: Ca = Ca_cr - (θ_min/A)^(1/β)
        try:
            # Initial guess for Ca_cr
            Ca_cr_guess = Ca_fit[-1] * 1.1
            
            # Fit linearized form: log(θ) = log(A) + β*log(Ca_cr - Ca)
            def power_law_residual(Ca_cr_test):
                if np.any(Ca_fit >= Ca_cr_test):
                    return np.inf
                log_diff = np.log(Ca_cr_test - Ca_fit)
                log_theta = np.log(theta_fit)
                p = np.polyfit(log_diff, log_theta, 1)
                return np.sum((log_theta - np.polyval(p, log_diff))**2)
            
            # Find best Ca_cr
            from scipy.optimize import minimize_scalar
            result = minimize_scalar(power_law_residual, bounds=(Ca_fit[-1], Ca_fit[-1]*1.5), method='bounded')
            if result.success:
                Ca_power = result.x
                if Ca_fit[-1] < Ca_power < Ca_fit[-1] * 1.2:
                    models.append(('Power law', Ca_power))
        except:
            pass
        
        # Choose the model with Ca closest to our last successful solve
        if models:
            # Sort by distance from last Ca
            models.sort(key=lambda x: abs(x[1] - Ca_fit[-1]))
            model_name, Ca_critical = models[0]
            print(f"  Best model: {model_name}, Ca_critical = {Ca_critical:.8f}")
            
            # Show all models for comparison
            if len(models) > 1:
                print("  Other models:")
                for name, ca in models[1:]:
                    print(f"    {name}: Ca_critical = {ca:.8f}")
        else:
            # Fallback to simple linear extrapolation
            print("  No good model fit, using linear extrapolation")
            if len(Ca_fit) >= 2:
                dtheta_dCa = (theta_fit[-1] - theta_fit[-2]) / (Ca_fit[-1] - Ca_fit[-2])
                Ca_critical = Ca_fit[-1] - theta_fit[-1] / dtheta_dCa
            else:
                Ca_critical = Ca_fit[-1]
        
        # Final sanity check
        if Ca_critical < Ca_fit[-1]:
            print(f"  Warning: Extrapolated Ca ({Ca_critical:.8f}) < last successful Ca ({Ca_fit[-1]:.8f})")
            Ca_critical = Ca_fit[-1]
        elif Ca_critical > Ca_fit[-1] * 1.5:
            print(f"  Warning: Extrapolated Ca too large, capping at 1.5x last value")
            Ca_critical = Ca_fit[-1] * 1.5
            
        # Get theta_min at the critical Ca (should be very small)
        theta_min_final = theta_fit[-1]  # Best we have
        
    else:
        # Not enough points for extrapolation
        print("  Not enough points for extrapolation")
        Ca_critical = Ca_final[-1]
        theta_min_final = theta_min_final_arr[-1]
    
    print(f"\nFinal Ca_critical = {Ca_critical:.8f}")
    print(f"θ_min at last computed point = {theta_min_final:.3e} rad ({theta_min_final*180/np.pi:.3e}°)")
    
    phase2_time = time.time() - start_time_phase2
    print(f"\nPhase 2 completed in {phase2_time:.1f} seconds")
    
    # Performance summary
    total_time = phase0_time + phase1_time + phase2_time
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Phase 0 (initial estimate):        {phase0_time:6.1f}s")
    print(f"Phase 1 (adaptive tracking):       {phase1_time:6.1f}s")
    print(f"Phase 2 (final refinement):        {phase2_time:6.1f}s")
    print(f"Total time:                        {total_time:6.1f}s")
    
    # Combine all data for plotting
    all_Ca = []
    all_theta_min = []
    all_x0 = []
    all_phase = []
    
    # Add Phase 0 data
    for i in range(len(Ca_values_phase0)):
        all_Ca.append(Ca_values_phase0[i])
        all_theta_min.append(theta_min_values_phase0[i])
        all_x0.append(x0_values_phase0[i] if i < len(x0_values_phase0) else np.nan)
        all_phase.append(0)
    
    # Add Phase 1 data
    for i in range(len(Ca_values)):
        # Check for duplicates
        is_duplicate = False
        for j in range(len(all_Ca)):
            if abs(all_Ca[j] - Ca_values[i]) < 1e-8:
                is_duplicate = True
                break
        
        if not is_duplicate:
            all_Ca.append(Ca_values[i])
            all_theta_min.append(theta_min_values[i])
            all_x0.append(x0_values[i] if i < len(x0_values) else np.nan)
            all_phase.append(1)
    
    # Add Phase 2 tracking data
    for i in range(1, len(tracking_Ca)):  # Skip first point (duplicate)
        # Check for duplicates
        is_duplicate = False
        for j in range(len(all_Ca)):
            if abs(all_Ca[j] - tracking_Ca[i]) < 1e-8:
                is_duplicate = True
                break
        
        if not is_duplicate:
            all_Ca.append(tracking_Ca[i])
            all_theta_min.append(tracking_theta_min[i])
            # We don't have x0 for tracking points, but we could compute it if needed
            all_x0.append(np.nan)
            all_phase.append(2)
    
    # Add final critical point (extrapolated)
    all_Ca.append(Ca_critical)
    all_theta_min.append(0.0)  # By definition, we're extrapolating to θ_min = 0
    all_x0.append(np.nan)  # Can't compute x0 at the extrapolated point
    all_phase.append(2)
    
    # Convert to arrays and sort
    all_Ca = np.array(all_Ca)
    all_theta_min = np.array(all_theta_min)
    all_x0 = np.array(all_x0)
    all_phase = np.array(all_phase)
    
    sort_idx = np.argsort(all_Ca)
    Ca_sorted = all_Ca[sort_idx]
    theta_min_sorted = all_theta_min[sort_idx]
    x0_sorted = all_x0[sort_idx]
    phase_sorted = all_phase[sort_idx]
    
    # Create plots
    plot_critical_ca_results(Ca_sorted, theta_min_sorted, x0_sorted, phase_sorted,
                           Ca_critical, theta_min_final, mu_r, lambda_slip, output_dir)
    
    # Save data
    csv_data = np.column_stack((Ca_sorted, theta_min_sorted, theta_min_sorted*180/np.pi, 
                               x0_sorted, phase_sorted))
    csv_path = os.path.join(output_dir, 'critical_ca_results.csv')
    np.savetxt(csv_path, csv_data, delimiter=',',
               header='Ca,theta_min_rad,theta_min_deg,x0,phase(0,1,2)', 
               comments='')
    print(f"\nData saved to: {csv_path}")
    
    return Ca_critical


def plot_critical_ca_results(Ca_sorted, theta_min_sorted, x0_sorted, phase_sorted,
                           Ca_critical, theta_min_final, mu_r, lambda_slip, output_dir):
    """Create plots showing the approach to critical Ca"""
    # Separate phases
    phase0_mask = phase_sorted == 0
    phase1_mask = phase_sorted == 1
    phase2_mask = phase_sorted == 2
    
    # Create plots
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: theta_min vs Ca
    theta_min_deg = theta_min_sorted * 180 / np.pi
    
    # Plot line
    ax1.plot(Ca_sorted, theta_min_deg, 'b-', linewidth=2.5, alpha=0.7)
    
    # Add scatter points for different phases
    if np.sum(phase0_mask) > 0:
        ax1.scatter(Ca_sorted[phase0_mask], theta_min_deg[phase0_mask], 
                   color='lightblue', s=40, alpha=0.7, marker='o', label='Phase 0 (initial)')
    if np.sum(phase1_mask) > 0:
        ax1.scatter(Ca_sorted[phase1_mask], theta_min_deg[phase1_mask], 
                   color='blue', s=40, alpha=0.8, marker='s', label='Phase 1 (adaptive)')
    if np.sum(phase2_mask) > 0:
        ax1.scatter(Ca_sorted[phase2_mask], theta_min_deg[phase2_mask], 
                   color='darkgreen', s=100, alpha=0.9, marker='*', label='Phase 2 (refined)')
    
    # Mark critical point
    ax1.scatter(Ca_critical, theta_min_final*180/np.pi, color='red', s=200, marker='*',
                edgecolor='black', linewidth=2, zorder=5, 
                label=f'Ca_cr = {Ca_critical:.8f}')
    
    ax1.set_xlabel('Ca (Capillary Number)', fontsize=14)
    ax1.set_ylabel('$\\theta_{min}$ [degrees]', fontsize=14)
    ax1.set_title('Approach to Critical Ca: $\\theta_{min}$ → 0', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=2)
    
    # Add inset for zoomed view near critical
    if len(Ca_sorted) > 10:
        # Find points near critical
        near_critical = theta_min_deg < 5  # Within 5 degrees
        if np.sum(near_critical) > 3:
            axins = ax1.inset_axes([0.5, 0.5, 0.45, 0.45])
            axins.plot(Ca_sorted[near_critical], theta_min_deg[near_critical], 'b-', linewidth=2)
            if np.sum(phase1_mask & near_critical) > 0:
                axins.scatter(Ca_sorted[phase1_mask & near_critical], 
                            theta_min_deg[phase1_mask & near_critical], 
                            color='blue', s=30, marker='s')
            if np.sum(phase2_mask) > 0:
                axins.scatter(Ca_sorted[phase2_mask], theta_min_deg[phase2_mask], 
                            color='darkgreen', s=80, marker='*')
            axins.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axins.set_xlabel('Ca', fontsize=10)
            axins.set_ylabel('$\\theta_{min}$ [deg]', fontsize=10)
            axins.grid(True, alpha=0.3)
            ax1.indicate_inset_zoom(axins, edgecolor="black")
    
    # Plot 2: x0 vs Ca
    valid_x0_mask = ~np.isnan(x0_sorted)
    
    if np.sum(valid_x0_mask) > 0:
        ax2.plot(Ca_sorted[valid_x0_mask], x0_sorted[valid_x0_mask], 'g-', linewidth=2.5, alpha=0.7)
        
        # Add scatter points
        if np.sum(phase0_mask & valid_x0_mask) > 0:
            ax2.scatter(Ca_sorted[phase0_mask & valid_x0_mask], 
                       x0_sorted[phase0_mask & valid_x0_mask], 
                       color='lightgreen', s=40, alpha=0.7, marker='o')
        if np.sum(phase1_mask & valid_x0_mask) > 0:
            ax2.scatter(Ca_sorted[phase1_mask & valid_x0_mask], 
                       x0_sorted[phase1_mask & valid_x0_mask], 
                       color='green', s=40, alpha=0.8, marker='s')
        if np.sum(phase2_mask & valid_x0_mask) > 0:
            ax2.scatter(Ca_sorted[phase2_mask & valid_x0_mask], 
                       x0_sorted[phase2_mask & valid_x0_mask], 
                       color='darkgreen', s=100, alpha=0.9, marker='*')
        
        ax2.set_xlabel('Ca (Capillary Number)', fontsize=14)
        ax2.set_ylabel('$x_0$ (Position at $\\theta_{min}$)', fontsize=14)
        ax2.set_title('$x_0$ vs Ca', fontsize=16, fontweight='bold')
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'No x0 data available', 
                 transform=ax2.transAxes, ha='center', va='center', fontsize=16)
        ax2.set_xlabel('Ca (Capillary Number)', fontsize=14)
        ax2.set_ylabel('$x_0$', fontsize=14)
        ax2.set_title('$x_0$ vs Ca', fontsize=16, fontweight='bold')
    
    plt.suptitle(f'Advanced Critical Ca Finding ($\\mu_r$ = {mu_r:.0e}, $\\lambda_{{slip}}$ = {lambda_slip:.0e})',
                 fontsize=18, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'critical_ca_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {plot_path}")


# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advanced critical Ca finder using three-phase approach')
    
    parser.add_argument('--mu_r', type=float, default=DEFAULT_MU_R,
                        help=f'Viscosity ratio μ_g/μ_l (default: {DEFAULT_MU_R})')
    parser.add_argument('--lambda_slip', type=float, default=DEFAULT_LAMBDA_SLIP,
                        help=f'Slip length (default: {DEFAULT_LAMBDA_SLIP})')
    parser.add_argument('--theta0', type=float, default=DEFAULT_THETA0*180/np.pi,
                        help=f'Initial contact angle in degrees (default: {DEFAULT_THETA0*180/np.pi:.0f})')
    parser.add_argument('--w', type=float, default=DEFAULT_W,
                        help=f'Curvature boundary condition at s=Delta (default: {DEFAULT_W})')
    parser.add_argument('--delta', type=float, default=DEFAULT_DELTA,
                        help=f'Domain size (default: {DEFAULT_DELTA})')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory for plots and data (default: output)')
    
    # Three separate tolerances
    parser.add_argument('--tolerance-init-ca', type=float, default=1e-6,
                        help='Tolerance for initial Ca_cr estimate (Phase 0)')
    parser.add_argument('--tolerance-s', type=float, default=1e-3,
                        help='Tolerance for theta_min in Phase 1 pseudo-arclength')
    parser.add_argument('--tolerance-th', type=float, default=1e-6,
                        help='Final tolerance for theta_min in Phase 2')
    
    args = parser.parse_args()
    
    # Convert theta0 to radians
    theta0_rad = args.theta0 * np.pi / 180
    
    # Run the advanced critical Ca finder
    Ca_critical = find_critical_ca_advanced(
        mu_r=args.mu_r,
        lambda_slip=args.lambda_slip,
        theta0=theta0_rad,
        w_bc=args.w,
        Delta=args.delta,
        output_dir=args.output_dir,
        tolerance_init_ca=args.tolerance_init_ca,
        tolerance_s=args.tolerance_s,
        tolerance_th=args.tolerance_th
    )
    
    if Ca_critical is not None:
        print(f"\n{'='*60}")
        print(f"FINAL RESULT: Ca_critical = {Ca_critical:.8f}")
        print(f"{'='*60}")
    else:
        print("\nFailed to find critical Ca")