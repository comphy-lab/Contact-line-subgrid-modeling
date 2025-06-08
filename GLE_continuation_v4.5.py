"""
/**
# GLE Continuation v4.5 - True Pseudo-Arclength Continuation

This module implements true pseudo-arclength continuation that can trace
through fold bifurcations by solving an extended system with arc-length
constraints.

Features:
- Full pseudo-arclength continuation with extended system
- Traces through fold bifurcations to capture unstable branches
- Creates S-shaped bifurcation curves
- Automatic fold detection and branch classification

Author: Vatsal Sanjay (vatsalsy@comphy-lab.org)
Date: 2025
*/
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, root
from scipy.integrate import solve_bvp
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Callable
import sys
import os
import argparse
import pickle

sys.path.append('src-local')
from gle_utils import GLE, boundary_conditions, f
from find_x0_utils import find_x0_and_theta_min, find_X_cl
from solution_types import SolutionResult


@dataclass
class ContinuationPoint:
  """Single point on the solution branch."""
  Ca: float
  X_cl: float  # Contact line position
  theta_min: float
  delta_X_cl: float  # X_cl - X_cl(Ca=0)
  arc_length: float  # Arc length parameter
  solution: object  # Full BVP solution
  tangent: Dict[str, float] = None  # Tangent vector
  converged: bool = True
  stability: str = 'unknown'  # 'stable', 'unstable', or 'unknown'


@dataclass
class ContinuationParams:
  """Parameters controlling the continuation algorithm."""
  mu_r: float
  lambda_slip: float
  theta0: float
  Delta: float = 10.0
  w_bc: float = 0.0
  tolerance: float = 1e-6
  initial_ds: float = 0.01
  max_ds: float = 0.05
  min_ds: float = 1e-5
  max_steps: int = 200
  max_newton_iter: int = 20
  verbose: bool = True


class GLEContinuationExtended:
  """
  True pseudo-arclength continuation solver for the GLE system.
  Solves extended system to trace through fold bifurcations.
  """
  
  def __init__(self, params: ContinuationParams):
    self.params = params
    self.X_cl_ref = None  # X_cl at Ca=0
    self.branch_history = []
    
  def solve_bvp_at_ca(self, Ca: float, s_guess: np.ndarray, y_guess: np.ndarray) -> Optional[object]:
    """Solve BVP at given Ca."""
    def ode_func(s, y):
      return GLE(s, y, Ca, self.params.mu_r, self.params.lambda_slip)
    
    def bc_func(ya, yb):
      return boundary_conditions(ya, yb, self.params.w_bc, self.params.theta0, self.params.lambda_slip)
    
    try:
      solution = solve_bvp(ode_func, bc_func, s_guess, y_guess, 
                          max_nodes=100000, tol=self.params.tolerance, verbose=0)
      return solution if solution.success else None
    except:
      return None
      
  def compute_solution_norm(self, sol1: np.ndarray, sol2: np.ndarray) -> float:
    """Compute norm between two solutions for arc-length constraint."""
    # Use a weighted norm focusing on theta values
    diff = sol2 - sol1
    # Weight theta differences more heavily
    weights = np.array([0.1, 1.0, 0.1])  # [h, theta, omega]
    weighted_diff = diff * weights[:, np.newaxis]
    return np.sqrt(np.sum(weighted_diff**2) / sol1.shape[1])
    
  def extended_system(self, vars: np.ndarray, point_ref: ContinuationPoint, 
                     ds: float, tangent: Dict) -> np.ndarray:
    """
    Extended system for pseudo-arclength continuation.
    
    Variables: [Ca, solution_coefficients...]
    Returns residuals for: [BVP_residuals..., arc_length_constraint]
    """
    Ca = vars[0]
    
    # Reconstruct solution from variables
    n_points = len(point_ref.solution.x)
    y_flat = vars[1:3*n_points+1]
    y_solution = y_flat.reshape((3, n_points))
    
    # Compute BVP residuals
    residuals = []
    
    # ODE residuals
    for i in range(1, n_points-1):
      s = point_ref.solution.x[i]
      y = y_solution[:, i]
      dyds_computed = GLE(s, y, Ca, self.params.mu_r, self.params.lambda_slip)
      
      # Finite difference approximation
      ds_forward = point_ref.solution.x[i+1] - s
      ds_backward = s - point_ref.solution.x[i-1]
      
      dyds_fd = np.zeros(3)
      for j in range(3):
        dyds_fd[j] = (y_solution[j, i+1] - y_solution[j, i-1]) / (ds_forward + ds_backward)
      
      residuals.extend(dyds_computed - dyds_fd)
    
    # Boundary condition residuals
    bc_residuals = boundary_conditions(
      y_solution[:, 0], y_solution[:, -1], 
      self.params.w_bc, self.params.theta0, self.params.lambda_slip
    )
    residuals.extend(bc_residuals)
    
    # Arc-length constraint
    # Calculate X_cl for current solution
    theta_vals = y_solution[1, :]
    s_vals = point_ref.solution.x
    x_vals = np.zeros_like(s_vals)
    for i in range(1, len(s_vals)):
      x_vals[i] = x_vals[i-1] + (s_vals[i] - s_vals[i-1]) * np.cos(theta_vals[i-1])
    X_cl_current = x_vals[-1]
    
    # Arc-length constraint: dot product with tangent = ds
    dCa = Ca - point_ref.Ca
    dX_cl = X_cl_current - point_ref.X_cl
    
    arc_constraint = (dCa * tangent['Ca'] + dX_cl * tangent['X_cl']) - ds
    residuals.append(arc_constraint)
    
    return np.array(residuals)
    
  def newton_extended(self, point_ref: ContinuationPoint, Ca_pred: float,
                     y_pred: np.ndarray, ds: float, tangent: Dict) -> Optional[Tuple[float, np.ndarray]]:
    """
    Newton iteration for the extended system.
    """
    # Initial guess
    vars_init = np.zeros(1 + 3 * len(point_ref.solution.x))
    vars_init[0] = Ca_pred
    vars_init[1:] = y_pred.flatten()
    
    # Solve extended system
    try:
      sol = root(
        lambda v: self.extended_system(v, point_ref, ds, tangent),
        vars_init,
        method='hybr',
        options={'maxfev': self.params.max_newton_iter * len(vars_init)}
      )
      
      if sol.success:
        Ca_new = sol.x[0]
        y_new = sol.x[1:].reshape((3, -1))
        return Ca_new, y_new
      else:
        return None
        
    except Exception as e:
      if self.params.verbose:
        print(f"  Newton iteration failed: {e}")
      return None
      
  def compute_tangent(self, p1: ContinuationPoint, p2: ContinuationPoint) -> Dict[str, float]:
    """Compute normalized tangent vector."""
    dCa = p2.Ca - p1.Ca
    dX_cl = p2.X_cl - p1.X_cl
    
    # Include solution change in tangent norm
    sol_norm = self.compute_solution_norm(p1.solution.y, p2.solution.y)
    
    # Normalize with all components
    norm = np.sqrt(dCa**2 + dX_cl**2 + sol_norm**2)
    
    if norm < 1e-12:
      return {'Ca': 0.0, 'X_cl': 1.0, 'sol_norm': 0.0}
      
    return {
      'Ca': dCa / norm,
      'X_cl': dX_cl / norm,
      'sol_norm': sol_norm / norm
    }
    
  def initialize_continuation(self, Ca_start: float) -> List[ContinuationPoint]:
    """Initialize continuation with reference point and first two solutions."""
    if self.params.verbose:
      print(f"\nInitializing extended pseudo-arclength continuation")
      print(f"Parameters: mu_r={self.params.mu_r}, lambda_slip={self.params.lambda_slip}")
      print(f"theta0={self.params.theta0*180/np.pi:.1f}°\n")
    
    # Get reference solution at Ca ≈ 0
    s_init = np.linspace(0, self.params.Delta, 10000)
    y_init = np.zeros((3, len(s_init)))
    y_init[0, :] = self.params.lambda_slip + s_init * np.sin(self.params.theta0)
    y_init[1, :] = self.params.theta0
    y_init[2, :] = 0.0
    
    result_ref = self.solve_bvp_at_ca(1e-6, s_init, y_init)
    if result_ref:
      theta_vals = result_ref.y[1, :]
      self.X_cl_ref = find_X_cl(result_ref.x, theta_vals)
      if self.params.verbose:
        print(f"Reference X_cl at Ca≈0: {self.X_cl_ref:.6f}")
    else:
      self.X_cl_ref = 0
      
    # Get first two solutions
    branch = []
    Ca_vals = [Ca_start, Ca_start + 0.001]
    
    for i, Ca in enumerate(Ca_vals):
      if i == 0:
        s_guess, y_guess = s_init, y_init
      else:
        s_guess = branch[0].solution.x
        y_guess = branch[0].solution.y
        
      result = self.solve_bvp_at_ca(Ca, s_guess, y_guess)
      if result is None:
        raise RuntimeError(f"Failed to get solution at Ca={Ca}")
        
      # Extract properties
      h_vals, theta_vals, w_vals = result.y
      x0, theta_min, _ = find_x0_and_theta_min(result.x, theta_vals)
      X_cl = find_X_cl(result.x, theta_vals)
      
      point = ContinuationPoint(
        Ca=Ca,
        X_cl=X_cl,
        theta_min=theta_min,
        delta_X_cl=X_cl - self.X_cl_ref,
        arc_length=i * self.params.initial_ds,
        solution=result
      )
      branch.append(point)
    
    # Compute initial tangent
    tangent = self.compute_tangent(branch[0], branch[1])
    branch[1].tangent = tangent
    
    return branch
    
  def trace_branch(self) -> List[ContinuationPoint]:
    """Main continuation loop with extended system."""
    # Initialize
    branch = self.initialize_continuation(0.001)
    ds = self.params.initial_ds
    current_point = branch[-1]
    
    fold_detected = False
    consecutive_failures = 0
    
    # Main loop
    for step in range(self.params.max_steps):
      # Update tangent
      if len(branch) >= 2:
        tangent = self.compute_tangent(branch[-2], branch[-1])
        current_point.tangent = tangent
      else:
        tangent = current_point.tangent
        
      # Predictor
      Ca_pred = current_point.Ca + ds * tangent['Ca']
      
      # Simple solution prediction
      y_pred = current_point.solution.y.copy()
      
      # Extended corrector
      result = self.newton_extended(current_point, Ca_pred, y_pred, ds, tangent)
      
      if result is None:
        consecutive_failures += 1
        # Reduce step size
        ds *= 0.5
        if ds < self.params.min_ds:
          if self.params.verbose:
            print(f"\nMinimum step size reached at step {step}")
          break
        if self.params.verbose and consecutive_failures == 1:
          print(f"  Reducing step size to {ds:.6f}")
        continue
      
      consecutive_failures = 0
      Ca_new, y_new = result
      
      # Create solution object
      sol_new = type('obj', (object,), {
        'x': current_point.solution.x,
        'y': y_new,
        'success': True
      })()
      
      # Extract properties
      theta_vals = y_new[1, :]
      x0, theta_min, _ = find_x0_and_theta_min(sol_new.x, theta_vals)
      X_cl = find_X_cl(sol_new.x, theta_vals)
      
      # Create new point
      new_point = ContinuationPoint(
        Ca=Ca_new,
        X_cl=X_cl,
        theta_min=theta_min,
        delta_X_cl=X_cl - self.X_cl_ref,
        arc_length=current_point.arc_length + ds,
        solution=sol_new,
        tangent=tangent
      )
      
      branch.append(new_point)
      
      # Print progress
      if self.params.verbose:
        print(f"Step {len(branch)-1}: Ca={new_point.Ca:.6f}, "
              f"X_cl={new_point.X_cl:.4f}, δX_cl={new_point.delta_X_cl:.4f}, "
              f"θ_min={new_point.theta_min*180/np.pi:.1f}°")
      
      # Check for fold
      if len(branch) >= 3 and not fold_detected:
        dCa_prev = branch[-2].Ca - branch[-3].Ca
        dCa_curr = branch[-1].Ca - branch[-2].Ca
        if dCa_prev * dCa_curr < 0:
          fold_detected = True
          if self.params.verbose:
            print(f"\n*** FOLD DETECTED at Ca ≈ {branch[-1].Ca:.6f} ***\n")
          # Mark stability
          for i, p in enumerate(branch):
            if i <= len(branch) - 2:
              p.stability = 'stable'
            else:
              p.stability = 'unstable'
              
      # Adaptive step size
      if consecutive_failures == 0:
        ds = min(ds * 1.2, self.params.max_ds)
        
      # Update current point
      current_point = new_point
      
      # Stop conditions
      if new_point.theta_min < 0.001:
        if self.params.verbose:
          print(f"\nReached θ_min ≈ 0 at Ca={new_point.Ca:.6f}")
        break
        
      if fold_detected and new_point.Ca < 0.001:
        if self.params.verbose:
          print(f"\nReturned to low Ca after fold")
        break
        
    self.branch_history = branch
    return branch
    
  def plot_results(self, branch: List[ContinuationPoint], output_dir: str = 'output'):
    """Create bifurcation diagram showing S-shaped curve."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    Ca_vals = np.array([p.Ca for p in branch])
    delta_X_vals = np.array([p.delta_X_cl for p in branch])
    theta_min_vals = np.array([p.theta_min * 180/np.pi for p in branch])
    
    # Find fold point
    fold_idx = None
    for i in range(1, len(branch)-1):
      if (Ca_vals[i] - Ca_vals[i-1]) * (Ca_vals[i+1] - Ca_vals[i]) < 0:
        fold_idx = i
        break
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot 1: δX_cl vs Ca (S-shaped curve)
    if fold_idx:
      # Split into stable/unstable branches
      ax1.plot(Ca_vals[:fold_idx+1], delta_X_vals[:fold_idx+1], 
              'b-', linewidth=3, label='Stable branch')
      ax1.plot(Ca_vals[fold_idx:], delta_X_vals[fold_idx:], 
              'b--', linewidth=2.5, alpha=0.6, label='Unstable branch')
      ax1.plot(Ca_vals[fold_idx], delta_X_vals[fold_idx], 'ro', 
              markersize=10, label=f'Fold at Ca={Ca_vals[fold_idx]:.4f}')
    else:
      ax1.plot(Ca_vals, delta_X_vals, 'b-', linewidth=3)
    
    ax1.set_xlabel('Ca', fontsize=14)
    ax1.set_ylabel('δX_cl', fontsize=14)
    ax1.set_title('Contact Line Displacement vs Capillary Number (Extended Pseudo-Arclength)', 
                 fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=12)
    
    # Add parameter box
    textstr = (f'μ_r = {self.params.mu_r:.1e}\n'
              f'λ_slip = {self.params.lambda_slip:.1e}\n'
              f'θ₀ = {self.params.theta0*180/np.pi:.0f}°')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    # Plot 2: θ_min vs Ca
    if fold_idx:
      ax2.plot(Ca_vals[:fold_idx+1], theta_min_vals[:fold_idx+1], 
              'g-', linewidth=3)
      ax2.plot(Ca_vals[fold_idx:], theta_min_vals[fold_idx:], 
              'g--', linewidth=2.5, alpha=0.6)
      ax2.plot(Ca_vals[fold_idx], theta_min_vals[fold_idx], 'ro', markersize=10)
    else:
      ax2.plot(Ca_vals, theta_min_vals, 'g-', linewidth=3)
      
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Ca', fontsize=14)
    ax2.set_ylabel('θ_min [degrees]', fontsize=14)
    ax2.set_title('Minimum Contact Angle vs Capillary Number', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=-5)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'bifurcation_diagram_extended.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    if self.params.verbose:
      print(f"\nBifurcation diagram saved to: {plot_file}")
    
    # Save data
    data_file = os.path.join(output_dir, 'bifurcation_data_extended.txt')
    with open(data_file, 'w') as f:
      f.write("# Extended pseudo-arclength continuation results\n")
      f.write(f"# mu_r={self.params.mu_r}, lambda_slip={self.params.lambda_slip}, ")
      f.write(f"theta0={self.params.theta0*180/np.pi}°\n")
      f.write("# Ca, X_cl, delta_X_cl, theta_min_deg, arc_length, stability\n")
      for p in branch:
        f.write(f"{p.Ca:.8f} {p.X_cl:.8f} {p.delta_X_cl:.8f} ")
        f.write(f"{p.theta_min*180/np.pi:.4f} {p.arc_length:.6f} {p.stability}\n")
    
    if self.params.verbose:
      print(f"Data saved to: {data_file}")


def main():
  """Main entry point."""
  parser = argparse.ArgumentParser(
    description='Extended pseudo-arclength continuation for GLE',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  
  # Physics parameters
  parser.add_argument('--mu_r', type=float, default=1e-6,
                      help='Viscosity ratio μ_g/μ_l')
  parser.add_argument('--lambda_slip', type=float, default=1e-3,
                      help='Slip length parameter λ')
  parser.add_argument('--theta0', type=float, default=60,
                      help='Initial contact angle in degrees')
  parser.add_argument('--Delta', type=float, default=10.0,
                      help='Domain size')
  parser.add_argument('--w_bc', type=float, default=0.0,
                      help='Boundary condition for omega')
  
  # Algorithm parameters
  parser.add_argument('--initial-ds', type=float, default=0.01,
                      help='Initial step size')
  parser.add_argument('--max-ds', type=float, default=0.05,
                      help='Maximum step size')
  parser.add_argument('--min-ds', type=float, default=1e-5,
                      help='Minimum step size')
  parser.add_argument('--max-steps', type=int, default=200,
                      help='Maximum continuation steps')
  parser.add_argument('--tolerance', type=float, default=1e-6,
                      help='Solver tolerance')
  parser.add_argument('--quiet', action='store_true',
                      help='Suppress progress output')
  
  args = parser.parse_args()
  
  # Create parameters
  params = ContinuationParams(
    mu_r=args.mu_r,
    lambda_slip=args.lambda_slip,
    theta0=np.radians(args.theta0),
    Delta=args.Delta,
    w_bc=args.w_bc,
    tolerance=args.tolerance,
    initial_ds=args.initial_ds,
    max_ds=args.max_ds,
    min_ds=args.min_ds,
    max_steps=args.max_steps,
    verbose=not args.quiet
  )
  
  # Run continuation
  solver = GLEContinuationExtended(params)
  branch = solver.trace_branch()
  
  # Plot results
  solver.plot_results(branch)
  
  # Print summary
  if params.verbose:
    print(f"\n{'='*60}")
    print(f"Extended pseudo-arclength continuation completed")
    print(f"Traced {len(branch)} points on solution branch")
    if len(branch) > 0:
      print(f"Ca range: {branch[0].Ca:.6f} to {branch[-1].Ca:.6f}")
      
      # Find critical Ca
      Ca_max = max(p.Ca for p in branch)
      critical_point = next(p for p in branch if p.Ca == Ca_max)
      print(f"Critical Ca (fold): {Ca_max:.6f}")
      print(f"θ_min at fold: {critical_point.theta_min*180/np.pi:.2f}°")
      print(f"δX_cl at fold: {critical_point.delta_X_cl:.4f}")
      
      # Count stable/unstable
      stable = sum(1 for p in branch if p.stability == 'stable')
      unstable = sum(1 for p in branch if p.stability == 'unstable')
      print(f"Stable points: {stable}")
      print(f"Unstable points: {unstable}")
    print(f"{'='*60}")


if __name__ == '__main__':
  main()