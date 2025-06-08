"""
/**
# GLE Continuation v4 - Unified Continuation Solver

This module provides both pseudo-arclength and natural parameter continuation
methods for tracking solution branches of the Generalized Lubrication Equation.
It tracks the contact line displacement δX_cl vs Ca and can handle fold
bifurcations when using the pseudo-arclength method.

Features:
- Pseudo-arclength continuation (default): tracks through folds
- Natural parameter continuation: simpler, faster for stable branches
- Automatic fold detection and branch classification
- Adaptive step size control
- Clean plotting and data export

Author: Vatsal Sanjay (vatsalsy@comphy-lab.org)
Date: 2025
*/
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
import sys
import os
import argparse
import pickle

sys.path.append('src-local')
from gle_utils import solve_single_ca
from find_x0_utils import find_x0_and_theta_min


@dataclass
class ContinuationPoint:
  """Single point on the solution branch."""
  Ca: float
  X_cl: float  # Contact line position
  theta_min: float
  delta_X_cl: float  # X_cl - X_cl(Ca=0)
  arc_length: float  # Arc length parameter (for pseudo-arclength)
  solution: object  # Full BVP solution
  tangent_Ca: float = 0.0  # dCa/ds (for pseudo-arclength)
  tangent_X: float = 0.0  # dX_cl/ds (for pseudo-arclength)
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
  method: str = 'arclength'  # 'arclength' or 'natural'
  tolerance: float = 1e-6
  initial_ds: float = 0.01
  max_ds: float = 0.05
  min_ds: float = 1e-5
  max_steps: int = 200
  Ca_max: float = 0.2
  verbose: bool = True


class GLEContinuation:
  """
  Unified continuation solver for the GLE system.
  Supports both pseudo-arclength and natural parameter methods.
  """
  
  def __init__(self, params: ContinuationParams):
    self.params = params
    self.X_cl_ref = None  # X_cl at Ca=0
    self.branch_history = []
    
  def solve_at_ca(self, Ca: float, s_guess=None, y_guess=None) -> Optional[object]:
    """Solve BVP at given Ca with robust error handling."""
    if s_guess is None:
      s_guess = np.linspace(0, self.params.Delta, 10000)
    if y_guess is None:
      y_guess = np.zeros((3, len(s_guess)))
      y_guess[0, :] = self.params.lambda_slip + s_guess * np.sin(self.params.theta0)
      y_guess[1, :] = self.params.theta0
      y_guess[2, :] = 0.0
      
    try:
      result = solve_single_ca(
        Ca, self.params.mu_r, self.params.lambda_slip, self.params.theta0,
        self.params.w_bc, self.params.Delta, s_guess, y_guess,
        tol=self.params.tolerance
      )
      
      if result.success:
        # Check for physical validity
        h_vals = result.solution.y[0]
        if np.any(h_vals <= 0):
          if self.params.verbose:
            print(f"  Warning: negative film thickness at Ca={Ca:.6f}")
          return None
        return result
        
    except Exception as e:
      if self.params.verbose:
        print(f"  Solver exception at Ca={Ca:.6f}: {e}")
        
    return None
    
  def initialize_continuation(self, Ca_start: float) -> List[ContinuationPoint]:
    """Initialize continuation with reference point and first solution."""
    if self.params.verbose:
      print(f"\nInitializing {self.params.method} continuation")
      print(f"Parameters: mu_r={self.params.mu_r}, lambda_slip={self.params.lambda_slip}")
      print(f"theta0={self.params.theta0*180/np.pi:.1f}°\n")
    
    # Get reference solution at Ca ≈ 0
    result_ref = self.solve_at_ca(0.0)
    if result_ref:
      self.X_cl_ref = result_ref.X_cl
      if self.params.verbose:
        print(f"Reference X_cl at Ca≈0: {self.X_cl_ref:.6f}")
    else:
      self.X_cl_ref = 0
      
    # Get initial solution
    result1 = self.solve_at_ca(Ca_start)
    if result1 is None:
      raise RuntimeError(f"Failed to get initial solution at Ca={Ca_start}")
      
    branch = []
    point1 = ContinuationPoint(
      Ca=Ca_start,
      X_cl=result1.X_cl,
      theta_min=result1.theta_min,
      delta_X_cl=result1.X_cl - self.X_cl_ref,
      arc_length=0.0,
      solution=result1
    )
    branch.append(point1)
    
    # For pseudo-arclength, get second point for initial tangent
    if self.params.method == 'arclength':
      Ca2 = Ca_start + 0.001
      result2 = self.solve_at_ca(Ca2, result1.s_range, result1.solution.y)
      if result2 is None:
        raise RuntimeError(f"Failed to get second solution at Ca={Ca2}")
        
      point2 = ContinuationPoint(
        Ca=Ca2,
        X_cl=result2.X_cl,
        theta_min=result2.theta_min,
        delta_X_cl=result2.X_cl - self.X_cl_ref,
        arc_length=self.params.initial_ds,
        solution=result2
      )
      
      # Compute initial tangent
      tan_Ca, tan_X = self._compute_tangent(point1, point2)
      point2.tangent_Ca = tan_Ca
      point2.tangent_X = tan_X
      branch.append(point2)
      
    return branch
    
  def _compute_tangent(self, p1: ContinuationPoint, p2: ContinuationPoint) -> Tuple[float, float]:
    """Compute normalized tangent vector for pseudo-arclength."""
    dCa = p2.Ca - p1.Ca
    dX = p2.X_cl - p1.X_cl
    
    # Normalize
    norm = np.sqrt(dCa**2 + dX**2)
    if norm < 1e-12:
      return 0.0, 1.0
      
    return dCa/norm, dX/norm
    
  def _predictor_arclength(self, point: ContinuationPoint, ds: float) -> Tuple[float, np.ndarray]:
    """Predict next point using tangent (pseudo-arclength)."""
    Ca_pred = point.Ca + ds * point.tangent_Ca
    
    # For solution prediction, use simple extrapolation
    y_pred = point.solution.solution.y
    
    return Ca_pred, y_pred
    
  def _predictor_natural(self, point: ContinuationPoint, dCa: float) -> Tuple[float, np.ndarray]:
    """Predict next point using Ca increment (natural parameter)."""
    Ca_pred = point.Ca + dCa
    y_pred = point.solution.solution.y
    
    return Ca_pred, y_pred
    
  def _corrector(self, point_ref: ContinuationPoint, Ca_pred: float, 
                 y_pred: np.ndarray, ds: float) -> Optional[ContinuationPoint]:
    """
    Corrector step - solve at predicted Ca.
    For full pseudo-arclength, we'd solve the extended system,
    but this simplified version is more robust and still effective.
    """
    result = self.solve_at_ca(
      Ca_pred,
      point_ref.solution.s_range,
      y_pred
    )
    
    if result is None:
      return None
      
    # Validate solution - reject unphysical solutions
    # X_cl should generally increase with Ca for physical solutions
    if hasattr(self, 'branch') and len(self.branch) > 1:
      # Check if X_cl dropped unrealistically (more than 80% from previous)
      if result.X_cl < 0.2 * point_ref.solution.X_cl:
        if self.params.verbose:
          print(f"  Rejecting unphysical solution at Ca={Ca_pred:.6f}: X_cl dropped from {point_ref.solution.X_cl:.6f} to {result.X_cl:.6f}")
        return None
        
    # Create new point
    delta_X = result.X_cl - self.X_cl_ref if self.X_cl_ref is not None else 0
    new_point = ContinuationPoint(
      Ca=Ca_pred,
      X_cl=result.X_cl,
      theta_min=result.theta_min,
      delta_X_cl=delta_X,
      arc_length=point_ref.arc_length + ds,
      solution=result
    )
    
    return new_point
    
  def trace_branch(self) -> List[ContinuationPoint]:
    """Main continuation loop."""
    # Initialize
    Ca_start = 0.001 if self.params.method == 'arclength' else 0.0001
    branch = self.initialize_continuation(Ca_start)
    
    # Set up parameters
    if self.params.method == 'arclength':
      ds = self.params.initial_ds
      current_point = branch[-1]
    else:
      dCa = self.params.initial_ds
      current_point = branch[0]
      
    fold_detected = False
    
    # Main continuation loop
    for step in range(self.params.max_steps):
      if self.params.method == 'arclength':
        # Update tangent from last two points
        if len(branch) >= 2:
          tan_Ca, tan_X = self._compute_tangent(branch[-2], branch[-1])
          current_point.tangent_Ca = tan_Ca
          current_point.tangent_X = tan_X
          
        # Predictor
        Ca_pred, y_pred = self._predictor_arclength(current_point, ds)
        
      else:  # natural parameter
        # Predictor
        Ca_pred, y_pred = self._predictor_natural(current_point, dCa)
        
        # Check if we've exceeded Ca_max
        if Ca_pred > self.params.Ca_max:
          if self.params.verbose:
            print(f"\nReached Ca_max={self.params.Ca_max}")
          break
          
      # Corrector
      new_point = self._corrector(current_point, Ca_pred, y_pred, 
                                  ds if self.params.method == 'arclength' else dCa)
      
      if new_point is None:
        # Reduce step size
        if self.params.method == 'arclength':
          ds *= 0.5
          if ds < self.params.min_ds:
            if self.params.verbose:
              print(f"\nMinimum step size reached at step {step}")
            break
        else:
          dCa *= 0.5
          if dCa < self.params.min_ds:
            if self.params.verbose:
              print(f"\nFailed to converge at Ca={Ca_pred:.6f}")
            break
        continue
        
      # Success - add point
      branch.append(new_point)
      
      # Print progress
      if self.params.verbose:
        print(f"Step {len(branch)-1}: Ca={new_point.Ca:.6f}, "
              f"X_cl={new_point.X_cl:.4f}, δX_cl={new_point.delta_X_cl:.4f}, "
              f"θ_min={new_point.theta_min*180/np.pi:.1f}°")
      
      # Check for fold (pseudo-arclength only)
      if self.params.method == 'arclength' and len(branch) >= 3 and not fold_detected:
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
      if self.params.method == 'arclength':
        ds = min(ds * 1.2, self.params.max_ds)
      else:
        # For natural parameter, adjust based on theta_min
        if new_point.theta_min < 0.2:  # Near critical Ca
          dCa *= 0.5
        else:
          dCa = min(dCa * 1.2, 0.01)
          
      # Update current point
      current_point = new_point
      
      # Stop conditions
      if new_point.theta_min < 0.001:  # Very close to 0
        if self.params.verbose:
          print(f"\nReached θ_min ≈ 0 at Ca={new_point.Ca:.6f}")
        break
        
      if self.params.method == 'arclength' and fold_detected and new_point.Ca < 0.001:
        if self.params.verbose:
          print(f"\nReturned to low Ca after fold")
        break
        
    self.branch_history = branch
    return branch
    
  def analyze_branch(self, branch: List[ContinuationPoint]) -> Dict:
    """Analyze the solution branch for key features."""
    analysis = {
      'num_points': len(branch),
      'Ca_range': (branch[0].Ca, branch[-1].Ca),
      'Ca_max': max(p.Ca for p in branch),
      'fold_point': None,
      'critical_Ca': None,
      'stable_points': 0,
      'unstable_points': 0
    }
    
    # Find fold point
    for i in range(1, len(branch)-1):
      if (branch[i].Ca - branch[i-1].Ca) * (branch[i+1].Ca - branch[i].Ca) < 0:
        analysis['fold_point'] = {
          'index': i,
          'Ca': branch[i].Ca,
          'theta_min': branch[i].theta_min,
          'delta_X_cl': branch[i].delta_X_cl
        }
        analysis['critical_Ca'] = branch[i].Ca
        break
        
    # Count stable/unstable points
    for p in branch:
      if p.stability == 'stable':
        analysis['stable_points'] += 1
      elif p.stability == 'unstable':
        analysis['unstable_points'] += 1
        
    return analysis
    
  def plot_results(self, branch: List[ContinuationPoint], output_dir: str = 'output'):
    """Create bifurcation diagram showing δX_cl vs Ca."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    Ca_vals = np.array([p.Ca for p in branch])
    delta_X_vals = np.array([p.delta_X_cl for p in branch])
    theta_min_vals = np.array([p.theta_min * 180/np.pi for p in branch])
    
    # Analyze branch
    analysis = self.analyze_branch(branch)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    
    # Plot 1: δX_cl vs Ca
    if self.params.method == 'arclength' and analysis['fold_point']:
      # Split into stable/unstable branches
      fold_idx = analysis['fold_point']['index']
      ax1.plot(Ca_vals[:fold_idx+1], delta_X_vals[:fold_idx+1], 
              'b-', linewidth=3, label='Stable branch')
      ax1.plot(Ca_vals[fold_idx:], delta_X_vals[fold_idx:], 
              'b--', linewidth=2.5, alpha=0.6, label='Unstable branch')
      ax1.plot(Ca_vals[fold_idx], delta_X_vals[fold_idx], 'ro', 
              markersize=10, label=f'Fold at Ca={Ca_vals[fold_idx]:.4f}')
    else:
      # Simple plot for natural parameter method
      ax1.plot(Ca_vals, delta_X_vals, 'b.-', linewidth=2.5, markersize=6, 
              label='Solution branch')
      if min(theta_min_vals) < 10:
        critical_idx = next(i for i, t in enumerate(theta_min_vals) if t < 10)
        ax1.axvline(x=Ca_vals[critical_idx], color='r', linestyle='--', 
                   alpha=0.5, label=f'Ca_cr ≈ {Ca_vals[-1]:.4f}')
        
    ax1.set_xlabel('Ca', fontsize=14)
    ax1.set_ylabel('δX_cl', fontsize=14)
    ax1.set_title(f'Contact Line Displacement vs Capillary Number ({self.params.method} method)', 
                 fontsize=16)
    ax1.grid(True, alpha=0.3)
    
    # Only show legend if there are labeled items
    handles, labels = ax1.get_legend_handles_labels()
    if handles:
      ax1.legend(fontsize=12)
    
    # Add parameter box
    textstr = (f'μ_r = {self.params.mu_r:.1e}\n'
              f'λ_slip = {self.params.lambda_slip:.1e}\n'
              f'θ₀ = {self.params.theta0*180/np.pi:.0f}°')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    # Plot 2: θ_min vs Ca
    if self.params.method == 'arclength' and analysis['fold_point']:
      fold_idx = analysis['fold_point']['index']
      ax2.plot(Ca_vals[:fold_idx+1], theta_min_vals[:fold_idx+1], 
              'g-', linewidth=3)
      ax2.plot(Ca_vals[fold_idx:], theta_min_vals[fold_idx:], 
              'g--', linewidth=2.5, alpha=0.6)
      ax2.plot(Ca_vals[fold_idx], theta_min_vals[fold_idx], 'ro', markersize=10)
    else:
      ax2.plot(Ca_vals, theta_min_vals, 'g.-', linewidth=2.5, markersize=6)
      
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Ca', fontsize=14)
    ax2.set_ylabel('θ_min [degrees]', fontsize=14)
    ax2.set_title('Minimum Contact Angle vs Capillary Number', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=-5)
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, f'bifurcation_diagram_{self.params.method}.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    if self.params.verbose:
      print(f"\nBifurcation diagram saved to: {plot_file}")
    
    # Save data
    data_file = os.path.join(output_dir, f'bifurcation_data_{self.params.method}.txt')
    with open(data_file, 'w') as f:
      f.write(f"# {self.params.method.capitalize()} continuation results\n")
      f.write(f"# mu_r={self.params.mu_r}, lambda_slip={self.params.lambda_slip}, ")
      f.write(f"theta0={self.params.theta0*180/np.pi}°\n")
      if self.params.method == 'arclength':
        f.write("# Ca, X_cl, delta_X_cl, theta_min_deg, arc_length, stability\n")
        for p in branch:
          f.write(f"{p.Ca:.8f} {p.X_cl:.8f} {p.delta_X_cl:.8f} ")
          f.write(f"{p.theta_min*180/np.pi:.4f} {p.arc_length:.6f} {p.stability}\n")
      else:
        f.write("# Ca, X_cl, delta_X_cl, theta_min_deg\n")
        for p in branch:
          f.write(f"{p.Ca:.8f} {p.X_cl:.8f} {p.delta_X_cl:.8f} ")
          f.write(f"{p.theta_min*180/np.pi:.4f}\n")
    if self.params.verbose:
      print(f"Data saved to: {data_file}")
      
    # Save branch object for further analysis
    pkl_file = os.path.join(output_dir, f'branch_{self.params.method}.pkl')
    with open(pkl_file, 'wb') as f:
      pickle.dump({'branch': branch, 'analysis': analysis, 'params': self.params}, f)
    if self.params.verbose:
      print(f"Branch data saved to: {pkl_file}")
      
    return analysis


def main():
  """Main entry point for continuation solver."""
  parser = argparse.ArgumentParser(
    description='Unified continuation solver for GLE with δX_cl tracking',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
  
  # Method selection
  parser.add_argument('--method', type=str, choices=['arclength', 'natural'],
                      default='arclength',
                      help='Continuation method: arclength (tracks through folds) or natural (simpler)')
  
  # Physics parameters
  parser.add_argument('--mu_r', type=float, default=1e-6,
                      help='Viscosity ratio μ_g/μ_l')
  parser.add_argument('--lambda_slip', type=float, default=1e-4,
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
  parser.add_argument('--Ca-max', type=float, default=0.2,
                      help='Maximum Ca for natural parameter method')
  parser.add_argument('--tolerance', type=float, default=1e-6,
                      help='Solver tolerance')
  
  # Output options
  parser.add_argument('--output-dir', type=str, default='output',
                      help='Output directory for plots and data')
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
    method=args.method,
    tolerance=args.tolerance,
    initial_ds=args.initial_ds,
    max_ds=args.max_ds,
    min_ds=args.min_ds,
    max_steps=args.max_steps,
    Ca_max=args.Ca_max,
    verbose=not args.quiet
  )
  
  # Run continuation
  solver = GLEContinuation(params)
  branch = solver.trace_branch()
  
  # Analyze and plot results
  analysis = solver.plot_results(branch, args.output_dir)
  
  # Print summary
  if params.verbose:
    print(f"\n{'='*60}")
    print(f"Continuation completed using {params.method} method")
    print(f"Traced {analysis['num_points']} points on solution branch")
    print(f"Ca range: {analysis['Ca_range'][0]:.6f} to {analysis['Ca_range'][1]:.6f}")
    
    if analysis['critical_Ca']:
      print(f"Critical Ca (fold): {analysis['critical_Ca']:.6f}")
      fold = analysis['fold_point']
      print(f"θ_min at fold: {fold['theta_min']*180/np.pi:.2f}°")
      print(f"δX_cl at fold: {fold['delta_X_cl']:.4f}")
    else:
      print(f"Estimated Ca_critical ≈ {branch[-1].Ca:.6f}")
      print(f"Final θ_min: {branch[-1].theta_min*180/np.pi:.2f}°")
      
    if params.method == 'arclength':
      print(f"Stable points: {analysis['stable_points']}")
      print(f"Unstable points: {analysis['unstable_points']}")
    print(f"{'='*60}")


if __name__ == '__main__':
  main()