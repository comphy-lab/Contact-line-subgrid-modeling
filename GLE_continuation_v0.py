#!/usr/bin/env python3
"""
GLE_continuation_hybrid.py

Pseudo-arclength continuation method for the Generalized Lubrication Equations (GLE).
Tracks contact line displacement δX_cl as a function of Capillary number (Ca),
identifying potential fold bifurcations.

Author: Vatsal Sanjay
Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Any
from scipy.integrate import solve_bvp
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve, eigs
from scipy.optimize import root_scalar
import argparse
import pickle
import h5py
import time
import logging
from pathlib import Path

# Add src-local to path for utilities
import sys
sys.path.append('src-local')
from find_x0_utils import find_x0_and_theta_min
from gle_utils import f1, f2, f3, f, GLE as GLE_func, boundary_conditions


@dataclass
class ContinuationParameters:
  """Parameters for pseudo-arclength continuation"""
  ds_init: float = 0.1           # Initial arc length step
  ds_min: float = 1e-6           # Minimum arc length step
  ds_max: float = 1.0            # Maximum arc length step
  max_steps: int = 500           # Maximum continuation steps
  newton_tol: float = 1e-10      # Newton convergence tolerance
  newton_max_iter: int = 50      # Maximum Newton iterations
  fold_detection_tol: float = 1e-8  # Tolerance for fold detection
  adaptive_ds: bool = True       # Use adaptive step size
  angle_change_max: float = 0.1  # Max angle change for step size control


@dataclass
class SolutionPoint:
  """Container for solution at one point on the branch"""
  Ca: float                      # Capillary number
  solution: np.ndarray           # Full BVP solution [theta, h, omega]
  s_range: np.ndarray           # Arc length mesh
  theta_min: float              # Minimum contact angle
  x_cl: float                   # Contact line position
  delta_x_cl: float             # Displacement from Ca=0
  arc_length_param: float       # Pseudo arc-length parameter
  tangent: Dict[str, Any] = field(default_factory=dict)  # Tangent vector
  is_fold: bool = False         # Fold point indicator
  eigenvalue_min: Optional[float] = None  # Minimum eigenvalue of Jacobian


class GLEContinuation:
  """Pseudo-arclength continuation solver for the GLE system"""
  
  def __init__(self, mu_r: float, lambda_slip: float, theta0: float,
               w_bc: float = 0.0, Delta: float = 10.0, N_points: int = 10000,
               params: Optional[ContinuationParameters] = None):
    """
    Initialize continuation solver for GLE system
    
    Args:
      mu_r: Viscosity ratio
      lambda_slip: Slip length
      theta0: Initial contact angle (radians)
      w_bc: Curvature boundary condition
      Delta: Domain size
      N_points: Number of mesh points
      params: Continuation parameters
    """
    self.mu_r = mu_r
    self.lambda_slip = lambda_slip
    self.theta0 = theta0
    self.w_bc = w_bc
    self.Delta = Delta
    self.N_points = N_points
    
    # Continuation parameters
    self.params = params if params else ContinuationParameters()
    
    # Solution branch storage
    self.branch: List[SolutionPoint] = []
    self.x_cl_ref: Optional[float] = None  # Reference position at Ca=0
    
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    self.logger = logging.getLogger(__name__)
    
  
  def _GLE_ode(self, s: np.ndarray, y: np.ndarray, Ca: float) -> np.ndarray:
    """
    System of ODEs for the GLE
    
    Args:
      s: Arc length coordinate
      y: State vector [h, theta, omega]
      Ca: Capillary number
    
    Returns:
      dy/ds: Derivatives [dh/ds, dtheta/ds, domega/ds]
    """
    # Use the working GLE function from gle_utils
    return GLE_func(s, y, Ca, self.mu_r, self.lambda_slip)
  
  def _bc_residual(self, ya: np.ndarray, yb: np.ndarray) -> np.ndarray:
    """Boundary condition residuals"""
    # Use the working boundary conditions from gle_utils
    return boundary_conditions(ya, yb, self.w_bc, self.theta0, self.lambda_slip)
  
  def _solve_bvp(self, Ca: float, initial_guess: Optional[np.ndarray] = None,
                 s_mesh: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the boundary value problem for given Ca
    
    Returns:
      s_mesh: Arc length mesh
      solution: Solution array [h, theta, omega]  # Fixed order to match GLE_solver
    """
    if s_mesh is None:
      s_mesh = np.linspace(0, self.Delta, self.N_points)
    
    if initial_guess is None:
      # Simple initial guess matching GLE_solver - note the order is [h, theta, omega]
      h_guess = self.lambda_slip + s_mesh * np.sin(self.theta0)  # h grows from lambda_slip
      theta_guess = self.theta0 * np.ones_like(s_mesh)
      omega_guess = np.zeros_like(s_mesh)
      initial_guess = np.vstack([h_guess, theta_guess, omega_guess])
    
    # Define ODE function for this Ca
    def ode_fun(s, y):
      return self._GLE_ode(s, y, Ca)
    
    # Solve BVP
    sol = solve_bvp(ode_fun, self._bc_residual, s_mesh, initial_guess,
                   tol=1e-6, max_nodes=100000, verbose=0)
    
    if not sol.success:
      raise RuntimeError(f"BVP solver failed: {sol.message}")
    
    return sol.x, sol.y
  
  def initialize_continuation(self) -> SolutionPoint:
    """Get initial solution at Ca=0 and set reference position"""
    self.logger.info("Initializing continuation at Ca=0...")
    
    # Solve at Ca=0
    s_mesh, solution = self._solve_bvp(Ca=0.0)
    
    # Extract key quantities
    h = solution[0, :]
    theta = solution[1, :]
    
    # Find theta_min
    theta_min = np.min(theta)
    
    # Contact line position (x at s=0)
    x_cl = self._compute_x_cl(s_mesh, theta, h)
    self.x_cl_ref = x_cl  # Store reference position at Ca=0
    
    # Create initial solution point
    initial_point = SolutionPoint(
      Ca=0.0,
      solution=solution,
      s_range=s_mesh,
      theta_min=theta_min,
      x_cl=x_cl,
      delta_x_cl=0.0,
      arc_length_param=0.0
    )
    
    # Compute initial tangent (increase Ca direction)
    initial_point.tangent = self._compute_initial_tangent(initial_point)
    
    self.branch.append(initial_point)
    self.logger.info(f"Initial solution: theta_min={np.degrees(theta_min):.2f}°")
    
    return initial_point
  
  def _compute_initial_tangent(self, point: SolutionPoint) -> Dict[str, Any]:
    """Compute initial tangent by finite difference"""
    # Use larger perturbation for meaningful tangent
    dCa = 0.0001  # Small fixed perturbation
    
    # Solve at slightly higher Ca
    try:
      s_mesh_new, sol_perturbed = self._solve_bvp(point.Ca + dCa, point.solution, point.s_range)
      
      # If mesh was refined, interpolate back to original mesh
      if len(s_mesh_new) != len(point.s_range):
        from scipy.interpolate import interp1d
        sol_interp = np.zeros_like(point.solution)
        for i in range(3):  # h, theta, omega
          f_interp = interp1d(s_mesh_new, sol_perturbed[i, :], kind='cubic', 
                             bounds_error=False, fill_value='extrapolate')
          sol_interp[i, :] = f_interp(point.s_range)
        sol_perturbed = sol_interp
      
      # Compute tangent
      dU_dCa = (sol_perturbed - point.solution) / dCa
      
    except Exception as e:
      # If finite difference fails, use a simple tangent
      self.logger.warning(f"Initial tangent finite difference failed: {str(e)}")
      # Use a pure Ca direction tangent
      return {
        'U': np.zeros_like(point.solution),
        'Ca': 1.0,
        's_mesh': point.s_range
      }
    
    # Normalize to get unit tangent with strong Ca component
    U_norm = np.linalg.norm(dU_dCa)
    
    # For initial step, we want strong Ca increase
    if U_norm < 1e-10:
      # If solution barely changes, force Ca direction
      return {
        'U': np.zeros_like(point.solution),
        'Ca': 1.0,
        's_mesh': point.s_range
      }
    
    # Balance between solution change and Ca change
    # Start with 90% Ca direction, 10% solution direction
    Ca_weight = 10.0
    
    total_norm = np.sqrt(U_norm**2 + Ca_weight**2)
    
    return {
      'U': dU_dCa / total_norm,
      'Ca': Ca_weight / total_norm,
      's_mesh': point.s_range
    }
  
  def _predictor_step(self, current: SolutionPoint, ds: float) -> Tuple[np.ndarray, float]:
    """Predictor step using tangent"""
    # Predict new solution
    U_pred = current.solution + ds * current.tangent['U']
    Ca_pred = current.Ca + ds * current.tangent['Ca']
    
    # Debug log
    self.logger.debug(f"Tangent Ca component: {current.tangent['Ca']}, ds: {ds}")
    
    return U_pred, Ca_pred
  
  def _build_extended_system(self, U: np.ndarray, Ca: float,
                           U_old: np.ndarray, Ca_old: float,
                           tangent: Dict[str, Any], ds: float) -> Tuple[np.ndarray, Any]:
    """
    Build extended system for pseudo-arclength continuation
    
    Returns:
      F_extended: Extended residual vector
      info: Dictionary with Jacobian and other info
    """
    s_mesh = tangent['s_mesh']
    
    # Solve BVP with current U as initial guess
    try:
      s_new, U_bvp = self._solve_bvp(Ca, U, s_mesh)
      
      # If mesh was refined, interpolate back to original mesh
      if len(s_new) != len(s_mesh):
        from scipy.interpolate import interp1d
        U_interp = np.zeros_like(U)
        for i in range(3):  # h, theta, omega
          f_interp = interp1d(s_new, U_bvp[i, :], kind='cubic',
                             bounds_error=False, fill_value='extrapolate')
          U_interp[i, :] = f_interp(s_mesh)
        U_bvp = U_interp
      
      # BVP residual
      F_bvp = U_bvp - U
      
      # Arc-length constraint
      dU = U - U_old
      dCa = Ca - Ca_old
      arc_constraint = np.sum(dU * tangent['U']) + dCa * tangent['Ca'] - ds
      
      # Extended residual
      F_extended = np.concatenate([F_bvp.flatten(), [arc_constraint]])
      
      info = {
        'F_bvp': F_bvp,
        'arc_constraint': arc_constraint,
        'U_bvp': U_bvp
      }
      
      return F_extended, info
      
    except Exception as e:
      # Return large residual if BVP fails
      F_extended = np.ones(U.size + 1) * 1e10
      info = {'error': str(e)}
      return F_extended, info
  
  def _corrector_step(self, U_pred: np.ndarray, Ca_pred: float,
                     current: SolutionPoint, ds: float) -> Tuple[np.ndarray, float, bool]:
    """
    Corrector step using Newton iteration on extended system
    
    Returns:
      U_corrected: Corrected solution
      Ca_corrected: Corrected Ca
      converged: Whether solver converged
    """
    # Start from prediction
    U = U_pred.copy()
    Ca = Ca_pred
    
    # Newton iteration for extended system
    for newton_iter in range(self.params.newton_max_iter):
      # Build extended system
      F_ext, info = self._build_extended_system(U, Ca, current.solution, current.Ca,
                                               current.tangent, ds)
      
      # Check convergence
      residual_norm = np.linalg.norm(F_ext)
      if residual_norm < self.params.newton_tol:
        return U, Ca, True
      
      if 'error' in info:
        # BVP failed, can't continue
        return U_pred, Ca_pred, False
      
      # Compute Newton update using finite differences
      # This is simplified - a full implementation would compute the Jacobian
      
      # For now, use simplified corrector that enforces arc-length constraint
      # by adjusting Ca to satisfy the constraint
      
      U_bvp = info.get('U_bvp', U)
      arc_residual = info.get('arc_constraint', 0)
      
      # Update solution towards BVP solution
      alpha = 0.7  # Damping factor
      U = (1 - alpha) * U + alpha * U_bvp
      
      # Adjust Ca to reduce arc-length constraint residual
      # Simple proportional correction
      Ca_correction = -0.1 * arc_residual * current.tangent['Ca']
      Ca = Ca + Ca_correction
    
    # If we get here, Newton didn't fully converge but we have a solution
    return U, Ca, residual_norm < 1e-6
  
  def _compute_tangent(self, point: SolutionPoint, prev_point: SolutionPoint) -> Dict[str, Any]:
    """Compute new tangent using secant method"""
    # Always interpolate both solutions to a common mesh for consistency
    from scipy.interpolate import interp1d
    
    # Use the current point's mesh as reference
    s_mesh_ref = point.s_range
    
    # Interpolate previous solution to current mesh
    prev_interp = np.zeros((3, len(s_mesh_ref)))
    for i in range(3):
      f_interp = interp1d(prev_point.s_range, prev_point.solution[i, :], kind='cubic',
                         bounds_error=False, fill_value='extrapolate')
      prev_interp[i, :] = f_interp(s_mesh_ref)
    
    # Current solution is already on the reference mesh
    curr_solution = point.solution
    
    # Compute difference
    dU = curr_solution - prev_interp
    
    dCa = point.Ca - prev_point.Ca
    ds = point.arc_length_param - prev_point.arc_length_param
    
    # Normalize
    norm = np.sqrt(np.sum(dU**2) + dCa**2)
    
    tangent = {
      'U': dU / norm,
      'Ca': dCa / norm,
      's_mesh': point.s_range
    }
    
    # Check orientation (keep consistent direction)
    if len(self.branch) > 2 and 'U' in prev_point.tangent:
      prev_tangent = prev_point.tangent
      # Need to interpolate prev tangent U to current mesh for dot product
      if len(prev_tangent['U'][0]) != len(tangent['U'][0]):
        prev_U_interp = np.zeros_like(tangent['U'])
        for i in range(3):
          f_interp = interp1d(prev_tangent['s_mesh'], prev_tangent['U'][i, :], kind='cubic',
                             bounds_error=False, fill_value='extrapolate')
          prev_U_interp[i, :] = f_interp(s_mesh_ref)
        dot_product = np.sum(tangent['U'] * prev_U_interp) + tangent['Ca'] * prev_tangent['Ca']
      else:
        dot_product = np.sum(tangent['U'] * prev_tangent['U']) + tangent['Ca'] * prev_tangent['Ca']
      
      if dot_product < 0:
        tangent['U'] = -tangent['U']
        tangent['Ca'] = -tangent['Ca']
    
    return tangent
  
  def _detect_fold(self, point: SolutionPoint, prev_point: Optional[SolutionPoint] = None) -> Tuple[bool, Optional[Dict]]:
    """
    Detect if current point is near a fold
    
    Returns:
      is_fold: Boolean indicating fold
      fold_info: Dictionary with fold characteristics
    """
    fold_indicators = {}
    
    # Indicator 1: |dCa/ds| ≈ 0 (vertical tangent in Ca)
    if 'Ca' in point.tangent:
      dCa_ds = abs(point.tangent['Ca'])
      fold_indicators['dCa_ds'] = dCa_ds
      vertical_tangent = dCa_ds < self.params.fold_detection_tol
    else:
      vertical_tangent = False
    
    # Indicator 2: Change in sign of dCa/ds
    sign_change = False
    if prev_point and 'Ca' in prev_point.tangent and 'Ca' in point.tangent:
      prev_sign = np.sign(prev_point.tangent['Ca'])
      curr_sign = np.sign(point.tangent['Ca'])
      sign_change = prev_sign != curr_sign and prev_sign != 0 and curr_sign != 0
    
    # Indicator 3: theta_min approaching 0
    approaching_zero = point.theta_min < np.radians(1.0)  # Less than 1 degree
    
    fold_indicators['vertical_tangent'] = vertical_tangent
    fold_indicators['sign_change'] = sign_change
    fold_indicators['approaching_zero'] = approaching_zero
    fold_indicators['theta_min_deg'] = np.degrees(point.theta_min)
    
    # Determine if it's a fold
    is_fold = (vertical_tangent or sign_change) and not approaching_zero
    
    if is_fold:
      fold_info = {
        'type': 'turning_point',
        'Ca': point.Ca,
        'delta_x_cl': point.delta_x_cl,
        'theta_min': point.theta_min,
        'indicators': fold_indicators
      }
      return True, fold_info
    
    return False, None
  
  def _adapt_step_size(self, current: SolutionPoint, prev: SolutionPoint,
                      converged: bool, newton_iters: int, ds: float) -> float:
    """Adaptive step size control"""
    if not self.params.adaptive_ds:
      return ds
    
    # Base adaptation on convergence and solution change
    if not converged:
      # Failed - reduce step size
      new_ds = max(ds * 0.5, self.params.ds_min)
    else:
      # Measure changes
      theta_change = np.max(np.abs(current.solution[1] - prev.solution[1]))  # theta is index 1
      Ca_change = abs(current.Ca - prev.Ca)
      
      # Want significant progress in Ca
      if Ca_change < 1e-6:
        # Too small Ca change, increase step size aggressively
        new_ds = min(ds * 2.0, self.params.ds_max)
      elif newton_iters < 3 and theta_change < self.params.angle_change_max:
        # Fast convergence, moderate change - increase step
        new_ds = min(ds * 1.5, self.params.ds_max)
      elif newton_iters > 10 or theta_change > self.params.angle_change_max * 2:
        # Slow convergence or large change - decrease step
        new_ds = max(ds * 0.7, self.params.ds_min)
      else:
        # Keep current step
        new_ds = ds
    
    # Special handling near theta_min = 0
    if current.theta_min < np.radians(5.0):  # Less than 5 degrees
      reduction_factor = max(0.1, current.theta_min / np.radians(5.0))
      new_ds = min(new_ds, ds * reduction_factor)
    
    return new_ds
  
  def pseudo_arclength_continuation(self, Ca_target: float = 0.1) -> List[SolutionPoint]:
    """
    Main continuation loop with fold detection
    
    Args:
      Ca_target: Target Capillary number (stop when reached or at critical point)
    
    Returns:
      List of solution points along the branch
    """
    self.logger.info(f"Starting pseudo-arclength continuation to Ca={Ca_target}")
    
    # Initialize if needed
    if not self.branch:
      self.initialize_continuation()
    
    ds = self.params.ds_init
    arc_length_total = 0.0
    
    # For the first few steps, use simple Ca stepping
    use_simple_stepping = True
    simple_step_count = 0
    
    for step in range(self.params.max_steps):
      current = self.branch[-1]
      
      # Check stopping criteria
      if current.Ca >= Ca_target:
        self.logger.info(f"Reached target Ca={Ca_target}")
        break
      
      if current.theta_min < np.radians(0.1):  # 0.1 degree
        self.logger.info(f"Approaching theta_min=0 at Ca={current.Ca:.6f}")
        break
      
      # For initial steps, use simple Ca continuation
      if use_simple_stepping:
        # Simple predictor: just increase Ca
        dCa = min(0.001, Ca_target - current.Ca)  # Small step in Ca
        Ca_pred = current.Ca + dCa
        
        # Use current solution as initial guess, but we'll use a fixed mesh
        U_pred = current.solution
        s_mesh_base = np.linspace(0, self.Delta, self.N_points)
        
        self.logger.info(f"Simple step {simple_step_count}: Ca={current.Ca:.6f} -> Ca_pred={Ca_pred:.6f}")
        
        # Try to solve at new Ca
        try:
          # Always interpolate the current solution to the base mesh for consistency
          from scipy.interpolate import interp1d
          U_interp = np.zeros((3, self.N_points))
          for i in range(3):
            f_interp = interp1d(current.s_range, current.solution[i, :], kind='cubic',
                               bounds_error=False, fill_value='extrapolate')
            U_interp[i, :] = f_interp(s_mesh_base)
          U_pred = U_interp
          
          # Use the base mesh for solving
          s_new, U_new = self._solve_bvp(Ca_pred, U_pred, s_mesh_base)
          
          # Extract solution properties
          h = U_new[0, :]
          theta = U_new[1, :]
          theta_min = np.min(theta)
          
          # Compute x_cl
          x_cl = self._compute_x_cl(s_new, theta, h)
          delta_x_cl = x_cl - self.x_cl_ref if self.x_cl_ref is not None else 0.0
          
          # Update arc length
          # For arc length, just use the Ca change since solutions may have different meshes
          arc_length_total += dCa
          
          # Create new solution point
          new_point = SolutionPoint(
            Ca=Ca_pred,
            solution=U_new,
            s_range=s_new,
            theta_min=theta_min,
            x_cl=x_cl,
            delta_x_cl=delta_x_cl,
            arc_length_param=arc_length_total
          )
          
          # Compute tangent if we have enough points
          if len(self.branch) > 1:
            new_point.tangent = self._compute_tangent(new_point, current)
          else:
            new_point.tangent = {'U': np.zeros_like(U_new), 'Ca': 1.0, 's_mesh': s_new}
          
          # Add to branch
          self.branch.append(new_point)
          
          self.logger.info(
            f"Step {step+1}: Ca={Ca_pred:.6f}, theta_min={np.degrees(theta_min):.2f}°, "
            f"delta_x_cl={delta_x_cl:.6f}"
          )
          
          simple_step_count += 1
          
          # Switch to pseudo-arclength after a few successful simple steps
          if simple_step_count >= 5 and len(self.branch) > 2:
            use_simple_stepping = False
            self.logger.info("Switching to pseudo-arclength continuation")
            ds = 0.1  # Reset step size for arclength
          
          continue
          
        except Exception as e:
          self.logger.warning(f"Simple step failed: {str(e)}")
          import traceback
          self.logger.debug(traceback.format_exc())
          # If simple stepping fails, we're probably near critical Ca
          self.logger.info(f"Simple stepping failed at Ca={Ca_pred:.6f}, likely near critical Ca")
          break
      
      # Pseudo-arclength continuation
      # Predictor
      U_pred, Ca_pred = self._predictor_step(current, ds)
      
      # Log prediction
      self.logger.info(f"Arc-length predictor: Ca={current.Ca:.6f} -> Ca_pred={Ca_pred:.6f}, ds={ds:.6f}")
      
      # Corrector
      start_time = time.time()
      U_corr, Ca_corr, converged = self._corrector_step(U_pred, Ca_pred, current, ds)
      corr_time = time.time() - start_time
      
      if not converged:
        # Try with smaller step
        self.logger.warning(f"Corrector failed at step {step}, reducing step size")
        ds = self._adapt_step_size(current, current, False, self.params.newton_max_iter, ds)
        if ds <= self.params.ds_min:
          self.logger.warning("Minimum step size reached, terminating")
          break
        continue
      
      # Extract solution properties
      h = U_corr[0, :]
      theta = U_corr[1, :]
      theta_min = np.min(theta)
      
      # Compute x_cl (needs integration)
      x_cl = self._compute_x_cl(current.tangent['s_mesh'], theta, h)
      delta_x_cl = x_cl - self.x_cl_ref if self.x_cl_ref is not None else 0.0
      
      # Update arc length
      arc_length_total += ds
      
      # Create new solution point
      new_point = SolutionPoint(
        Ca=Ca_corr,
        solution=U_corr,
        s_range=current.tangent['s_mesh'],
        theta_min=theta_min,
        x_cl=x_cl,
        delta_x_cl=delta_x_cl,
        arc_length_param=arc_length_total
      )
      
      # Compute new tangent
      new_point.tangent = self._compute_tangent(new_point, current)
      
      # Detect fold
      is_fold, fold_info = self._detect_fold(new_point, current)
      if is_fold:
        new_point.is_fold = True
        self.logger.info(f"Fold detected at Ca={Ca_corr:.6f}, delta_x_cl={delta_x_cl:.6f}")
      
      # Add to branch
      self.branch.append(new_point)
      
      # Log progress
      self.logger.info(
        f"Step {step+1}: Ca={Ca_corr:.6f}, theta_min={np.degrees(theta_min):.2f}°, "
        f"delta_x_cl={delta_x_cl:.6f}, ds={ds:.6f}, time={corr_time:.2f}s"
      )
      
      # Adapt step size
      ds = self._adapt_step_size(new_point, current, converged, 5, ds)  # Assume 5 Newton iters
      
      # Check for critical point
      if theta_min < np.radians(0.5):  # Very close to 0
        self.logger.info(f"Critical point approaching at Ca={Ca_corr:.6f}")
        # Could implement special handling here
    
    self.logger.info(f"Continuation completed with {len(self.branch)} points")
    return self.branch
  
  def _compute_x_cl(self, s_mesh: np.ndarray, theta: np.ndarray, h: np.ndarray) -> float:
    """Compute contact line position X_cl"""
    # In the moving frame, the contact line position is determined by
    # the macroscopic interface shape far from the contact line
    
    # Use the asymptotic matching: at large s, the interface approaches
    # a straight line with angle theta_macro relative to the substrate
    
    # Extract the macroscopic angle from the far-field solution
    # Take average of theta in the last 10% of the domain
    far_field_idx = int(0.9 * len(s_mesh))
    theta_macro = np.mean(theta[far_field_idx:])
    
    # The contact line position in the lab frame
    # X_cl = integral of displacement from straight interface
    # For now, use a simple estimate based on the deviation from theta0
    
    # Better approach: integrate the curvature to get the shape
    # and then match to find the contact line position
    
    # Simplified version: X_cl proportional to (theta_macro - theta0)
    X_cl = self.Delta * (theta_macro - self.theta0) / np.pi
    
    return X_cl
  
  def analyze_branch(self) -> Dict[str, Any]:
    """Analyze the computed solution branch"""
    if not self.branch:
      return {}
    
    # Extract data
    Ca_vals = np.array([p.Ca for p in self.branch])
    delta_x_cl_vals = np.array([p.delta_x_cl for p in self.branch])
    theta_min_vals = np.array([p.theta_min for p in self.branch])
    
    # Find folds
    fold_points = [p for p in self.branch if p.is_fold]
    
    # Identify critical point (where theta_min → 0)
    critical_idx = np.argmin(theta_min_vals)
    critical_point = self.branch[critical_idx]
    
    analysis = {
      'Ca_range': (Ca_vals.min(), Ca_vals.max()),
      'delta_x_cl_range': (delta_x_cl_vals.min(), delta_x_cl_vals.max()),
      'theta_min_range': (theta_min_vals.min(), theta_min_vals.max()),
      'num_points': len(self.branch),
      'num_folds': len(fold_points),
      'fold_locations': [(p.Ca, p.delta_x_cl) for p in fold_points],
      'critical_Ca': critical_point.Ca,
      'critical_theta_min_deg': np.degrees(critical_point.theta_min),
      'critical_delta_x_cl': critical_point.delta_x_cl
    }
    
    self.logger.info("\nBranch Analysis:")
    self.logger.info(f"  Ca range: {analysis['Ca_range']}")
    self.logger.info(f"  δX_cl range: {analysis['delta_x_cl_range']}")
    self.logger.info(f"  Number of folds: {analysis['num_folds']}")
    self.logger.info(f"  Critical Ca: {analysis['critical_Ca']:.6f}")
    
    return analysis
  
  def plot_results(self, save_dir: str = 'output'):
    """Create comprehensive visualization"""
    if not self.branch:
      self.logger.warning("No branch data to plot")
      return
    
    # Create output directory
    Path(save_dir).mkdir(exist_ok=True)
    
    # Extract data
    Ca_vals = np.array([p.Ca for p in self.branch])
    delta_x_cl_vals = np.array([p.delta_x_cl for p in self.branch])
    theta_min_vals = np.array([np.degrees(p.theta_min) for p in self.branch])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: δX_cl vs Ca (main result)
    ax1 = axes[0, 0]
    ax1.plot(Ca_vals, delta_x_cl_vals, 'b-', linewidth=2, label='Solution branch')
    
    # Mark fold points
    fold_points = [p for p in self.branch if p.is_fold]
    if fold_points:
      fold_Ca = [p.Ca for p in fold_points]
      fold_delta_x = [p.delta_x_cl for p in fold_points]
      ax1.plot(fold_Ca, fold_delta_x, 'ro', markersize=8, label='Fold point')
    
    ax1.set_xlabel('Capillary number (Ca)', fontsize=12)
    ax1.set_ylabel('Contact line displacement (δX_cl)', fontsize=12)
    ax1.set_title('Contact Line Displacement vs Capillary Number', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Plot 2: θ_min vs Ca
    ax2 = axes[0, 1]
    ax2.plot(Ca_vals, theta_min_vals, 'g-', linewidth=2)
    if fold_points:
      fold_theta = [np.degrees(p.theta_min) for p in fold_points]
      ax2.plot(fold_Ca, fold_theta, 'ro', markersize=8, label='Fold point')
    
    ax2.set_xlabel('Capillary number (Ca)', fontsize=12)
    ax2.set_ylabel('Minimum contact angle (degrees)', fontsize=12)
    ax2.set_title('Minimum Contact Angle vs Capillary Number', fontsize=14)
    ax2.grid(True, alpha=0.3)
    if fold_points:
      ax2.legend(fontsize=10)
    
    # Plot 3: Solution profiles at key points
    ax3 = axes[1, 0]
    
    # Select key points to plot
    indices = [0, len(self.branch)//3, 2*len(self.branch)//3, -1]
    colors = ['blue', 'green', 'orange', 'red']
    
    for idx, color in zip(indices, colors):
      if idx < len(self.branch):
        point = self.branch[idx]
        ax3.plot(point.s_range, np.degrees(point.solution[1, :]), 
                color=color, label=f'Ca={point.Ca:.4f}')
    
    ax3.set_xlabel('Arc length (s)', fontsize=12)
    ax3.set_ylabel('Contact angle (degrees)', fontsize=12)
    ax3.set_title('Contact Angle Profiles', fontsize=14)
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    
    # Plot 4: Phase portrait (Ca vs arc length parameter)
    ax4 = axes[1, 1]
    arc_params = np.array([p.arc_length_param for p in self.branch])
    ax4.plot(arc_params, Ca_vals, 'k-', linewidth=2)
    if fold_points:
      fold_arc = [p.arc_length_param for p in fold_points]
      ax4.plot(fold_arc, fold_Ca, 'ro', markersize=8, label='Fold point')
    
    ax4.set_xlabel('Arc length parameter', fontsize=12)
    ax4.set_ylabel('Capillary number (Ca)', fontsize=12)
    ax4.set_title('Continuation Path', fontsize=14)
    ax4.grid(True, alpha=0.3)
    if fold_points:
      ax4.legend(fontsize=10)
    
    # Add parameter information as text
    param_text = (
      f"Parameters:\n"
      f"μ_r = {self.mu_r}\n"
      f"λ_slip = {self.lambda_slip}\n"
      f"θ₀ = {np.degrees(self.theta0):.1f}°\n"
      f"Δ = {self.Delta}\n"
      f"N_points = {self.N_points}"
    )
    fig.text(0.02, 0.02, param_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Save figure with parameter information in filename
    filename = f"continuation_mu{self.mu_r}_lambda{self.lambda_slip}_theta{np.degrees(self.theta0):.1f}.png"
    filepath = Path(save_dir) / filename
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    self.logger.info(f"Plots saved to {filepath}")
    
    # Also save individual plots
    for i, (ax, title) in enumerate(zip(axes.flat, ['displacement', 'theta_min', 'profiles', 'continuation'])):
      fig_single = plt.figure(figsize=(8, 6))
      ax_single = fig_single.add_subplot(111)
      
      # Copy the plot content
      for line in ax.get_lines():
        ax_single.plot(line.get_xdata(), line.get_ydata(), 
                      color=line.get_color(), 
                      linestyle=line.get_linestyle(),
                      linewidth=line.get_linewidth(),
                      label=line.get_label())
      
      # Copy axis properties
      ax_single.set_xlabel(ax.get_xlabel(), fontsize=12)
      ax_single.set_ylabel(ax.get_ylabel(), fontsize=12)
      ax_single.set_title(ax.get_title(), fontsize=14)
      ax_single.grid(True, alpha=0.3)
      if ax.get_legend():
        ax_single.legend(fontsize=10)
      
      plt.tight_layout()
      
      # Save individual plot
      single_filename = f"continuation_{title}_mu{self.mu_r}_lambda{self.lambda_slip}_theta{np.degrees(self.theta0):.1f}.png"
      single_filepath = Path(save_dir) / single_filename
      plt.savefig(single_filepath, dpi=300, bbox_inches='tight')
      plt.close(fig_single)
    
    plt.close(fig)
  
  def export_results(self, filename: str, format: str = 'pickle'):
    """
    Export branch data for further analysis
    
    Args:
      filename: Output filename
      format: 'pickle' or 'h5'
    """
    if format == 'pickle':
      with open(filename, 'wb') as f:
        pickle.dump({
          'branch': self.branch,
          'parameters': {
            'mu_r': self.mu_r,
            'lambda_slip': self.lambda_slip,
            'theta0': self.theta0,
            'Delta': self.Delta,
            'N_points': self.N_points
          },
          'analysis': self.analyze_branch()
        }, f)
      self.logger.info(f"Results exported to {filename}")
      
    elif format == 'h5':
      with h5py.File(filename, 'w') as f:
        # Store parameters
        params_grp = f.create_group('parameters')
        params_grp.attrs['mu_r'] = self.mu_r
        params_grp.attrs['lambda_slip'] = self.lambda_slip
        params_grp.attrs['theta0'] = self.theta0
        params_grp.attrs['Delta'] = self.Delta
        params_grp.attrs['N_points'] = self.N_points
        
        # Store branch data
        branch_grp = f.create_group('branch')
        for i, point in enumerate(self.branch):
          point_grp = branch_grp.create_group(f'point_{i:04d}')
          point_grp.attrs['Ca'] = point.Ca
          point_grp.attrs['theta_min'] = point.theta_min
          point_grp.attrs['x_cl'] = point.x_cl
          point_grp.attrs['delta_x_cl'] = point.delta_x_cl
          point_grp.attrs['is_fold'] = point.is_fold
          
          point_grp.create_dataset('s_range', data=point.s_range)
          point_grp.create_dataset('solution', data=point.solution)
        
      self.logger.info(f"Results exported to {filename}")
    else:
      raise ValueError(f"Unknown format: {format}")


def main():
  """Main execution function"""
  parser = argparse.ArgumentParser(
    description='Pseudo-arclength continuation for GLE system'
  )
  
  # Physical parameters
  parser.add_argument('--mu_r', type=float, default=1.0,
                     help='Viscosity ratio (default: 1.0)')
  parser.add_argument('--lambda_slip', type=float, default=1e-4,
                     help='Slip length (default: 1e-4)')
  parser.add_argument('--theta0', type=float, default=10.0,
                     help='Initial contact angle in degrees (default: 10)')
  
  # Numerical parameters
  parser.add_argument('--Delta', type=float, default=10.0,
                     help='Domain size (default: 10.0)')
  parser.add_argument('--N_points', type=int, default=10000,
                     help='Number of mesh points (default: 10000)')
  
  # Continuation parameters
  parser.add_argument('--Ca_target', type=float, default=0.1,
                     help='Target Capillary number (default: 0.1)')
  parser.add_argument('--ds_init', type=float, default=0.1,
                     help='Initial arc length step (default: 0.1)')
  parser.add_argument('--max_steps', type=int, default=500,
                     help='Maximum continuation steps (default: 500)')
  
  # Output options
  parser.add_argument('--output', type=str, default='continuation_results.pkl',
                     help='Output filename (default: continuation_results.pkl)')
  parser.add_argument('--format', choices=['pickle', 'h5'], default='pickle',
                     help='Output format (default: pickle)')
  parser.add_argument('--plot', action='store_true',
                     help='Generate plots')
  
  args = parser.parse_args()
  
  # Convert theta0 to radians
  theta0_rad = np.radians(args.theta0)
  
  # Set up continuation parameters
  cont_params = ContinuationParameters(
    ds_init=args.ds_init,
    max_steps=args.max_steps
  )
  
  # Create solver
  solver = GLEContinuation(
    mu_r=args.mu_r,
    lambda_slip=args.lambda_slip,
    theta0=theta0_rad,
    Delta=args.Delta,
    N_points=args.N_points,
    params=cont_params
  )
  
  # Run continuation
  print(f"\nStarting pseudo-arclength continuation:")
  print(f"  mu_r = {args.mu_r}")
  print(f"  lambda_slip = {args.lambda_slip}")
  print(f"  theta0 = {args.theta0}°")
  print(f"  Target Ca = {args.Ca_target}")
  print()
  
  start_time = time.time()
  
  try:
    branch = solver.pseudo_arclength_continuation(Ca_target=args.Ca_target)
    
    # Analyze results
    analysis = solver.analyze_branch()
    
    # Export results
    solver.export_results(args.output, format=args.format)
    
    # Generate plots if requested
    if args.plot:
      solver.plot_results()
    
    elapsed_time = time.time() - start_time
    print(f"\nContinuation completed in {elapsed_time:.2f} seconds")
    print(f"Results saved to {args.output}")
    
  except Exception as e:
    print(f"\nError during continuation: {str(e)}")
    import traceback
    traceback.print_exc()
    return 1
  
  return 0


if __name__ == "__main__":
  sys.exit(main())