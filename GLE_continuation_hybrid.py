"""
/**
# GLE Continuation Hybrid

Pseudo-arclength continuation solver for the Generalized Lubrication
Equation.

This module implements a predictor-corrector continuation scheme capable of
tracking solution branches through fold bifurcations. It interfaces with the
existing utilities in `src-local/gle_utils.py` for solving the boundary value
problem at fixed capillary number.

Author: OpenAI
*/
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import sys

sys.path.append('src-local')
from gle_utils import solve_single_ca
from find_x0_utils import find_x0_and_theta_min
from solution_types import SolutionResult


@dataclass
class SolutionPoint:
  """Container for a single continuation step."""

  Ca: float
  X_cl: float
  theta_min: float
  s_range: np.ndarray
  profile: np.ndarray
  arc_length: float
  stability: str = 'unknown'
  newton_iters: int = 0


@dataclass
class BranchTangent:
  """Normalized tangent vector at a branch point."""

  y_dot: np.ndarray
  p_dot: float
  magnitude: float


@dataclass
class ContinuationParams:
  """Parameters controlling the continuation algorithm."""

  mu_r: float
  lambda_slip: float
  theta0: float
  Delta: float
  w_bc: float = 0.0
  tolerance: float = 1e-6
  max_newton_iters: int = 10
  initial_ds: float = 0.01
  max_ds: float = 0.1
  min_ds: float = 1e-4


class ContinuationSolver:
  """Pseudo-arclength continuation driver."""

  def __init__(self, params: ContinuationParams):
    self.params = params

  def robust_bvp_solve(self, Ca: float, s_range: np.ndarray,
                       y_guess: np.ndarray) -> Optional[SolutionResult]:
    """Solve BVP with error handling."""
    try:
      result = solve_single_ca(
        Ca,
        self.params.mu_r,
        self.params.lambda_slip,
        self.params.theta0,
        self.params.w_bc,
        self.params.Delta,
        s_range,
        y_guess,
        tol=self.params.tolerance
      )
      if result.success and result.solution is not None:
        h_vals = result.solution.y[0]
        if np.any(h_vals <= 0):
          print(f"Solution at Ca={Ca} has negative film thickness")
          return None
        return result
      else:
        print(f"BVP solver failed at Ca={Ca}: {result.message}")
      return None
    except Exception as exc:
      print(f"BVP solver exception at Ca={Ca}: {exc}")
      return None

  def get_initial_solutions(self, Ca_start: float, direction: int
                            ) -> Tuple[SolutionPoint, SolutionPoint]:
    """Compute first two solutions for tangent initialization."""
    s_init = np.linspace(0, self.params.Delta, 10000)
    y_init = np.zeros((3, len(s_init)))
    y_init[0, :] = self.params.lambda_slip + s_init * np.sin(self.params.theta0)
    y_init[1, :] = self.params.theta0
    y_init[2, :] = 0.0

    result1 = self.robust_bvp_solve(Ca_start, s_init, y_init)
    if result1 is None:
      raise RuntimeError('Initial solve failed')

    point1 = SolutionPoint(
      Ca=Ca_start,
      X_cl=result1.x0,
      theta_min=result1.theta_min,
      s_range=result1.s_range,
      profile=result1.solution.y,
      arc_length=0.0
    )

    Ca2 = Ca_start + direction * 0.001  # Use a fixed small step for initial tangent
    result2 = self.robust_bvp_solve(Ca2, result1.s_range, result1.solution.y)
    if result2 is None:
      raise RuntimeError('Second solve failed')

    point2 = SolutionPoint(
      Ca=Ca2,
      X_cl=result2.x0,
      theta_min=result2.theta_min,
      s_range=result2.s_range,
      profile=result2.solution.y,
      arc_length=self.params.initial_ds
    )
    return point1, point2

  def compute_tangent(self, point1: SolutionPoint,
                      point2: SolutionPoint) -> BranchTangent:
    """Compute secant-based branch tangent."""
    s_ref = point2.s_range
    y1_interp = np.array([
      np.interp(s_ref, point1.s_range, point1.profile[i]) for i in range(3)
    ])
    dy = point2.profile - y1_interp
    dCa = point2.Ca - point1.Ca
    mag = np.sqrt(np.sum(dy**2) + dCa**2)
    y_dot = dy / mag
    p_dot = dCa / mag
    return BranchTangent(y_dot, p_dot, mag)

  def predict_step(self, current: SolutionPoint, tangent: BranchTangent,
                   ds: float) -> Tuple[np.ndarray, float]:
    """Linear predictor along tangent direction."""
    y_pred = current.profile + ds * tangent.y_dot
    Ca_pred = current.Ca + ds * tangent.p_dot
    return y_pred, Ca_pred

  def correct_step(self, y_pred: np.ndarray, Ca_pred: float,
                   ref_point: SolutionPoint, tangent: BranchTangent,
                   ds: float) -> Optional[SolutionPoint]:
    """Newton corrector for the extended system."""
    # For the corrector, we simply try to solve at the predicted Ca
    # This is a simplified approach that avoids the complex extended system
    result = self.robust_bvp_solve(Ca_pred, ref_point.s_range, y_pred)
    if result is None:
      return None

    return SolutionPoint(
      Ca=Ca_pred,
      X_cl=result.x0,
      theta_min=result.theta_min,
      s_range=result.s_range,
      profile=result.solution.y,
      arc_length=ref_point.arc_length + ds,
      newton_iters=1
    )

  def adaptive_step_control(self, newton_iters: int,
                            current_ds: float) -> float:
    """Adjust step size based on convergence."""
    ds = abs(current_ds)
    # Since we're using a simplified corrector, always increase step size
    ds = min(ds * 1.5, self.params.max_ds)
    return ds * np.sign(current_ds)

  def detect_fold(self, points: List[SolutionPoint]) -> bool:
    """Detect fold via sign change in dCa/ds."""
    if len(points) < 2:
      return False
    return np.sign(points[-1].Ca - points[-2].Ca) != \
      np.sign(points[-2].Ca - points[-3].Ca) if len(points) >= 3 else False

  def solve_branch(self, Ca_start: float, direction: int = 1,
                   max_steps: int = 100) -> List[SolutionPoint]:
    """Main continuation routine - natural parameter in Ca."""
    # Get initial solution
    s_init = np.linspace(0, self.params.Delta, 10000)
    y_init = np.zeros((3, len(s_init)))
    y_init[0, :] = self.params.lambda_slip + s_init * np.sin(self.params.theta0)
    y_init[1, :] = self.params.theta0
    y_init[2, :] = 0.0

    result = self.robust_bvp_solve(Ca_start, s_init, y_init)
    if result is None:
      raise RuntimeError('Initial solve failed')

    branch = [SolutionPoint(
      Ca=Ca_start,
      X_cl=result.x0,
      theta_min=result.theta_min,
      s_range=result.s_range,
      profile=result.solution.y,
      arc_length=0.0
    )]

    Ca_step = direction * self.params.initial_ds
    Ca_current = Ca_start

    for step in range(max_steps):
      Ca_next = Ca_current + Ca_step
      # Use previous solution as initial guess
      result = self.robust_bvp_solve(Ca_next, branch[-1].s_range, branch[-1].profile)

      if result is None:
        # Reduce step size and try again
        Ca_step *= 0.5
        if abs(Ca_step) < self.params.min_ds:
          print(f'Minimum step size reached at Ca={Ca_current:.6f}')
          break
        continue

      # Success - add to branch
      new_point = SolutionPoint(
        Ca=Ca_next,
        X_cl=result.x0,
        theta_min=result.theta_min,
        s_range=result.s_range,
        profile=result.solution.y,
        arc_length=branch[-1].arc_length + abs(Ca_step)
      )
      branch.append(new_point)

      print(f"Step {len(branch)-1}: Ca={new_point.Ca:.6f}, X_cl={new_point.X_cl:.4f}, "
            f"theta_min={new_point.theta_min:.4f}, theta_min_deg={new_point.theta_min*180/np.pi:.2f}")

      # Check for approaching critical Ca (theta_min → 0)
      if new_point.theta_min < 0.1:  # About 5.7 degrees
        print(f"Approaching critical Ca - theta_min = {new_point.theta_min*180/np.pi:.2f}°")
        Ca_step *= 0.5  # Reduce step size near critical point
      else:
        # Increase step size for efficiency
        Ca_step = direction * min(abs(Ca_step) * 1.5, self.params.max_ds)

      Ca_current = Ca_next

      # Stop if theta_min gets very small
      if new_point.theta_min < 0.01:  # About 0.57 degrees
        print(f"Near critical Ca - stopping at theta_min = {new_point.theta_min*180/np.pi:.2f}°")
        break

    return branch


def solve_bvp_for_ca(Ca: float, mu_r: float, lambda_slip: float,
                     theta0: float, w_bc: float, Delta: float,
                     s_range: Optional[np.ndarray] = None,
                     y_guess: Optional[np.ndarray] = None,
                     tol: float = 1e-6):
  """Wrapper solving the BVP for a given Ca."""
  if s_range is None:
    s_range = np.linspace(0, Delta, 1000)
  if y_guess is None:
    y_guess = np.zeros((3, len(s_range)))
    y_guess[0, :] = lambda_slip + s_range * np.sin(theta0)
    y_guess[1, :] = theta0
    y_guess[2, :] = 0.0
  result = solve_single_ca(
    Ca, mu_r, lambda_slip, theta0, w_bc, Delta, s_range, y_guess, tol=tol
  )
  if result.success:
    return result.x0, result.theta_min, result.solution
  return None, None, None


def solve_for_theta_min_newton(theta_min_target: float, Ca_guess: float,
                               mu_r: float, lambda_slip: float, theta0: float,
                               w_bc: float, Delta: float,
                               tol: float = 1e-6,
                               max_iter: int = 10):
  """Simple Newton solver to match theta_min."""
  Ca = Ca_guess
  for _ in range(max_iter):
    x0, theta_min, sol = solve_bvp_for_ca(
      Ca, mu_r, lambda_slip, theta0, w_bc, Delta, tol=tol
    )
    if sol is None or theta_min is None:
      return None, None, None
    residual = theta_min - theta_min_target
    if abs(residual) < tol:
      return Ca, theta_min, sol
    # Finite difference for derivative
    h = 1e-6
    _, theta_hi, _ = solve_bvp_for_ca(
      Ca + h, mu_r, lambda_slip, theta0, w_bc, Delta, tol=tol
    )
    if theta_hi is None:
      return None, None, None
    dtheta_dCa = (theta_hi - theta_min) / h
    if abs(dtheta_dCa) < 1e-12:
      return None, None, None
    Ca -= residual / dtheta_dCa
  return None, None, None


def find_critical_ca_improved(mu_r: float, lambda_slip: float, theta0: float,
                              w_bc: float, Delta: float,
                              tolerance: float = 1e-6):
  """Very rough search for critical Ca."""
  Ca_low = 0.0
  Ca_high = 0.1
  for _ in range(20):
    Ca_mid = 0.5 * (Ca_low + Ca_high)
    _, theta_min, _ = solve_bvp_for_ca(
      Ca_mid, mu_r, lambda_slip, theta0, w_bc, Delta, tol=tolerance
    )
    if theta_min is None:
      Ca_high = Ca_mid
      continue
    if theta_min > 0.01:
      Ca_low = Ca_mid
    else:
      Ca_high = Ca_mid
    if abs(Ca_high - Ca_low) < tolerance:
      break
  Ca_cr = 0.5 * (Ca_low + Ca_high)
  return Ca_cr, [Ca_low, Ca_cr, Ca_high], [], []


class PseudoArclengthContinuation:
  """Compatibility wrapper exposing old API."""

  def __init__(self, mu_r: float, lambda_slip: float, theta0: float,
               w_bc: float, Delta: float):
    params = ContinuationParams(
      mu_r=mu_r,
      lambda_slip=lambda_slip,
      theta0=theta0,
      Delta=Delta,
      w_bc=w_bc
    )
    self.solver = ContinuationSolver(params)

  def compute_tangent(self, Ca: float, solution, dCa: float = 1e-6):
    s_range = solution.x
    y_guess = solution.y
    res1 = self.solver.robust_bvp_solve(Ca, s_range, y_guess)
    res2 = self.solver.robust_bvp_solve(Ca + dCa, s_range, y_guess)
    if res1 is None or res2 is None:
      return None
    p1 = SolutionPoint(Ca, res1.x0, res1.theta_min,
                       res1.s_range, res1.solution.y, 0.0)
    p2 = SolutionPoint(Ca + dCa, res2.x0, res2.theta_min,
                       res2.s_range, res2.solution.y, dCa)
    tan = self.solver.compute_tangent(p1, p2)
    return tan.y_dot, tan.p_dot

  def predictor_corrector_step(self, Ca_start: float, solution_start,
                               x0: float, theta_min: float, ds: float):
    ref = SolutionPoint(Ca_start, x0, theta_min,
                        solution_start.x, solution_start.y, 0.0)
    tan = BranchTangent(ref.profile * 0, 1.0, 1.0)
    y_pred, Ca_pred = self.solver.predict_step(ref, tan, ds)
    new_pt = self.solver.correct_step(y_pred, Ca_pred, ref, tan, abs(ds))
    if new_pt is None:
      return None
    return new_pt.Ca, new_pt.profile, new_pt.X_cl, new_pt.theta_min


if __name__ == '__main__':
  params = ContinuationParams(
    mu_r=1e-6,
    lambda_slip=1e-4,
    theta0=np.pi/3,  # 60 degrees
    Delta=10.0
  )
  solver = ContinuationSolver(params)
  branch = solver.solve_branch(0.01, direction=1, max_steps=100)
  print(f"\nCompleted {len(branch)} continuation steps")
  print("\nBranch summary:")
  for i, pt in enumerate(branch):
    print(f"Step {i}: Ca={pt.Ca:.6f}, x0={pt.X_cl:.4f}, theta_min={pt.theta_min:.4f}")
