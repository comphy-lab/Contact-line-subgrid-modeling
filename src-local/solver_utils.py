"""
Core solver functions for the GLE solver.
"""

import numpy as np
from scipy.integrate import solve_bvp
from typing import Tuple, Optional, Any, List
from solution_types import SolutionResult
from find_x0_utils import find_x0_and_theta_min
from gle_utils import get_adaptive_max_nodes, GLE, boundary_conditions
from functools import partial

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