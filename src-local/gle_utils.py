"""
Core utilities for solving the Generalized Lubrication Equations (GLE).

This module contains the fundamental ODE system and solver functions
without any dependencies on higher-level modules to avoid circular imports.
"""

import numpy as np
from scipy.integrate import solve_bvp
from functools import partial
from typing import Tuple, Optional, Any
from dataclasses import dataclass

from solution_types import SolutionResult
from find_x0_utils import find_x0_and_theta_min


# Mathematical functions for GLE
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
    denominator = 3 * (mu_r * f1(theta) * f2(np.pi - theta) + f1(np.pi - theta) * f2(theta))
    
    # Add small epsilon to prevent division by zero
    epsilon = 1e-10
    denominator = np.where(np.abs(denominator) < epsilon, 
                          np.sign(denominator) * epsilon + (denominator == 0) * epsilon, 
                          denominator)
    return numerator / denominator


def GLE(s: float, y: np.ndarray, Ca: float, mu_r: float, lambda_slip: float) -> list:
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
    h_min = lambda_slip * 1e-6  # Minimum h as a fraction of slip length
    h_safe = np.maximum(h, h_min)
    
    # Ensure theta stays in physical range (0, pi)
    theta_safe = np.clip(theta, 1e-10, np.pi - 1e-10)
    
    dh_ds = np.sin(theta_safe)  # dh/ds = sin(theta)
    dt_ds = omega  # omega = dtheta/ds
    
    # Use safe values to prevent division by zero
    dw_ds = 3 * Ca * f(theta_safe, mu_r) / (h_safe * (h_safe + 3 * lambda_slip)) - np.cos(theta_safe)
    
    return [dh_ds, dt_ds, dw_ds]


def boundary_conditions(ya: np.ndarray, yb: np.ndarray, w_bc: float, 
                       theta0: float, lambda_slip: float) -> list:
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
