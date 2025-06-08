"""
Utility functions for finding x0 - the position where theta reaches its minimum value
"""

import numpy as np
from typing import Tuple, Optional, List

def find_x0_from_solution(s_vals: np.ndarray, theta_vals: np.ndarray, 
                         tolerance: float = 1e-10) -> Tuple[Optional[float], Optional[int], float]:
    """
    Find x0 - the x-position where theta reaches its minimum value.
    
    This is a key parameter for the contact line problem. When theta_min approaches 0,
    we are at the critical capillary number (Ca_cr).
    
    Args:
        s_vals: Arc length values
        theta_vals: Contact angle values in radians
        tolerance: Not used in this implementation (kept for backward compatibility)
    
    Returns:
        (x0, idx, theta_min) where:
        - x0: x-position where theta reaches its minimum (always exists)
        - idx: Index in arrays where this occurs
        - theta_min: The minimum value of theta
    """
    # Find the index where theta is minimum
    idx_min = np.argmin(theta_vals)
    theta_min = theta_vals[idx_min]
    
    # Calculate x values by integrating cos(theta)
    x_vals = np.zeros_like(s_vals)
    if len(s_vals) > 1:
        # Use trapezoidal integration for better accuracy
        for i in range(1, len(s_vals)):
            ds = s_vals[i] - s_vals[i-1]
            # Average cos(theta) over the interval
            cos_avg = 0.5 * (np.cos(theta_vals[i-1]) + np.cos(theta_vals[i]))
            x_vals[i] = x_vals[i-1] + ds * cos_avg
    
    # Get x0 at the minimum theta position
    x0 = x_vals[idx_min]
    
    return x0, idx_min, theta_min


def find_x0_and_theta_min(s_vals: np.ndarray, theta_vals: np.ndarray, 
                         tolerance: float = 1e-3) -> Tuple[Optional[float], float, Optional[int]]:
    """
    Find both x0 (position where theta reaches its minimum) and theta_min.
    
    With the new definition, x0 is always at the position where theta = theta_min.
    When theta_min approaches 0, we are at the critical capillary number (Ca_cr).
    
    Args:
        s_vals: Arc length values
        theta_vals: Contact angle values in radians
        tolerance: Not used in this implementation (kept for backward compatibility)
    
    Returns:
        (x0, theta_min, x0_idx) where:
        - x0: x-position where theta reaches its minimum (always exists)
        - theta_min: Minimum value of theta
        - x0_idx: Index where theta minimum occurs
    """
    # Find x0 and theta_min using the unified function
    x0, x0_idx, theta_min = find_x0_from_solution(s_vals, theta_vals, tolerance)
    
    return x0, theta_min, x0_idx


def find_X_cl(s_vals: np.ndarray, theta_vals: np.ndarray) -> float:
    """
    Find X_cl - the contact line position (maximum x value).
    
    X_cl is the x-coordinate at the end of the domain, representing
    the contact line position in the lubrication approximation.
    
    Args:
        s_vals: Arc length values
        theta_vals: Contact angle values in radians
    
    Returns:
        X_cl: Contact line position (x at s=Delta)
    """
    # Calculate x values by integrating cos(theta)
    x_vals = np.zeros_like(s_vals)
    if len(s_vals) > 1:
        # Use trapezoidal integration for better accuracy
        for i in range(1, len(s_vals)):
            ds = s_vals[i] - s_vals[i-1]
            # Average cos(theta) over the interval
            cos_avg = 0.5 * (np.cos(theta_vals[i-1]) + np.cos(theta_vals[i]))
            x_vals[i] = x_vals[i-1] + ds * cos_avg
    
    # X_cl is the maximum x value (at the end of the domain)
    X_cl = x_vals[-1]
    
    return X_cl