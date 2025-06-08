"""
Generalized Lubrication Equations (GLE) Solver
==============================================

This module solves the coupled ODEs for contact line dynamics using the
Generalized Lubrication Equations. It provides a straightforward solver
that attempts to find a solution for a given Capillary number (Ca).

If the solver fails (typically because Ca exceeds the critical value),
it suggests using GLE_critical_ca.py or GLE_critical_ca_advanced.py
to find the critical Ca.

Author: Aman and Vatsal
Created: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import os
import sys
from functools import partial
import argparse
from typing import Tuple, Optional
import time

# Import utilities from src-local
sys.path.append('src-local')
from find_x0_utils import find_x0_and_theta_min
from gle_utils import (
    GLE, boundary_conditions,
    f1, f2, f3, f
)
from solver_utils import solve_single_ca
from GLE_criticalCa import find_critical_ca_lower_branch

# Default parameters
DEFAULT_DELTA = 1e1  # Maximum s-value for the solver
DEFAULT_CA = 0.01  # Capillary number (conservative default)
DEFAULT_LAMBDA_SLIP = 1e-4  # Slip length
DEFAULT_MU_R = 1e-6  # μ_g/μ_l
DEFAULT_THETA0 = np.pi/2  # theta at s = 0
DEFAULT_W = 0  # curvature boundary condition at s = Δ
DEFAULT_NGRID = 10000  # Number of grid points
DEFAULT_TOLERANCE = 1e-6  # Solver tolerance


def solve_gle(Ca: float, mu_r: float, lambda_slip: float, theta0: float,
              w_bc: float, Delta: float, ngrid: int = DEFAULT_NGRID,
              tolerance: float = DEFAULT_TOLERANCE) -> Tuple[Optional[object], float]:
    """
    Solve the GLE for given parameters.
    
    Args:
        Ca: Capillary number
        mu_r: Viscosity ratio (μ_g/μ_l)
        lambda_slip: Slip length
        theta0: Initial contact angle
        w_bc: Curvature boundary condition at s=Delta
        Delta: Maximum s-value
        ngrid: Number of grid points
        tolerance: Solver tolerance
        
    Returns:
        (solution, solve_time): BVP solution object and time taken
    """
    print(f"\nSolving GLE with Ca = {Ca:.6f}, mu_r = {mu_r:.2e}, lambda_slip = {lambda_slip:.2e}")
    
    # Setup grid and initial guess
    s_range = np.linspace(0, Delta, ngrid)
    y_guess = np.zeros((3, ngrid))
    # Physical initial guess for h
    y_guess[0, :] = lambda_slip + s_range * np.sin(theta0)
    y_guess[1, :] = theta0
    y_guess[2, :] = 0
    
    # Create partial functions with parameters
    GLE_with_params = partial(GLE, Ca=Ca, mu_r=mu_r, lambda_slip=lambda_slip)
    bc_with_params = partial(boundary_conditions, w_bc=w_bc, theta0=theta0, lambda_slip=lambda_slip)
    
    # Solve the BVP
    start_time = time.time()
    try:
        solution = solve_bvp(GLE_with_params, bc_with_params, s_range, y_guess,
                           max_nodes=100000, tol=tolerance, verbose=2)
        solve_time = time.time() - start_time
        
        if solution.success:
            print(f"Solution converged in {solve_time:.2f} seconds")
            print(f"Number of mesh points: {len(solution.x)}")
            
            # Check solution validity
            theta_vals = solution.y[1, :]
            if np.any(theta_vals < 0) or np.any(theta_vals > np.pi):
                print("WARNING: Solution has unphysical theta values")
                return None, solve_time
                
            return solution, solve_time
        else:
            print(f"Solver failed after {solve_time:.2f} seconds")
            print(f"Failure message: {solution.message}")
            return None, solve_time
            
    except Exception as e:
        solve_time = time.time() - start_time
        print(f"Solver error after {solve_time:.2f} seconds: {str(e)}")
        return None, solve_time


def create_solution_plots(solution, Ca: float, mu_r: float, lambda_slip: float,
                         theta0: float, Delta: float, output_dir: str) -> None:
    """Create visualization plots for the solution."""
    
    # Extract solution data
    s_values = solution.x
    h_values, theta_values, w_values = solution.y
    theta_values_deg = theta_values * 180 / np.pi
    
    # Calculate x(s) by integrating cos(theta)
    x_values = np.zeros_like(s_values)
    x_values[1:] = np.cumsum(np.diff(s_values) * np.cos(theta_values[:-1]))
    
    # Find x0 and theta_min
    x0, theta_min, x0_idx = find_x0_and_theta_min(s_values, theta_values)
    
    # Create figure
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # Title
    fig.suptitle(f'GLE Solution at Ca = {Ca:.6f}', fontsize=16, fontweight='bold')
    
    solver_color = '#1f77b4'
    
    # Plot 1: h(s) vs s
    ax1.plot(s_values, h_values, '-', color=solver_color, linewidth=2.5)
    ax1.set_xlabel('s', fontsize=14)
    ax1.set_ylabel('h(s)', fontsize=14)
    ax1.set_title('Film Thickness Profile', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, Delta)
    ax1.set_ylim(lambda_slip, 1.01*max(h_values))
    
    # Parameter box
    textstr = f'Ca = {Ca:.6f}\n$\\lambda_\\text{{slip}}$ = {lambda_slip:.0e}\n$\\mu_r$ = {mu_r:.0e}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.95, textstr, transform=ax1.transAxes, fontsize=16,
             verticalalignment='top', bbox=props)
    
    # Plot 2: theta(s) vs s
    ax2.plot(s_values, theta_values_deg, '-', color=solver_color, linewidth=2.5)
    ax2.set_xlabel('s', fontsize=14)
    ax2.set_ylabel('$\\theta(s)$ [degrees]', fontsize=14)
    ax2.set_title('Contact Angle Profile', fontsize=16, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, Delta)
    ax2.set_ylim(0.99*min(theta_values_deg), 1.01*max(theta_values_deg))
    
    # Add info to legend
    ax2.plot([], [], ' ', label=f'θ(0) = {theta0*180/np.pi:.0f}°')
    ax2.plot([], [], ' ', label=f'θ_min = {theta_min*180/np.pi:.2f}°')
    ax2.legend(loc='best', fontsize=12, fancybox=True, framealpha=0.8, 
               facecolor='lightblue', edgecolor='none')
    
    # Mark minimum point
    if x0_idx is not None:
        ax2.scatter(s_values[x0_idx], theta_values_deg[x0_idx], 
                   color='red', s=100, marker='o', zorder=5, 
                   edgecolors='darkred', linewidth=2)
    
    # Plot 3: theta(s) vs h(s)
    ax3.plot(h_values, theta_values_deg, '-', color=solver_color, linewidth=2.5)
    ax3.set_xlabel('h(s)', fontsize=14)
    ax3.set_ylabel('$\\theta(s)$ [degrees]', fontsize=14)
    ax3.set_title('Contact Angle vs Film Thickness', fontsize=16, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(lambda_slip, 1.01*max(h_values))
    ax3.set_ylim(0.99*min(theta_values_deg), 1.01*max(theta_values_deg))
    
    # Plot 4: x(s) vs h(s)
    ax4.plot(h_values, x_values, '-', color=solver_color, linewidth=2.5)
    ax4.set_xlabel('h(s)', fontsize=14)
    ax4.set_ylabel('x(s)', fontsize=14)
    ax4.set_title('Horizontal Position vs Film Thickness', fontsize=16, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(lambda_slip, 1.01*max(h_values))
    ax4.set_ylim(0, 1.01*max(x_values))
    
    # Mark x0 position
    if x0 is not None and x0_idx is not None:
        h_at_theta_min = h_values[x0_idx]
        x_at_theta_min = x_values[x0_idx]
        ax4.scatter(h_at_theta_min, x_at_theta_min, color='red', s=100, 
                    marker='o', zorder=5, edgecolors='darkred', linewidth=2,
                    label=f'$(h, x)$ at $\\theta_{{\\min}}$')
        ax4.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(output_dir, 'GLE_profiles.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")
    
    # Save data to CSV
    csv_data = np.column_stack((s_values, h_values, theta_values, w_values, x_values))
    csv_path = os.path.join(output_dir, 'GLE_profiles.csv')
    np.savetxt(csv_path, csv_data, delimiter=',',
               header='s,h,theta,w,x', comments='')
    print(f"Data saved to: {csv_path}")


def run_solver_and_plot(**kwargs):
  """Run solver and generate plots (compatibility wrapper)."""
  Ca = kwargs.get('Ca', DEFAULT_CA)
  mu_r = kwargs.get('mu_r', DEFAULT_MU_R)
  lambda_slip = kwargs.get('lambda_slip', DEFAULT_LAMBDA_SLIP)
  theta0 = kwargs.get('theta0', DEFAULT_THETA0)
  w = kwargs.get('w', DEFAULT_W)
  Delta = kwargs.get('Delta', DEFAULT_DELTA)
  ngrid = kwargs.get('ngrid', DEFAULT_NGRID)
  output_dir = kwargs.get('output_dir', 'output')

  solution, _ = solve_gle(Ca, mu_r, lambda_slip, theta0, w, Delta, ngrid)
  if solution is not None:
    create_solution_plots(solution, Ca, mu_r, lambda_slip,
                          theta0, Delta, output_dir)
  return solution, (solution.x if solution else []), (
    solution.y[0] if solution else []), (
    solution.y[1] if solution else []), (
    solution.y[2] if solution else [])


def main():
    """Main function to run the GLE solver."""
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Solve Generalized Lubrication Equations (GLE) for contact line dynamics'
    )
    
    # Add arguments
    parser.add_argument('--delta', type=float, default=DEFAULT_DELTA,
                        help=f'Maximum s-value for the solver (default: {DEFAULT_DELTA})')
    parser.add_argument('--ca', type=float, default=DEFAULT_CA,
                        help=f'Capillary number (default: {DEFAULT_CA})')
    parser.add_argument('--lambda_slip', type=float, default=DEFAULT_LAMBDA_SLIP,
                        help=f'Slip length (default: {DEFAULT_LAMBDA_SLIP})')
    parser.add_argument('--mu_r', type=float, default=DEFAULT_MU_R,
                        help=f'Viscosity ratio μ_g/μ_l (default: {DEFAULT_MU_R})')
    parser.add_argument('--theta0', type=float, default=DEFAULT_THETA0*180/np.pi,
                        help=f'Initial contact angle in degrees (default: {DEFAULT_THETA0*180/np.pi:.0f})')
    parser.add_argument('--w', type=float, default=DEFAULT_W,
                        help=f'Curvature boundary condition at s=Delta (default: {DEFAULT_W})')
    parser.add_argument('--ngrid', type=int, default=DEFAULT_NGRID,
                        help=f'Number of grid points (default: {DEFAULT_NGRID})')
    parser.add_argument('--tolerance', type=float, default=DEFAULT_TOLERANCE,
                        help=f'Solver tolerance (default: {DEFAULT_TOLERANCE})')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory for plots and data (default: output)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert theta0 to radians
    theta0_rad = args.theta0 * np.pi / 180
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print parameters
    print("GLE Solver")
    print("="*60)
    print(f"Parameters:")
    print(f"  Delta: {args.delta}")
    print(f"  Ca: {args.ca}")
    print(f"  lambda_slip: {args.lambda_slip}")
    print(f"  mu_r: {args.mu_r}")
    print(f"  theta0: {args.theta0}° ({theta0_rad:.4f} rad)")
    print(f"  w: {args.w}")
    print(f"  ngrid: {args.ngrid}")
    print(f"  tolerance: {args.tolerance}")
    print("="*60)
    
    # Solve the GLE
    solution, solve_time = solve_gle(
        Ca=args.ca,
        mu_r=args.mu_r,
        lambda_slip=args.lambda_slip,
        theta0=theta0_rad,
        w_bc=args.w,
        Delta=args.delta,
        ngrid=args.ngrid,
        tolerance=args.tolerance
    )
    
    if solution is not None:
        print("\nSolution found successfully!")
        
        # Analyze solution
        x0, theta_min, _ = find_x0_and_theta_min(solution.x, solution.y[1, :])
        print(f"\nSolution properties:")
        print(f"  θ_min = {theta_min*180/np.pi:.2f}°")
        print(f"  x0 (position at θ_min) = {x0:.6f}")
        
        # Create plots and save data
        create_solution_plots(solution, args.ca, args.mu_r, args.lambda_slip,
                            theta0_rad, args.delta, args.output_dir)
        
        print(f"\nAll output saved to: {args.output_dir}/")
        
    else:
        print("\n" + "="*60)
        print("SOLUTION FAILED")
        print("="*60)
        print("\nThe solver could not find a solution for the given parameters.")
        print("This typically happens when Ca exceeds the critical value (Ca_cr).")
        print("\nTo find the critical Ca, please use one of these tools:")
        print("  1. python GLE_criticalCa.py          - Standard critical Ca finder")
        print("  2. python GLE_criticalCa_advanced.py - Advanced finder with adaptive mesh")
        print("\nExample:")
        print(f"  python GLE_criticalCa.py --mu_r {args.mu_r} --lambda_slip {args.lambda_slip}")
        print("="*60)


if __name__ == "__main__":
    main()