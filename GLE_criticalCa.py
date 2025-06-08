"""
Generalized Lubrication Equations (GLE) Solver with Critical Ca Finder
======================================================================

This module solves the coupled ODEs for contact line dynamics using the
Generalized Lubrication Equations. It includes an optimized critical 
Capillary number (Ca_cr) finder using a hybrid approach:

- Two-stage search: coarse logarithmic search followed by refinement
- Refinement uses Inverse Quadratic Interpolation (IQI), Newton-Raphson,
  and bisection methods adaptively based on solution behavior
- Parallel execution is used only for initial exploration of multiple
  Ca values (via ThreadPoolExecutor)

The solver automatically detects when the requested Ca exceeds Ca_critical
and finds the critical value where θ_min approaches zero.

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any
import time

# Import utilities from src-local
import sys
sys.path.append('src-local')
from find_x0_utils import find_x0_and_theta_min
from solution_types import SolutionResult, SolutionCache
from gle_utils import (
    solve_single_ca, GLE, boundary_conditions,
    f1, f2, f3, f, get_adaptive_max_nodes
)
from ca_continuation_utils import (
    get_decimal_places_from_tolerance,
    get_adaptive_tolerance, interpolate_solution,
    inverse_quadratic_interpolation, hybrid_iqi_newton_refinement,
    find_critical_ca_lower_branch
)

# Default parameters (can be overridden via command line)
DEFAULT_DELTA = 1e1  # Maximum s-value for the solver
DEFAULT_CA = 0.01  # Capillary number (initial guess, same as GLE_solver.py)
DEFAULT_LAMBDA_SLIP = 1e-4  # Slip length
DEFAULT_MU_R = 1e-6  # μ_g/μ_l
DEFAULT_THETA0 = np.pi/2  # theta at s = 0
DEFAULT_W = 0  # curvature boundary condition at s = Δ
DEFAULT_NGRID_INIT = 10000  # Initial number of grid points


def parallel_solve_multiple_ca(Ca_list: List[float], mu_r: float, lambda_slip: float, 
                              theta0: float, w_bc: float, Delta: float,
                              s_range_list: List[np.ndarray], y_guess_list: List[np.ndarray],
                              tol_list: List[float], max_nodes_list: List[int],
                              max_workers: int = 4) -> List[SolutionResult]:
    """
    Solve for multiple Ca values in parallel using threading.
    
    Returns:
        List of SolutionResult objects sorted by Ca
    """
    with ThreadPoolExecutor(max_workers=min(len(Ca_list), max_workers)) as executor:
        # Submit all tasks
        futures = {}
        for Ca, s_range, y_guess, tol, max_nodes in zip(
            Ca_list, s_range_list, y_guess_list, tol_list, max_nodes_list
        ):
            future = executor.submit(
                solve_single_ca, Ca, mu_r, lambda_slip, theta0, w_bc, Delta, 
                s_range, y_guess, tol, max_nodes
            )
            futures[future] = Ca
        
        # Collect results
        results = []
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
    
    return sorted(results, key=lambda x: x.Ca)

# ============================================================================
# Plotting Functions
# ============================================================================

def plot_ca_search_results(Ca_values: List[float], x0_values: List[float], 
                         theta_min_values: List[float], output_dir: str = 'output',
                         tolerance: float = 1e-6) -> None:
    """Create plots showing theta_min vs Ca and x_0 vs Ca during critical Ca search (x0 is position where theta is minimum)."""
    if not Ca_values:
        return
        
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # If x0_values is empty or all None, only plot theta_min
    has_x0_data = x0_values and any(x is not None for x in x0_values)
    
    if has_x0_data:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
        ax2 = None
    
    # Convert theta_min to degrees
    theta_min_deg = np.array(theta_min_values) * 180 / np.pi
    
    # Plot 1: theta_min vs Ca
    ax1.scatter(Ca_values, theta_min_deg, s=50, alpha=0.7, color='darkred', 
                edgecolors='black', linewidth=1)
    ax1.plot(Ca_values, theta_min_deg, '-', color='crimson', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Ca (Capillary Number)', fontsize=12)
    ax1.set_ylabel('$\\theta_{min}$ [degrees]', fontsize=12)
    ax1.set_title('Minimum Contact Angle vs Capillary Number', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    if max(Ca_values) / min(Ca_values) > 100:
        ax1.set_xscale('log')
        ax1.set_xlabel('Ca (Capillary Number) [log scale]', fontsize=12)
    
    # Mark critical Ca on theta_min plot
    Ca_cr = Ca_values[-1]
    decimal_places = get_decimal_places_from_tolerance(tolerance)
    ax1.axvline(x=Ca_cr, color='green', linestyle='--', linewidth=2, alpha=0.7, 
                label=f'Ca_cr ≈ {Ca_cr:.{decimal_places}f} ± {tolerance}')
    ax1.legend(fontsize=14)
    
    # Plot 2: x_0 vs Ca (only if x0 data exists)
    if has_x0_data and ax2 is not None:
        # Filter out None values for plotting
        Ca_with_x0 = [Ca for Ca, x0 in zip(Ca_values, x0_values) if x0 is not None]
        x0_filtered = [x0 for x0 in x0_values if x0 is not None]
        
        if Ca_with_x0 and x0_filtered:
            ax2.scatter(Ca_with_x0, x0_filtered, s=50, alpha=0.7, color='darkblue', 
                        edgecolors='black', linewidth=1)
            ax2.plot(Ca_with_x0, x0_filtered, '-', color='blue', linewidth=2, alpha=0.8)
            ax2.set_xlabel('Ca (Capillary Number)', fontsize=12)
            ax2.set_ylabel('$x_0$ (Position at $\\theta_{min}$)', fontsize=12)
            ax2.set_title('Position at Minimum Contact Angle vs Capillary Number', 
                          fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            
            if max(Ca_with_x0) / min(Ca_with_x0) > 100:
                ax2.set_xscale('log')
                ax2.set_xlabel('Ca (Capillary Number) [log scale]', fontsize=12)
            
            ax2.axvline(x=Ca_cr, color='green', linestyle='--', linewidth=2, alpha=0.7,
                        label=f'Ca_cr ≈ {Ca_cr:.{decimal_places}f} ± {tolerance}')
            ax2.legend(fontsize=14)
    
    plt.suptitle('Critical Ca Search Results', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'pyGLE_criticalCa.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Critical Ca search results plot saved to: {plot_path}")
    
    # Save data to CSV
    if has_x0_data:
        # Include x0 values, replacing None with NaN
        x0_for_csv = [x if x is not None else np.nan for x in x0_values]
        csv_data = np.column_stack((Ca_values, theta_min_values, theta_min_deg, x0_for_csv))
        csv_path = os.path.join(output_dir, 'pyGLE_criticalCa.csv')
        np.savetxt(csv_path, csv_data, delimiter=',', 
                   header='Ca,theta_min_rad,theta_min_deg,x0', comments='')
    else:
        # No x0 data
        csv_data = np.column_stack((Ca_values, theta_min_values, theta_min_deg))
        csv_path = os.path.join(output_dir, 'pyGLE_criticalCa.csv')
        np.savetxt(csv_path, csv_data, delimiter=',', 
                   header='Ca,theta_min_rad,theta_min_deg', comments='')
    print(f"Critical Ca search data saved to: {csv_path}")

def setup_initial_guess(Delta: float, nGridInit: int, lambda_slip: float, 
                       theta0: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create initial guess for the BVP solver.
    
    Args:
        Delta: Maximum s-value
        nGridInit: Number of grid points
        lambda_slip: Slip length
        theta0: Initial contact angle
        
    Returns:
        (s_range, y_guess): Grid and initial guess arrays
    """
    s_range = np.linspace(0, Delta, nGridInit)
    y_guess = np.zeros((3, s_range.size))
    # Use a more physical initial guess for h
    # h grows from lambda_slip following approximately h ~ s for small s
    y_guess[0, :] = lambda_slip + s_range * np.sin(theta0)
    y_guess[1, :] = theta0
    y_guess[2, :] = 0
    
    return s_range, y_guess


def attempt_direct_solve(Ca: float, mu_r: float, lambda_slip: float, theta0: float,
                        w: float, Delta: float, s_range: np.ndarray, 
                        y_guess: np.ndarray) -> Tuple[SolutionResult, float]:
    """
    Attempt to solve directly at the requested Ca.
    
    Returns:
        (result, time_elapsed): Solution result and time taken
    """
    print(f"\nAttempting direct solve with Ca = {Ca}, mu_r = {mu_r}...")
    start_time = time.time()
    result = solve_single_ca(Ca, mu_r, lambda_slip, theta0, w, Delta, 
                           s_range, y_guess)
    time_elapsed = time.time() - start_time
    
    if result.success:
        print(f"Direct solve succeeded in {time_elapsed:.2f} seconds")
    else:
        print(f"Direct solve failed after {time_elapsed:.2f} seconds")
    
    return result, time_elapsed


def find_critical_ca_with_timing(mu_r: float, lambda_slip: float, theta0: float,
                                w: float, Delta: float, nGridInit: int, 
                                output_dir: str, Ca_requested: float,
                                tolerance: float) -> Tuple:
    """
    Find critical Ca with detailed timing information.
    
    Returns:
        (Ca_cr, Ca_values, x0_values, theta_min_values, s_range_refined, 
         y_guess_refined, timing_info)
    """
    print("\nFinding critical Ca...")
    start_time_total = time.time()
    
    # Call the critical Ca finder from ca_continuation_utils
    result = find_critical_ca_lower_branch(
        mu_r, lambda_slip, theta0, w, Delta, nGridInit, output_dir, 
        Ca_requested=Ca_requested, tolerance=tolerance
    )
    
    total_time = time.time() - start_time_total
    
    # Extract results
    Ca_cr, Ca_values, x0_values, theta_min_values, s_range_refined, y_guess_refined = result
    
    # Create timing info dictionary
    timing_info = {
        'total_time': total_time,
        'solutions_found': len(Ca_values),
        'avg_time_per_solution': total_time / len(Ca_values) if Ca_values else 0
    }
    
    print(f"\nCritical Ca search completed in {total_time:.2f} seconds")
    print(f"Found {len(Ca_values)} solutions")
    if Ca_values:
        print(f"Average time per solution: {timing_info['avg_time_per_solution']:.2f} seconds")
    
    return Ca_cr, Ca_values, x0_values, theta_min_values, s_range_refined, y_guess_refined, timing_info


def save_solution_data(s_values: np.ndarray, h_values: np.ndarray, 
                      theta_values: np.ndarray, w_values: np.ndarray,
                      x_values: np.ndarray, output_dir: str) -> str:
    """
    Save solution data to CSV file.
    
    Returns:
        Path to saved file
    """
    csv_data = np.column_stack((s_values, h_values, theta_values, 
                               w_values, x_values))
    csv_path = os.path.join(output_dir, 'pyGLE_criticalCa-profiles.csv')
    np.savetxt(csv_path, csv_data, delimiter=',', 
               header='s,h,theta,w,x', comments='')
    return csv_path


def run_solver_and_plot(Delta: float, Ca: float, lambda_slip: float, mu_r: float, 
                       theta0: float, w: float, nGridInit: int = DEFAULT_NGRID_INIT, 
                       GUI: bool = False, output_dir: str = 'output',
                       tolerance: float = 1e-6) -> Tuple:
    """
    Run the solver and create plots with detailed timing information.
    
    Returns:
        (solution, s_values, h_values, theta_values, w_values)
    """
    total_start_time = time.time()
    
    # Set matplotlib backend
    if not GUI:
        import matplotlib
        matplotlib.use('Agg')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Setup initial guess
    s_range_local, y_guess_local = setup_initial_guess(Delta, nGridInit, 
                                                       lambda_slip, theta0)

    # Step 2: Try direct solve first
    result, direct_solve_time = attempt_direct_solve(Ca, mu_r, lambda_slip, 
                                                     theta0, w, Delta, 
                                                     s_range_local, y_guess_local)
    
    Ca_actual = Ca
    Ca_cr = None
    Ca_values = []
    x0_values = []
    theta_min_values = []
    timing_info = {'direct_solve_time': direct_solve_time}
    
    # Step 3: If direct solve fails, find critical Ca
    if not result.success:
        (Ca_cr, Ca_values, x0_values, theta_min_values, 
         s_range_refined, y_guess_refined, ca_timing) = find_critical_ca_with_timing(
            mu_r, lambda_slip, theta0, w, Delta, nGridInit, output_dir, 
            Ca_requested=Ca, tolerance=tolerance
        )
        
        timing_info.update(ca_timing)
        
        # Step 4: Solve at critical Ca
        if Ca_cr > 0:
            Ca_actual = Ca_cr
            print(f"\nSolving at critical Ca = {Ca_cr:.{get_decimal_places_from_tolerance(tolerance)}f}")
            
            final_solve_start = time.time()
            result = solve_single_ca(Ca_actual, mu_r, lambda_slip, theta0, w, Delta,
                                   s_range_refined, y_guess_refined)
            timing_info['final_solve_time'] = time.time() - final_solve_start
        else:
            print("Failed to find critical Ca")
            return None, None, None, None, None

    if not result.success:
        print("Failed to obtain solution")
        return None, None, None, None, None

    # Step 5: Extract and process solution
    solution = result.solution
    s_values_local = solution.x
    h_values_local, theta_values_local, w_values_local = solution.y
    
    # Calculate x(s)
    x_values_local = np.zeros_like(s_values_local)
    x_values_local[1:] = np.cumsum(np.diff(s_values_local) * np.cos(theta_values_local[:-1]))

    # Step 6: Create plots
    plotting_start = time.time()
    create_solution_plots(s_values_local, h_values_local, theta_values_local, 
                         w_values_local, x_values_local, Ca_actual, Ca, Ca_cr,
                         lambda_slip, mu_r, theta0, Delta, GUI, output_dir, tolerance)

    # Plot critical Ca search results if available
    if Ca_values:
        plot_ca_search_results(Ca_values, x0_values, theta_min_values, output_dir, tolerance)
    
    timing_info['plotting_time'] = time.time() - plotting_start

    # Step 7: Save data
    csv_path = save_solution_data(s_values_local, h_values_local, theta_values_local,
                                 w_values_local, x_values_local, output_dir)
    print(f"Data saved to: {csv_path}")

    # Step 8: Print timing summary
    total_time = time.time() - total_start_time
    print_timing_summary(timing_info, total_time)

    return solution, s_values_local, h_values_local, theta_values_local, w_values_local


def print_timing_summary(timing_info: Dict, total_time: float) -> None:
    """Print detailed timing information."""
    print("\n" + "="*60)
    print("TIMING SUMMARY")
    print("="*60)
    
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"  - Direct solve attempt: {timing_info.get('direct_solve_time', 0):.2f} seconds")
    
    if 'total_time' in timing_info:
        print(f"  - Critical Ca search: {timing_info['total_time']:.2f} seconds")
        if timing_info.get('solutions_found', 0) > 0:
            print(f"    * Solutions evaluated: {timing_info['solutions_found']}")
            print(f"    * Average per solution: {timing_info['avg_time_per_solution']:.2f} seconds")
    
    if 'final_solve_time' in timing_info:
        print(f"  - Final solve at Ca_cr: {timing_info['final_solve_time']:.2f} seconds")
    
    print(f"  - Plotting and I/O: {timing_info.get('plotting_time', 0):.2f} seconds")
    print("="*60)

def create_solution_plots(s_values: np.ndarray, h_values: np.ndarray, 
                         theta_values: np.ndarray, w_values: np.ndarray,
                         x_values: np.ndarray, Ca_actual: float, Ca_requested: float,
                         Ca_cr: Optional[float], lambda_slip: float, mu_r: float,
                         theta0: float, Delta: float, GUI: bool, output_dir: str,
                         tolerance: float = 1e-6) -> None:
    """Create the 2x2 subplot grid for solution visualization."""
    plt.style.use('seaborn-v0_8-darkgrid')
    
    theta_values_deg = theta_values * 180 / np.pi
    solver_color = '#1f77b4'
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # Title
    if Ca_cr is not None:
        decimal_places = get_decimal_places_from_tolerance(tolerance) if tolerance else 6
        fig.suptitle(f'GLE Solution at Critical Ca = {Ca_actual:.{decimal_places}f} (requested Ca = {Ca_requested:.{decimal_places}f})', 
                     fontsize=16, fontweight='bold', color='darkred')
    else:
        decimal_places = get_decimal_places_from_tolerance(tolerance) if tolerance else 6
        fig.suptitle(f'GLE Solution at Ca = {Ca_actual:.{decimal_places}f}', 
                     fontsize=16, fontweight='bold')
    
    # Plot 1: h(s) vs s
    ax1.plot(s_values, h_values, '-', color=solver_color, linewidth=2.5)
    ax1.set_xlabel('s', fontsize=14)
    ax1.set_ylabel('h(s)', fontsize=14)
    ax1.set_title('Film Thickness Profile', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, Delta)
    ax1.set_ylim(lambda_slip, 1.01*max(h_values))
    
    # Parameter box
    if Ca_cr is not None:
        decimal_places = get_decimal_places_from_tolerance(tolerance)
        textstr = f'Ca = {Ca_actual:.{decimal_places}f} (Ca_cr)\n$\\lambda_\\text{{slip}}$ = {lambda_slip:.0e}\n$\\mu_r$ = {mu_r:.0e}'
        facecolor = 'salmon'
    else:
        decimal_places = get_decimal_places_from_tolerance(tolerance)
        textstr = f'Ca = {Ca_actual:.{decimal_places}f}\n$\\lambda_\\text{{slip}}$ = {lambda_slip:.0e}\n$\\mu_r$ = {mu_r:.0e}'
        facecolor = 'wheat'
    props = dict(boxstyle='round', facecolor=facecolor, alpha=0.5)
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
    ax2.plot([], [], ' ', label=f'θ(0) = {theta0*180/np.pi:.0f}°')
    ax2.legend(loc='best', fontsize=12, fancybox=True, framealpha=0.8, 
               facecolor='lightblue', edgecolor='none')
    
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
    
    # Mark x0 position (where theta is minimum)
    x0, theta_min, x0_idx = find_x0_and_theta_min(s_values, theta_values)
    if x0 is not None and x0_idx is not None:
        # Get the h-value and x-value at the position where theta is minimum
        h_at_theta_min = h_values[x0_idx]
        x_at_theta_min = x_values[x0_idx]
        ax4.scatter(h_at_theta_min, x_at_theta_min, color='red', s=100, 
                    marker='o', zorder=5, edgecolors='darkred', linewidth=2,
                    label=f'$(h, x)$ at $\\theta_{{\\min}}$')
        ax4.legend()
    
    plt.tight_layout()
    
    if GUI:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, 'pyGLE_criticalCa-profiles.png'), dpi=300, bbox_inches='tight')
        plt.close()

# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(
        description='Solve Generalized Lubrication Equations (GLE) with critical Ca finder using hybrid IQI/Newton/bisection refinement'
    )
    
    # Add arguments
    parser.add_argument('--delta', type=float, default=DEFAULT_DELTA,
                        help=f'Maximum s-value for the solver (default: {DEFAULT_DELTA})')
    parser.add_argument('--ca', type=float, default=DEFAULT_CA,
                        help=f'Initial Ca to attempt (default: {DEFAULT_CA}). If it fails, will search for critical Ca')
    parser.add_argument('--lambda_slip', type=float, default=DEFAULT_LAMBDA_SLIP,
                        help=f'Slip length (default: {DEFAULT_LAMBDA_SLIP})')
    parser.add_argument('--mu_r', type=float, default=DEFAULT_MU_R,
                        help=f'Viscosity ratio μ_g/μ_l (default: {DEFAULT_MU_R})')
    parser.add_argument('--theta0', type=float, default=DEFAULT_THETA0*180/np.pi,
                        help=f'Initial contact angle in degrees (default: {DEFAULT_THETA0*180/np.pi:.0f})')
    parser.add_argument('--w', type=float, default=DEFAULT_W,
                        help=f'Curvature boundary condition at s=Delta (default: {DEFAULT_W})')
    parser.add_argument('--ngrid-init', type=int, default=DEFAULT_NGRID_INIT,
                        help=f'Initial number of grid points (default: {DEFAULT_NGRID_INIT})')
    parser.add_argument('--tolerance', type=float, default=1e-6,
                        help='Tolerance for Ca_cr refinement (default: 1e-6)')
    parser.add_argument('--gui', action='store_true',
                        help='Display plots in GUI mode')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory for plots and data (default: output)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert theta0 from degrees to radians
    theta0_rad = args.theta0 * np.pi / 180
    
    # Run solver
    solution, s_values, h_values, theta_values, w_values = run_solver_and_plot(
        Delta=args.delta,
        Ca=args.ca,
        lambda_slip=args.lambda_slip,
        mu_r=args.mu_r,
        theta0=theta0_rad,
        w=args.w,
        nGridInit=args.ngrid_init,
        GUI=args.gui,
        output_dir=args.output_dir,
        tolerance=args.tolerance
    )

    if solution is not None:
        print(f"Solution converged: {solution.success}")
        print(f"Number of iterations: {solution.niter}")
    else:
        print("No solution found.")
    
    # Print parameters used
    print("\nParameters used:")
    print(f"  Delta: {args.delta}")
    print(f"  Ca: {args.ca}")
    print(f"  lambda_slip: {args.lambda_slip}")
    print(f"  mu_r: {args.mu_r}")
    print(f"  theta0: {args.theta0}° ({theta0_rad:.4f} rad)")
    print(f"  w: {args.w}")
    print(f"  tolerance: {args.tolerance}")

    if not args.gui:
        print(f"\nPlots saved to: {args.output_dir}/")