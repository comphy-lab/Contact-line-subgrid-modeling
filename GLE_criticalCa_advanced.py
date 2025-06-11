"""
Generalized Lubrication Equations (GLE) Solver - Advanced Critical Ca Finder
===========================================================================

This module implements an advanced critical Capillary number (Ca_cr) finder
with adaptive mesh refinement. It provides a more robust and accurate method
compared to the basic finder in GLE_solver.py.

Features:
- Adaptive mesh refinement based on solution properties
- Optimized search strategy

Note: This is NOT a continuation method - it specifically finds Ca_critical
where θ_min → 0. A proper continuation method that traces the full bifurcation
diagram will be implemented separately.

Author: Vatsal and Aman
Created: 2025
"""

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

# Import critical Ca finding from GLE_criticalCa
from GLE_criticalCa import find_critical_ca_lower_branch

# Default parameters
DEFAULT_DELTA = 10.0  # Large domain for continuation
DEFAULT_LAMBDA_SLIP = 1e-4  # Slip length
DEFAULT_MU_R = 1e-6  # \mu_g/\mu_l
DEFAULT_THETA0 = np.pi/2  # theta at s = 0
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



def find_critical_ca_advanced(mu_r, lambda_slip, theta0=DEFAULT_THETA0, w_bc=DEFAULT_W,
                            Delta=DEFAULT_DELTA, output_dir='output',
                            tolerance=1e-6, adaptive_mesh=True):
    """
    Advanced critical Ca finder with enhanced Phase 0 only.
    
    Features:
    - Adaptive mesh refinement near critical point
    - Optimized search strategy
    
    This is NOT a continuation method - it specifically finds Ca_critical
    """
    print(f"\nADVANCED CRITICAL Ca FINDER")
    print(f"Parameters: theta_0 = {theta0/np.pi*180:.1f}°, mu_r = {mu_r}, lambda_slip = {lambda_slip}")
    print(f"Tolerance: {tolerance}")
    print(f"Method: Enhanced Phase 0 with adaptive mesh refinement")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Enhanced Phase 0: Advanced Ca_cr estimation
    print("\n" + "="*60)
    print("ENHANCED CRITICAL Ca SEARCH")
    print("="*60)
    start_time = time.time()
    
    # Step 1: Initial estimate with standard method
    print("Step 1: Initial Ca_cr estimate...")
    nGridInit = DEFAULT_NGRID
    if adaptive_mesh:
        # Start with moderate mesh for initial search
        nGridInit = min(DEFAULT_NGRID, 5000)
        
    Ca_cr_initial, Ca_values, x0_values, theta_min_values, s_range_initial, y_guess_initial = find_critical_ca_lower_branch(
        mu_r, lambda_slip, theta0, w_bc, Delta, 
        nGridInit=nGridInit, 
        output_dir=output_dir,
        Ca_requested=1.0,  # Large value to trigger search
        tolerance=tolerance*10  # Coarser tolerance for initial search
    )
    
    print(f"\nInitial estimate: Ca_cr ≈ {Ca_cr_initial:.6f}")
    
    # Step 2: Adaptive mesh refinement near critical Ca
    if adaptive_mesh and Ca_cr_initial > 0:
        print("\nStep 2: Adaptive mesh refinement near critical Ca...")
        
        # Analyze solution to determine mesh requirements
        x0_test, theta_min_test, solution_test = solve_bvp_for_ca(
            Ca_cr_initial * 0.95, mu_r, lambda_slip, theta0, w_bc, Delta
        )
        
        if solution_test is not None:
            props = analyze_solution_properties(solution_test, Ca_cr_initial * 0.95, verbose=True)
            
            # Determine optimal mesh size based on solution gradients
            if props['max_omega'] > 100 or props['max_theta_grad'] > 10:
                nGridRefined = min(50000, DEFAULT_NGRID * 5)
                print(f"  High curvature detected. Increasing mesh to {nGridRefined} points")
            elif props['max_omega'] > 50 or props['max_theta_grad'] > 5:
                nGridRefined = min(30000, DEFAULT_NGRID * 3)
                print(f"  Moderate curvature detected. Increasing mesh to {nGridRefined} points")
            else:
                nGridRefined = DEFAULT_NGRID
                print(f"  Standard mesh sufficient: {nGridRefined} points")
            
            # Refine the search with better mesh
            if nGridRefined > nGridInit:
                print("  Refining search with adaptive mesh...")
                Ca_cr_refined, Ca_values_refined, x0_values_refined, theta_min_values_refined, s_range_refined, y_guess_refined = find_critical_ca_lower_branch(
                    mu_r, lambda_slip, theta0, w_bc, Delta, 
                    nGridInit=nGridRefined, 
                    output_dir=output_dir,
                    Ca_requested=Ca_cr_initial * 1.2,  # Search near initial estimate
                    tolerance=tolerance
                )
                
                if Ca_cr_refined > 0:
                    Ca_cr_initial = Ca_cr_refined
                    Ca_values.extend(Ca_values_refined)
                    x0_values.extend(x0_values_refined)
                    theta_min_values.extend(theta_min_values_refined)
                    print(f"  Refined estimate: Ca_cr ≈ {Ca_cr_refined:.8f}")
    
    # Use the best computed Ca value (not extrapolated)
    Ca_critical = Ca_cr_initial
    
    total_time = time.time() - start_time
    print(f"\nCompleted in {total_time:.1f} seconds")
    print(f"Final Ca_critical = {Ca_critical:.8f}")
    
    # Get theta_min at the critical Ca
    if len(theta_min_values) > 0:
        theta_min_final = theta_min_values[-1]
        print(f"θ_min at last computed point = {theta_min_final:.3e} rad ({theta_min_final*180/np.pi:.3e}°)")
    else:
        theta_min_final = 0.0
    
    # Prepare data for plotting and saving
    # Sort all data by Ca
    if len(Ca_values) > 0:
        sort_idx = np.argsort(Ca_values)
        Ca_sorted = np.array(Ca_values)[sort_idx]
        theta_min_sorted = np.array(theta_min_values)[sort_idx]
        x0_sorted = np.array(x0_values)[sort_idx]
    else:
        Ca_sorted = np.array([Ca_critical])
        theta_min_sorted = np.array([0.0])
        x0_sorted = np.array([np.nan])
    
    # The critical point is the last computed value
    # No need to add extrapolated point
    
    # Create plots
    plot_critical_ca_results(Ca_sorted, theta_min_sorted, x0_sorted, 
                           Ca_critical, theta_min_final, mu_r, lambda_slip, output_dir)
    
    # Save data
    csv_data = np.column_stack((Ca_sorted, theta_min_sorted, theta_min_sorted*180/np.pi, x0_sorted))
    csv_path = os.path.join(output_dir, 'pyGLE_criticalCa_advanced.csv')
    np.savetxt(csv_path, csv_data, delimiter=',',
               header='Ca,theta_min_rad,theta_min_deg,x0', 
               comments='')
    print(f"\nData saved to: {csv_path}")
    
    # Step 4: Create final solution plots at Ca_critical
    print("\nGenerating final solution plots at Ca_critical...")
    # Get the final solution (use existing cached solution if available)
    x0_final, theta_min_final, solution_final = solve_bvp_for_ca(
        Ca_critical, mu_r, lambda_slip, theta0, w_bc, Delta,
        s_range=s_range_initial, y_guess=y_guess_initial, tol=tolerance
    )
    
    if solution_final is not None:
        create_final_solution_plots(solution_final, Ca_critical, mu_r, lambda_slip, 
                                   theta0, w_bc, Delta, output_dir, tolerance)
    else:
        print("Warning: Could not generate final solution plots")
    
    return Ca_critical


def plot_critical_ca_results(Ca_sorted, theta_min_sorted, x0_sorted,
                           Ca_critical, theta_min_final, mu_r, lambda_slip, output_dir):
    """Create plots showing the approach to critical Ca"""
    
    # Create plots
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot 1: theta_min vs Ca
    theta_min_deg = theta_min_sorted * 180 / np.pi
    
    # Plot line
    ax1.plot(Ca_sorted, theta_min_deg, 'b-', linewidth=2.5, alpha=0.7)
    
    # Add scatter points
    ax1.scatter(Ca_sorted, theta_min_deg, 
                color='blue', s=50, alpha=0.7, marker='o', label='Computed points')
    
    # Highlight the last few points near critical
    if len(Ca_sorted) > 5:
        ax1.scatter(Ca_sorted[-6:-1], theta_min_deg[-6:-1], 
                    color='darkblue', s=80, alpha=0.9, marker='s', 
                    label='Near critical points')
    
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
            axins.scatter(Ca_sorted[near_critical], theta_min_deg[near_critical], 
                        color='blue', s=40, marker='o')
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
        ax2.scatter(Ca_sorted[valid_x0_mask], x0_sorted[valid_x0_mask], 
                   color='green', s=50, alpha=0.7, marker='o')
        
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
    
    plot_path = os.path.join(output_dir, 'pyGLE_criticalCa_advanced.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {plot_path}")


def create_final_solution_plots(solution, Ca_critical, mu_r, lambda_slip, 
                               theta0, w_bc, Delta, output_dir, tolerance):
    """Create detailed solution plots at the critical Ca, similar to GLE_solver.py"""
    
    # Extract solution data
    s_values = solution.x
    h_values, theta_values, w_values = solution.y
    theta_values_deg = theta_values * 180 / np.pi
    
    # Calculate x(s) by integrating cos(theta)
    x_values = np.zeros_like(s_values)
    x_values[1:] = np.cumsum(np.diff(s_values) * np.cos(theta_values[:-1]))
    
    # Find x0 and theta_min
    x0, theta_min, x0_idx = find_x0_and_theta_min(s_values, theta_values)
    
    # Create figure with 2x2 subplots
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # Set title
    decimal_places = get_decimal_places_from_tolerance(tolerance)
    fig.suptitle(f'GLE Solution at Critical Ca = {Ca_critical:.{decimal_places}f}', 
                 fontsize=16, fontweight='bold', color='darkred')
    
    solver_color = '#1f77b4'
    
    # Plot 1: h(s) vs s
    ax1.plot(s_values, h_values, '-', color=solver_color, linewidth=2.5)
    ax1.set_xlabel('s', fontsize=14)
    ax1.set_ylabel('h(s)', fontsize=14)
    ax1.set_title('Film Thickness Profile', fontsize=16, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, Delta)
    ax1.set_ylim(lambda_slip, 1.01*max(h_values))
    
    # Add parameter box
    textstr = f'Ca = {Ca_critical:.{decimal_places}f} (Ca_cr)\n$\\lambda_\\text{{slip}}$ = {lambda_slip:.0e}\n$\\mu_r$ = {mu_r:.0e}'
    props = dict(boxstyle='round', facecolor='salmon', alpha=0.5)
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
    
    # Add legend with initial angle and minimum angle
    ax2.plot([], [], ' ', label=f'θ(0) = {theta0*180/np.pi:.0f}°')
    ax2.plot([], [], ' ', label=f'θ_min = {theta_min*180/np.pi:.2f}°')
    ax2.legend(loc='best', fontsize=12, fancybox=True, framealpha=0.8, 
               facecolor='lightblue', edgecolor='none')
    
    # Mark the minimum point
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
                    label=f'$(h, x)$ at $\\theta_{{\\min}}$ = ({h_at_theta_min:.3f}, {x0:.3f})')
        ax4.legend()
    
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'pyGLE_criticalCa_advanced-profiles.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Final solution plot saved to: {plot_path}")
    
    # Save solution data to CSV
    csv_data = np.column_stack((s_values, h_values, theta_values, w_values, x_values))
    csv_path = os.path.join(output_dir, 'pyGLE_criticalCa_advanced-profiles.csv')
    np.savetxt(csv_path, csv_data, delimiter=',', 
               header='s,h,theta,w,x', comments='')
    print(f"Solution data saved to: {csv_path}")


# Helper function for decimal places
def get_decimal_places_from_tolerance(tolerance):
    """Determine appropriate decimal places based on tolerance"""
    if tolerance >= 1:
        return 0
    elif tolerance >= 0.1:
        return 1
    elif tolerance >= 0.01:
        return 2
    elif tolerance >= 0.001:
        return 3
    elif tolerance >= 0.0001:
        return 4
    elif tolerance >= 0.00001:
        return 5
    elif tolerance >= 0.000001:
        return 6
    elif tolerance >= 0.0000001:
        return 7
    else:
        return 8


# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Advanced critical Ca finder with adaptive mesh refinement')
    
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
    
    # Single tolerance parameter
    parser.add_argument('--tolerance', type=float, default=1e-6,
                        help='Tolerance for Ca_cr refinement (default: 1e-6)')
    parser.add_argument('--adaptive-mesh', action='store_true', default=True,
                        help='Use adaptive mesh refinement (default: True)')
    parser.add_argument('--no-adaptive-mesh', dest='adaptive_mesh', action='store_false',
                        help='Disable adaptive mesh refinement')
    
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
        tolerance=args.tolerance,
        adaptive_mesh=args.adaptive_mesh
    )
    
    if Ca_critical is not None:
        print(f"\n{'='*60}")
        print(f"FINAL RESULT: Ca_critical = {Ca_critical:.8f}")
        print(f"{'='*60}")
    else:
        print("\nFailed to find critical Ca")