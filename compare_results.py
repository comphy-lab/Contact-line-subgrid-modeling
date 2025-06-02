#!/usr/bin/env python3
"""
compare_results.py

Script to compare outputs from Python and C implementations of the GLE solver.

Author: Vatsal Sanjay
Date: 2025-05-31
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys

def load_python_results():
    """Load results from Python solver output files"""
    data_file = 'output/data-python.csv'
    
    if not os.path.exists(data_file):
        print("Python solver output file not found. Please run the Python solver first.")
        return None, False
    
    # Load data
    df_python = pd.read_csv(data_file)
    # Add theta_deg column for compatibility
    df_python['theta_deg'] = df_python['theta'] * 180 / np.pi
    df_python['theta_rad'] = df_python['theta']  # theta is already in radians
    
    return df_python, True

def load_c_results():
    """Load results from C solver output files"""
    data_file = 'output/data-c-gsl.csv'

    if not os.path.exists(data_file):
        print("C solver output file not found. Please run the C solver first.")
        return None, False

    # Load consolidated data
    df_c = pd.read_csv(data_file)
    # Add theta_deg column for compatibility
    df_c['theta_deg'] = df_c['theta'] * 180 / np.pi
    df_c['theta_rad'] = df_c['theta']  # theta is already in radians

    return df_c, True

def compare_solutions(df_python, df_c):
    """Compare Python and C solutions"""
    # Interpolate C results to Python s values for comparison
    from scipy.interpolate import interp1d

    # Create interpolation functions for C results
    f_h_c = interp1d(df_c['s'], df_c['h'], bounds_error=False, fill_value='extrapolate')
    f_theta_c = interp1d(df_c['s'], df_c['theta_rad'], bounds_error=False, fill_value='extrapolate')

    # Interpolate to Python s values
    h_c_interp = f_h_c(df_python['s'])
    theta_c_interp = f_theta_c(df_python['s'])

    # Calculate differences
    h_diff = np.abs(df_python['h'] - h_c_interp)
    theta_diff = np.abs(df_python['theta_rad'] - theta_c_interp)

    # Calculate relative errors where values are not too small
    h_rel_err = np.where(np.abs(df_python['h']) > 1e-10,
                         h_diff / np.abs(df_python['h']),
                         h_diff)
    theta_rel_err = np.where(np.abs(df_python['theta_rad']) > 1e-10,
                            theta_diff / np.abs(df_python['theta_rad']),
                            theta_diff)

    # Print statistics
    print("\n=== Comparison Statistics ===")
    print(f"Maximum absolute difference in h: {np.max(h_diff):.6e}")
    print(f"Maximum relative error in h: {np.max(h_rel_err):.6e}")
    print(f"Maximum absolute difference in theta: {np.max(theta_diff):.6e} rad")
    print(f"Maximum relative error in theta: {np.max(theta_rel_err):.6e}")

    # Check boundary conditions
    print("\n=== Boundary Conditions Check ===")
    print(f"Python: h(0) = {df_python['h'].iloc[0]:.6e}, expected = 1e-5")
    print(f"C:      h(0) = {df_c['h'].iloc[0]:.6e}, expected = 1e-5")
    print(f"Python: theta(0) = {df_python['theta_deg'].iloc[0]:.6f}°, expected = 30°")
    print(f"C:      theta(0) = {df_c['theta_deg'].iloc[0]:.6f}°, expected = 30°")

    return h_diff, theta_diff

def plot_comparison(df_python, df_c):
    """Create comparison plots"""
    # Set up the figure with nice styling
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Define colors
    python_color = '#1f77b4'  # Blue
    c_color = '#ff7f0e'       # Orange

    # Plot h(s)
    ax1.plot(df_python['s'] * 1e6, df_python['h'] * 1e6, '-', 
             color=python_color, label='Python (solve_bvp)', linewidth=2.5)
    ax1.plot(df_c['s'] * 1e6, df_c['h'] * 1e6, '--', 
             color=c_color, label='C (GSL shooting method)', linewidth=2.5)
    ax1.set_xlabel('s [μm]', fontsize=12)
    ax1.set_ylabel('h(s) [μm]', fontsize=12)
    ax1.set_title('Film Thickness Profile Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=11, frameon=True, shadow=True)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 4e-4 * 1e6)
    
    # Add text box with parameters
    textstr = f'Ca = 1.0\nλ_slip = 1e-5\nμ_r = 1e-3'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    # Plot theta(s)
    ax2.plot(df_python['s'] * 1e6, df_python['theta_deg'], '-', 
             color=python_color, label='Python (solve_bvp)', linewidth=2.5)
    ax2.plot(df_c['s'] * 1e6, df_c['theta_deg'], '--', 
             color=c_color, label='C (GSL shooting method)', linewidth=2.5)
    ax2.set_xlabel('s [μm]', fontsize=12)
    ax2.set_ylabel('θ(s) [degrees]', fontsize=12)
    ax2.set_title('Contact Angle Profile Comparison', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11, frameon=True, shadow=True)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 4e-4 * 1e6)
    
    # Add initial condition text
    ax2.text(0.02, 0.05, f'θ(0) = 30°', transform=ax2.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    plt.savefig('output/comparison_python_vs_c.png', dpi=300, bbox_inches='tight')
    print("\nComparison plot saved to: output/comparison_python_vs_c.png")

def main():
    print("=== GLE Solver Output Comparison ===\n")

    # Load Python results
    df_python, python_exists = load_python_results()
    if not python_exists:
        print("\nPlease run the Python solver first:")
        print("  python GLE_solver.py")
        return 1

    # Load C results
    df_c, c_exists = load_c_results()
    if not c_exists:
        print("\nPlease run the C solver first:")
        print("  make run")
        return 1

    # Compare solutions
    h_diff, theta_diff = compare_solutions(df_python, df_c)

    # Create comparison plots
    plot_comparison(df_python, df_c)

    # Determine if solutions match within tolerance
    tol = 1e-6
    if np.max(h_diff) < tol and np.max(theta_diff) < tol:
        print(f"\n✓ Solutions match within tolerance ({tol})")
        return 0
    else:
        print(f"\n✗ Solutions differ by more than tolerance ({tol})")
        return 1

if __name__ == "__main__":
    sys.exit(main())
