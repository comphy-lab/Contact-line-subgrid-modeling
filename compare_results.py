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
    # Python solver doesn't save to CSV, so we need to run it and capture results
    from GLE_solver import run_solver_and_plot

    print("Running Python solver...")
    solution, s_values, h_values, theta_values, w_values = run_solver_and_plot(GUI=False)

    # Convert to DataFrame for easier comparison
    df_python = pd.DataFrame({
        's': s_values,
        'h': h_values,
        'theta_rad': theta_values,
        'theta_deg': theta_values * 180 / np.pi,
        'omega': w_values
    })

    return df_python, solution.success

def load_c_results():
    """Load results from C solver output files"""
    h_file = 'output/GLE_h_profile_c.csv'
    theta_file = 'output/GLE_theta_profile_c.csv'

    if not os.path.exists(h_file) or not os.path.exists(theta_file):
        print("C solver output files not found. Please run the C solver first.")
        return None, False

    # Load h data
    df_h = pd.read_csv(h_file)
    df_theta = pd.read_csv(theta_file)

    # Merge on s values
    df_c = pd.merge(df_h, df_theta, on='s')
    df_c['theta_rad'] = df_c['theta_deg'] * np.pi / 180

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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

    # Plot h(s)
    ax1.plot(df_python['s'], df_python['h'], 'b-', label='Python', linewidth=2)
    ax1.plot(df_c['s'], df_c['h'], 'r--', label='C', linewidth=2, alpha=0.7)
    ax1.set_xlabel('s')
    ax1.set_ylabel('h(s)')
    ax1.set_title('Film Thickness Profile Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 4e-4)

    # Plot theta(s)
    ax2.plot(df_python['s'], df_python['theta_deg'], 'b-', label='Python', linewidth=2)
    ax2.plot(df_c['s'], df_c['theta_deg'], 'r--', label='C', linewidth=2, alpha=0.7)
    ax2.set_xlabel('s')
    ax2.set_ylabel('θ(s) [degrees]')
    ax2.set_title('Contact Angle Profile Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 4e-4)

    plt.tight_layout()
    plt.savefig('output/comparison_python_vs_c.png', dpi=150, bbox_inches='tight')
    print("\nComparison plot saved to: output/comparison_python_vs_c.png")

def main():
    print("=== GLE Solver Output Comparison ===\n")

    # Load Python results
    df_python, python_success = load_python_results()
    if not python_success:
        print("WARNING: Python solver did not converge successfully!")

    # Load C results
    df_c, c_exists = load_c_results()
    if not c_exists:
        print("\nPlease run the C solver first:")
        print("  make -f Makefile.corrected run")
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
