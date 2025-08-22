"""
Generalized Lubrication Equations (GLE) Solver for Horizontal Plate with h-based Boundary Condition

This code solves the GLE system for a horizontal plate being immersed into (or withdrawn from) 
a liquid bath, with the boundary condition specified at a target film thickness h_end rather 
than a fixed arc length.

PHYSICAL PROBLEM:
- A horizontal plate is immersed into a liquid bath at constant velocity (advancing contact line)
- A thin liquid film forms on the plate surface as it advances into the liquid
- Near the contact line, molecular-scale physics becomes important
- The GLE system captures the transition from molecular to continuum scales

ODE SYSTEM (3 coupled equations):
  dh/ds = sin(θ)                                               [Kinematic condition]
  dθ/ds = ω                                                    [Geometric relation]  
  dω/ds = 3*Ca*f(θ,μᵣ)/(h*(h+3*λ)) + sin(θ)/l_cap²          [Momentum balance + gravity]

Where:
- s: arc length coordinate along the interface (integration variable)
- h(s): film thickness profile
- θ(s): interface angle with respect to the horizontal substrate
- ω(s): interface curvature (dθ/ds)
- Ca: Capillary number (viscous/surface tension forces ratio)
- λ: slip length (molecular scale parameter)
- μᵣ: viscosity ratio (gas/liquid)
- f(θ,μᵣ): viscous dissipation function from wedge flow analysis
- l_cap: capillary length

BOUNDARY CONDITIONS (NEW FORMULATION):
- At s = λ_slip (contact line): h(λ_slip) = λ_slip, θ(λ_slip) = θ₀
- At h = h_end (target thickness): θ(h_end) = θ_end

INTEGRATION DOMAIN:
- Integration proceeds from s = λ_slip to s = s_max (determined iteratively)
- s_max is found such that h(s_max) = h_end and θ(s_max) = θ_end
- This avoids the contact line singularity at s = 0
- The slip length λ_slip provides the molecular cutoff scale

KEY DIFFERENCES FROM v3:
- Boundary condition at target film thickness h_end instead of fixed s_max
- Uses iterative shooting method to find correct integration domain
- More physically meaningful BC for coating applications
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.optimize import brentq
import os
import sys
from functools import partial

# Parameters for horizontal plate immersion
Ca = 0.00972  # Capillary number
mu_r = 1e-3   # \mu_g/\mu_l (viscosity ratio: gas/liquid)

# Length scales for normalization
# NOTE: This implementation uses slip length normalization (lambda_slip = 1)
# All dimensional lengths are normalized by the slip length:
# - h, s are in units of lambda_slip  
# - l_cap is the dimensionless capillary length = l_cap_dimensional/lambda_slip_dimensional
# Alternatively, capillary length normalization can be used by setting l_cap = 1
lambda_slip = 1e0  # Slip length (= 1 for normalization by slip length)
l_cap = 1e6 # Dimensionless capillary length (l_cap_dim/lambda_slip_dim) - controls gravity effects in horizontal plate

# NEW: h-based boundary condition parameters
h_end = 10.0  # Target film thickness where boundary condition is applied
theta_end = 90*np.pi/180  # Angle at h = h_end (default: interface becomes vertical)
s_max_initial = l_cap  # Initial guess for integration domain

N_grid = min(1000000, int(s_max_initial/lambda_slip)) # Number of grid points (adaptive)

# Initial conditions for horizontal plate immersion
h0 = lambda_slip  # h at s = lambda_slip (film thickness at start of domain)
theta0 = 72*np.pi/180  # theta at s = lambda_slip (contact angle at start of domain, measured from horizontal)

# Define f1, f2, and f3 functions
def f1(theta):
    return theta**2 - np.sin(theta)**2

def f2(theta):
    return theta - np.sin(theta) * np.cos(theta)

def f3(theta):
    return theta * (np.pi - theta) + np.sin(theta)**2

# Define f(theta, mu_r) function
def f(theta, mu_r):
    numerator = 2 * np.sin(theta)**3 * (mu_r**2 * f1(theta) + 2 * mu_r * f3(theta) + f1(np.pi - theta))
    denominator = 3 * (mu_r * f1(theta) * f2(np.pi - theta) + f1(np.pi - theta) * f2(theta))
    return numerator / denominator


# Define the coupled ODEs system with state: y = [h, theta, omega]
def GLE(s, y):
    h, theta, omega = y
    dh_ds = np.sin(theta)                                                    # Kinematic condition
    dtheta_ds = omega                                                        # Geometric relation
    domega_ds = 3 * Ca * f(theta, mu_r) / (h * (h + 3 * lambda_slip)) + np.sin(theta)/l_cap**2  # Momentum balance + horizontal gravity term
    return [dh_ds, dtheta_ds, domega_ds]

# Boundary conditions function (same structure as v3, but s_max is now dynamically determined)
def boundary_conditions(ya, yb):
    # ya corresponds to s = lambda_slip (start of domain)
    # yb corresponds to s = s_max (end of domain, where h = h_end)
    h_a, theta_a, omega_a = ya 
    h_b, theta_b, omega_b = yb 
    return [
        h_a - h0,              # h(lambda_slip) = h0 = lambda_slip
        theta_a - theta0,      # theta(lambda_slip) = theta0 (contact angle at start)
        theta_b - theta_end    # theta(s_max) = theta_end (angle at target thickness)
    ]

def solve_gle_for_s_max(s_max):
    """
    Solve the GLE system for a given s_max and return h(s_max) - h_end.
    This function is used by the root finder to determine the correct s_max.
    """
    # Update grid size based on s_max
    N_grid_local = min(1000000, max(1000, int(s_max/lambda_slip)))
    
    # Define the range of s
    s_range_local = np.logspace(np.log10(lambda_slip), np.log10(s_max), N_grid_local)
    
    # Initial guess for the solution
    y_guess_local = np.zeros((3, s_range_local.size))
    y_guess_local[0, :] = np.logspace(np.log10(h0), np.log10(max(h_end, s_max)), s_range_local.size)  # Logarithmic guess for h
    y_guess_local[1, :] = np.linspace(theta0, theta_end, s_range_local.size)  # Linear interpolation for theta
    y_guess_local[2, :] = 0  # Initial guess for omega
    
    # Solve the ODEs
    try:
        solution = solve_bvp(GLE, boundary_conditions, s_range_local, y_guess_local, max_nodes=1000000)
        
        if solution.success:
            # Return h(s_max) - h_end (should be zero when correct s_max is found)
            h_final = solution.y[0, -1]
            return h_final - h_end
        else:
            # If solution failed, return a large penalty
            return 1e6
    except:
        # If solve_bvp throws an exception, return a large penalty
        return 1e6

def find_s_max_for_h_end(h_end_target, theta_end_target, s_min=None, s_max_upper=None, tol=1e-6, max_iter=50):
    """
    Find the correct s_max such that h(s_max) = h_end_target and θ(s_max) = θ_end_target.
    
    Args:
        h_end_target: Target film thickness where BC is applied
        theta_end_target: Target angle at h = h_end
        s_min: Minimum search bound for s_max (default: 2*lambda_slip)
        s_max_upper: Maximum search bound for s_max (default: 10*l_cap)
        tol: Tolerance for convergence
        max_iter: Maximum iterations for root finding
    
    Returns:
        s_max: Integration domain length where h(s_max) = h_end
    """
    global h_end, theta_end  # Update global variables for boundary_conditions
    h_end = h_end_target
    theta_end = theta_end_target
    
    # Set search bounds
    if s_min is None:
        s_min = 2 * lambda_slip
    if s_max_upper is None:
        s_max_upper = 10 * l_cap
    
    print(f"Finding s_max for h_end = {h_end:.3f}, theta_end = {theta_end*180/np.pi:.1f}°")
    
    # Check if bounds bracket the root
    f_min = solve_gle_for_s_max(s_min)
    f_max = solve_gle_for_s_max(s_max_upper)
    
    print(f"  At s_min = {s_min:.2e}: h - h_end = {f_min:.3e}")
    print(f"  At s_max = {s_max_upper:.2e}: h - h_end = {f_max:.3e}")
    
    if f_min * f_max > 0:
        print(f"  Warning: Root may not be bracketed. Expanding search range...")
        if f_min > 0:  # Both positive, need smaller s_max
            s_max_upper = s_min + (s_max_upper - s_min) * 0.1
        else:  # Both negative, need larger s_max
            s_max_upper = s_max_upper * 2
    
    try:
        # Use Brent's method to find the root
        s_max_optimal = brentq(solve_gle_for_s_max, s_min, s_max_upper, xtol=tol, maxiter=max_iter)
        print(f"  Converged: s_max = {s_max_optimal:.6e}")
        return s_max_optimal
    
    except ValueError as e:
        print(f"  Root finding failed: {e}")
        print(f"  Using initial guess: s_max = {s_max_initial}")
        return s_max_initial

def run_solver_and_plot(GUI=False, output_dir='output'):
    """Run the GLE solver for horizontal plate with h-based BC and either display or save plots

    Args:
        GUI (bool): If True, display plots. If False, save to files.
        output_dir (str): Directory to save plots when GUI=False

    Returns:
        tuple: (solution, s_values, h_values, theta_values, w_values)
    """
    # Set matplotlib backend based on GUI parameter
    if not GUI:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    
    # Create output directory if it doesn't exist (always create for CSV)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Starting GLE solver v4 with h-based boundary condition")
    print(f"Target: h_end = {h_end:.3f}, theta_end = {theta_end*180/np.pi:.1f}°")
    
    # Find the correct s_max for the h-based boundary condition
    s_max_optimal = find_s_max_for_h_end(h_end, theta_end)
    
    # Update grid size based on optimal s_max
    N_grid_optimal = min(1000000, max(1000, int(s_max_optimal/lambda_slip)))
    
    # Final solve with the optimal s_max
    s_range_local = np.logspace(np.log10(lambda_slip), np.log10(s_max_optimal), N_grid_optimal)
    y_guess_local = np.zeros((3, s_range_local.size))
    y_guess_local[0, :] = np.logspace(np.log10(h0), np.log10(max(h_end, s_max_optimal)), s_range_local.size)
    y_guess_local[1, :] = np.linspace(theta0, theta_end, s_range_local.size)
    y_guess_local[2, :] = 0
    
    print(f"Final solve with s_max = {s_max_optimal:.6e}, N_grid = {N_grid_optimal}")
    solution = solve_bvp(GLE, boundary_conditions, s_range_local, y_guess_local, max_nodes=1000000)

    # Extract the solution
    s_values_local = solution.x
    h_values_local, theta_values_local, omega_values_local = solution.y
    theta_values_deg = theta_values_local*180/np.pi

    # Verify boundary condition satisfaction
    h_final = h_values_local[-1]
    theta_final = theta_values_local[-1]
    print(f"Final values: h = {h_final:.6f} (target: {h_end:.6f}), theta = {theta_final*180/np.pi:.2f}° (target: {theta_end*180/np.pi:.2f}°)")

    # Convert s to x. dx = cos(theta) ds (horizontal coordinate along plate)
    x_values_local = np.zeros_like(s_values_local)
    x_values_local[1:] = np.cumsum(np.diff(s_values_local) * np.cos(theta_values_local[:-1]))

    # Plot the results with nice styling
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Define color
    solver_color = '#1f77b4'  # Blue
    
    # First create the combined plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 16))
    
    # Plot h(x)
    ax1.plot(x_values_local, h_values_local, '-',
             color=solver_color, linewidth=2.5)
    ax1.set_xlabel('$x(s/l^*)$ ', fontsize=12)
    ax1.set_ylabel('$h(s/l^*)$ ', fontsize=12)
    ax1.set_title('Film Thickness Profile (h-based BC)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, np.max(x_values_local))
    ax1.set_ylim(0, np.max(h_values_local))

    # Add text box with parameters
    textstr = f'Ca = {Ca}\\nλ_slip = {lambda_slip:.0e}\\nμ_r = {mu_r:.0e}\\nh_end = {h_end:.1f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    # Plot theta(s)
    ax2.plot(s_values_local, theta_values_deg, '.',
             color=solver_color, linewidth=2.5)
    ax2.set_xlabel('$s/l^*$', fontsize=12)
    ax2.set_ylabel('$\\theta(s/l^*)$ [degrees]', fontsize=12)
    ax2.set_title('Contact Angle Profile (h-based BC)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(lambda_slip, s_max_optimal)
    ax2.set_ylim(np.min(theta_values_deg), np.max(theta_values_deg))
    # log-log for ax2
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    # Add initial condition text
    ax2.text(0.02, 0.05, f'θ(0) = {theta0*180/np.pi:.0f}°', transform=ax2.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()

    if GUI:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, 'GLE_profiles.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Save data to CSV file
    csv_data = np.column_stack((s_values_local, h_values_local, theta_values_local))
    csv_path = os.path.join(output_dir, 'data-python.csv')
    np.savetxt(csv_path, csv_data, delimiter=',', header='s,h,theta', comments='')
    print(f"Data saved to: {csv_path}")

    return solution, s_values_local, h_values_local, theta_values_local, omega_values_local

# Main execution
if __name__ == "__main__":
    # Check for command line argument
    gui_mode = False  # Default is no GUI
    if len(sys.argv) > 1 and sys.argv[1] == '--gui':
        gui_mode = True

    solution, s_values_final, h_values_final, theta_values_final, omega_values_final = run_solver_and_plot(GUI=gui_mode)

    print(f"Solution converged: {solution.success}")
    print(f"Number of iterations: {solution.niter}")
    
    # Print the final values
    print(f"Final theta: {theta_values_final[-1]*180/np.pi:.2f} degrees")
    print(f"Final curvature omega: {omega_values_final[-1]:.6f}")

    if not gui_mode:
        print("Plot saved to: output/GLE_profiles.png")


# Note: This v4 implementation uses an iterative shooting method to enforce boundary 
# conditions at a target film thickness h_end rather than a fixed arc length s_max.
# This is more physically meaningful for coating applications where the film thickness
# is the controlled parameter.