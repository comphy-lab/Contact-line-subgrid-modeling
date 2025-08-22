"""
Generalized Lubrication Equations (GLE) Solver for Plate Withdrawal Problem

This code solves the GLE system for a vertical plate withdrawing from a liquid bath.
The vertical plate geometry is why the gravitational term appears as np.cos(theta) in the
momentum equation (line 51) - gravity acts vertically downward.

PHYSICAL PROBLEM:
- A vertical plate is withdrawn from a liquid bath at constant velocity
- A thin liquid film is entrained on the plate surface
- Near the contact line, molecular-scale physics becomes important
- The GLE system captures the transition from molecular to continuum scales

ODE SYSTEM (3 coupled equations):
  dh/ds = sin(θ)                                               [Kinematic condition]
  dθ/ds = ω                                                    [Geometric relation]  
  dω/ds = 3*Ca*f(θ,μᵣ)/(h*(h+3*λ)) - cos(θ)/l_cap²          [Momentum balance + gravity]

Where:
- s: arc length coordinate along the interface (integration variable)
- h(s): film thickness profile
- θ(s): interface angle with respect to the substrate
- ω(s): interface curvature (dθ/ds)
- Ca: Capillary number (viscous/surface tension forces ratio)
- λ: slip length (molecular scale parameter)
- μᵣ: viscosity ratio (gas/liquid)
- f(θ,μᵣ): viscous dissipation function from wedge flow analysis
- l_cap: capillary length

BOUNDARY CONDITIONS:
- At s = λ_slip (contact line): h(λ_slip) = λ_slip, θ(λ_slip) = θ₀
- At s = s_max (far field): ω(s_max) = 0 (flat far-field assumption)

INTEGRATION DOMAIN:
- Integration proceeds from s = λ_slip to s = s_max
- This avoids the contact line singularity at s = 0
- The slip length λ_slip provides the molecular cutoff scale
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import os
import sys
from functools import partial

# Parameters for vertical plate withdrawal
Ca = 0.005  # Capillary number
mu_r = 1e-3 # \mu_g/\mu_l (viscosity ratio: gas/liquid)

# Length scales for normalization
# NOTE: This implementation uses slip length normalization (lambda_slip = 1)
# All dimensional lengths are normalized by the slip length:
# - h, s are in units of lambda_slip  
# - l_cap is the dimensionless capillary length = l_cap_dimensional/lambda_slip_dimensional
# Alternatively, capillary length normalization can be used by setting l_cap = 1
lambda_slip = 1e0  # Slip length (= 1 for normalization by slip length)
l_cap = 1e6 # Dimensionless capillary length (l_cap_dim/lambda_slip_dim) - controls gravity effects in vertical plate
s_max = l_cap # maximum s/l* -> for receeding cases, to capture the dip, we need to go beyond the capillary length

N_grid = min(1000000, int(s_max/lambda_slip)) # Number of grid points (s_max/lambda_slip in dimensionless units)

# Initial conditions  
h0 = lambda_slip  # h at s = lambda_slip (film thickness at start of domain)
theta0 = np.pi/3  # theta at s = lambda_slip (contact angle at start of domain)
omega_at_smax = 0  # omega at s = s_max (curvature at far field, typically 0 for flat far-field)


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
    domega_ds = 3 * Ca * f(theta, mu_r) / (h * (h + 3 * lambda_slip)) - np.cos(theta)/l_cap**2  # Momentum balance + vertical gravity term
    return [dh_ds, dtheta_ds, domega_ds]

# Set up the solver parameters
# Boundary value problem with 3 boundary conditions (all fixed):
# 1. h(s=lambda_slip) = h0 = lambda_slip  [film thickness at contact line]
# 2. theta(s=lambda_slip) = theta0        [contact angle at contact line]  
# 3. omega(s=s_max) = omega_at_smax       [curvature at far field, typically 0]
# The solver iteratively adjusts the solution to satisfy all three BCs simultaneously

def boundary_conditions(ya, yb, omega_bc):
    # ya corresponds to s = lambda_slip (start of domain)
    # yb corresponds to s = s_max (end of domain)
    h_a, theta_a, omega_a = ya 
    h_b, theta_b, omega_b = yb 
    return [
        h_a - h0,              # h(lambda_slip) = h0 = lambda_slip
        theta_a - theta0,      # theta(lambda_slip) = theta0
        omega_b - omega_bc     # omega(s_max) = omega_bc (from DNS or set to 0)
    ]

def run_solver_and_plot(GUI=False, output_dir='output'):
    """Run the GLE solver for vertical plate withdrawal and either display or save plots

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

    # Initial guess for the solution
    s_range_local = np.logspace(np.log10(lambda_slip), np.log10(s_max), N_grid)  # Define the range of s
    y_guess_local = np.zeros((3, s_range_local.size))  # Initial guess for [theta, w, h]
    y_guess_local[0, :] = np.logspace(np.log10(h0), np.log10(s_max), s_range_local.size)  # Logarithmic guess for h from h0 to s_max
    y_guess_local[1, :] = theta0  # Initial guess for theta
    y_guess_local[2, :] = 0          # Initial guess for omega

    # Solve the ODEs
    # Use partial to pass omega_bc as a parameter to boundary_conditions
    bc_with_omega_bc = partial(boundary_conditions, omega_bc=omega_at_smax)
    solution = solve_bvp(GLE, bc_with_omega_bc, s_range_local, y_guess_local, max_nodes=1000000)

    # Extract the solution
    s_values_local = solution.x
    h_values_local, theta_values_local, omega_values_local = solution.y
    theta_values_deg = theta_values_local*180/np.pi

    # Convert s to x. dx = cos(theta) ds
    x_values_local = np.zeros_like(s_values_local)
    theta_mid = (theta_values_local[:-1] + theta_values_local[1:]) / 2
    x_values_local[1:] = np.cumsum(np.diff(s_values_local) * np.cos(theta_mid))

    # Plot the results with nice styling
    plt.style.use('seaborn-v0_8-darkgrid')

    # Define color
    solver_color = '#1f77b4'  # Blue

    # First create the combined plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Plot h(x)
    ax1.plot(x_values_local, h_values_local, '-',
             color=solver_color, linewidth=2.5)
    ax1.set_xlabel('$x(s/l^*)$ ', fontsize=12)
    ax1.set_ylabel('$h(s/l^*)$ ', fontsize=12)
    ax1.set_title('Film Thickness Profile', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, np.max(x_values_local))
    ax1.set_ylim(np.min(h_values_local), np.max(h_values_local))

    # Add text box with parameters
    textstr = f'Ca = {Ca}\nλ_slip = {lambda_slip:.0e}\nμ_r = {mu_r:.0e}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)

    # Plot theta(s)
    ax2.plot(s_values_local, theta_values_deg, '.',
             color=solver_color, linewidth=2.5)
    ax2.set_xlabel('$s/l^*$', fontsize=12)
    ax2.set_ylabel('$\\theta(s/l^*)$ [degrees]', fontsize=12)
    ax2.set_title('Contact Angle Profile', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(lambda_slip, s_max)
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

    if not gui_mode:
        print("Plot saved to: output/GLE_profiles.png")


# Note: difference between this code and the ones from our [coalleauges](https://doi.org/10.1140/epjs/s11734-024-01443-5) is that we are solving for a specific control parameter whereas they use continuation method to track solution branches as parameters vary.
