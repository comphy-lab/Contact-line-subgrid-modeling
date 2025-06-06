import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import os
import sys
from functools import partial

# Parameters

Delta = 1e0  # Minimum dimensionless grid cell size (for the DNS) and the maximum s-value for the solver.
Ca = 0.0246  # Capillary number
lambda_slip = 1e-4  # Slip length
mu_r = 1e-6 # \mu_g/\mu_l

# Boundary conditions
theta0 = np.pi/2  # theta at s = 0
w = 0  # curvature boundary condition at s = \Delta, this will be fed back from the DNS!
h0 = lambda_slip  # h at s = 0

# Define f1, f2, and f3 functions needed for the GLE
def f1(theta):
    return theta**2 - np.sin(theta)**2

def f2(theta):
    return theta - np.sin(theta) * np.cos(theta)

def f3(theta):
    return theta * (np.pi - theta) + np.sin(theta)**2

# Define f(theta, mu_r) function
def f(theta, mu_r):
    numerator = 2 * np.sin(theta)**3 * (mu_r**2 * f1(theta) + 2 * mu_r * f3(theta) + f1(np.pi - theta))
    denominator = 3 * (mu_r * f1(theta) * f2(np.pi - theta) - f1(np.pi - theta) * f2(theta))
    return numerator / denominator

# Define the coupled ODEs system
def GLE(s, y):
    h, theta, omega = y
    dh_ds = np.sin(theta) # dh/ds = sin(theta)
    dt_ds = omega # omega = dtheta/ds
    dw_ds = - 3 * Ca * f(theta, mu_r) / (h * (h + 3 * lambda_slip)) - np.cos(theta)
    return [dh_ds, dt_ds, dw_ds]


""" Note:
Set up the solver parameters:
- Need to set the boundary conditions for the ODEs. Since we are setting them at different points, we need 3 as fixed, 3 as guesses
- $\\Theta$ at $s=0$, $h$ at $s=0$, $d\\Theta/ds$ at $s=\\Delta$
- The guesses follow the known BCs when solved
- The 3rd "known" BC is the curvature at $s=\\Delta$, which is not known, but can be fed back from the DNS
"""

# Boundary conditions
def boundary_conditions(ya, yb, w_bc):
    # ya corresponds to s = 0, yb corresponds to s = 4*Delta
    h_a, theta_a, w_a = ya # boundary conditions at s = 0
    h_b, theta_b, w_b = yb # boundary conditions at s = Delta
    return [
        theta_a - theta0,      # theta(0), this forces theta_a to be essentially theta0. We set.
        h_a - lambda_slip,      # h(0) = lambda_slip, this forces h_a to be essentially lambda_slip. We set.
        w_b - w_bc         # w(Delta) = w_bc (curvature at s=Delta), this forces w_b (curvature at s=Delta) to be essentially w_bc, comes from the DNS.
    ]

def run_solver_and_plot(GUI=False, output_dir='output'):
    """Run the solver and either display or save plots

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
    s_range_local = np.linspace(0, Delta, 1000)  # Define the range of s
    y_guess_local = np.zeros((3, s_range_local.size))  # Initial guess for [theta, w, h]
    y_guess_local[0, :] = theta0  # Initial guess for theta
    y_guess_local[1, :] = np.linspace(lambda_slip, Delta, s_range_local.size)  # Linear guess for h
    y_guess_local[2, :] = 0          # Initial guess for dTheta/ds

    # Solve the ODEs
    # Use partial to pass w as a parameter to boundary_conditions
    bc_with_w = partial(boundary_conditions, w_bc=w)
    solution = solve_bvp(GLE, bc_with_w, s_range_local, y_guess_local, max_nodes=1000000)

    # Extract the solution
    s_values_local = solution.x
    h_values_local, theta_values_local, w_values_local = solution.y
    theta_values_deg = theta_values_local*180/np.pi
    
    # Calculate x(s) = integral of cos(theta) ds
    x_values_local = np.zeros_like(s_values_local)
    x_values_local[1:] = np.cumsum(np.diff(s_values_local) * np.cos(theta_values_local[:-1]))

    # Plot the results with nice styling
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Define color
    solver_color = '#1f77b4'  # Blue
    
    # Create 2x2 subplot grid
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: h(s) vs s
    ax1.plot(s_values_local, h_values_local, '-', 
             color=solver_color, linewidth=2.5)
    ax1.set_xlabel('s', fontsize=12)
    ax1.set_ylabel('h(s)', fontsize=12)
    ax1.set_title('Film Thickness Profile', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, Delta)
    ax1.set_ylim(lambda_slip, 1.01*max(h_values_local))
    
    # Add text box with parameters
    textstr = f'Ca = {Ca}\n$\\lambda_\\text{{slip}}$ = {lambda_slip:.0e}\n$\\mu_r$ = {mu_r:.0e}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Plot 2: theta(s) vs s
    ax2.plot(s_values_local, theta_values_deg, '-', 
             color=solver_color, linewidth=2.5)
    ax2.set_xlabel('s', fontsize=12)
    ax2.set_ylabel('$\\theta(s)$ [degrees]', fontsize=12)
    ax2.set_title('Contact Angle Profile', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, Delta)
    ax2.set_ylim(0.99*min(theta_values_deg), 1.01*max(theta_values_deg))
    
    # Add initial condition text
    ax2.text(0.02, 0.05, f'θ(0) = {theta0*180/np.pi:.0f}°', transform=ax2.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Plot 3: theta(s) vs h(s)
    ax3.plot(h_values_local, theta_values_deg, '-', 
             color=solver_color, linewidth=2.5)
    ax3.set_xlabel('h(s)', fontsize=12)
    ax3.set_ylabel('$\\theta(s)$ [degrees]', fontsize=12)
    ax3.set_title('Contact Angle vs Film Thickness', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(lambda_slip, 1.01*max(h_values_local))
    ax3.set_ylim(0.99*min(theta_values_deg), 1.01*max(theta_values_deg))
    
    # Plot 4: h(s) vs x(s)
    ax4.plot(x_values_local, h_values_local, '-', 
             color=solver_color, linewidth=2.5)
    ax4.set_xlabel('x(s)', fontsize=12)
    ax4.set_ylabel('h(s)', fontsize=12)
    ax4.set_title('Film Thickness vs Horizontal Position', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1.01*max(x_values_local))
    ax4.set_ylim(lambda_slip, 1.01*max(h_values_local))
    
    plt.tight_layout()
    
    if GUI:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, 'GLE_profiles.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Save data to CSV file
    csv_data = np.column_stack((s_values_local, h_values_local, theta_values_local, w_values_local, x_values_local))
    csv_path = os.path.join(output_dir, 'data-python.csv')
    np.savetxt(csv_path, csv_data, delimiter=',', header='s,h,theta,w,x', comments='')
    print(f"Data saved to: {csv_path}")

    return solution, s_values_local, h_values_local, theta_values_local, w_values_local

# Main execution
if __name__ == "__main__":
    # Check for command line argument
    gui_mode = False  # Default is no GUI
    if len(sys.argv) > 1 and sys.argv[1] == '--gui':
        gui_mode = True

    solution, s_values_final, h_values_final, theta_values_final, w_values_final = run_solver_and_plot(GUI=gui_mode)

    print(f"Solution converged: {solution.success}")
    print(f"Number of iterations: {solution.niter}")

    if not gui_mode:
        print("Plot saved to: output/GLE_profiles.png")


# Note: difference between this code and the ones from our [coalleauges](https://doi.org/10.1140/epjs/s11734-024-01443-5) is that we are solving for a specific control parameter whereas they use continuation method to track solution branches as parameters vary.
