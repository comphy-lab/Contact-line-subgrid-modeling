import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import os
import sys
from functools import partial

#Parameters
Ca = 0.0246  # Capillary number
lambda_slip = 1e-4  # Slip length
mu_r = 1e-3 # \mu_g/\mu_l

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

# Initial conditions
h0 = lambda_slip  # h at s = 0
theta0 = np.pi/2  # theta at s = 0
w = 0  # curvature boundary condition at s = \Delta, this needs to be not remain constant, but fed back from the DNS



# Define the coupled ODEs system
def GLE(s, y):
    h, theta, omega = y
    dh_ds = np.sin(theta) # dh/ds = sin(theta)
    dt_ds = omega # omega = dtheta/ds
    dw_ds = 3 * Ca * f(theta, mu_r) / (h * (h + 3 * lambda_slip)) - np.cos(theta)
    return [dh_ds, dt_ds, dw_ds]

# Set up the solver parameters
# Need to set the initial conditions for the ODEs. Since we are setting them at different points, we need 3 as fixed, 3 as guesses
# \Theta at s=0, h at s=0, dTheta/ds at h=\Delta
# The guesses follow the known BCs when solved
# The 3rd "known" BC is the curvature at h=\Delta, which is not known, but can be fed back from the DNS

Delta = 1e-4  # Miminum grid cell size

# Boundary conditions
def boundary_conditions(ya, yb, w_bc):
    # ya corresponds to s = 0, yb corresponds to s = 4*Delta
    h_a, theta_a, w_a = ya # boundary conditions at s = 0
    h_b, theta_b, w_b = yb # boundary conditions at s = Delta
    return [
        theta_a - theta0,      # theta(0) = pi/6, this forces theta_a to be essentially theta0. We set.
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
    s_range_local = np.linspace(0, 10, 1000)  # Define the range of s
    y_guess_local = np.zeros((3, s_range_local.size))  # Initial guess for [theta, w, h]
    y_guess_local[0, :] = np.linspace(lambda_slip, 10, s_range_local.size)  # Linear guess for h
    y_guess_local[1, :] = np.pi / 2  # Initial guess for theta
    y_guess_local[2, :] = 0          # Initial guess for dTheta/ds

    # Solve the ODEs
    # Use partial to pass w as a parameter to boundary_conditions
    bc_with_w = partial(boundary_conditions, w_bc=w)
    solution = solve_bvp(GLE, bc_with_w, s_range_local, y_guess_local, max_nodes=1000000)

    # Extract the solution
    s_values_local = solution.x
    h_values_local, theta_values_local, w_values_local = solution.y
    theta_values_deg = theta_values_local*180/np.pi

    x_values_local = np.zeros_like(s_values_local)
    x_values_local[1:] = np.cumsum(np.diff(s_values_local) * np.cos(theta_values_local[:-1]))

    # Plot the results with nice styling
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Define color
    solver_color = '#1f77b4'  # Blue
    
    # First create the combined plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot h(s)
    ax1.plot(s_values_local, h_values_local, '-', 
             color=solver_color, linewidth=2.5)
    ax1.set_xlabel('x ', fontsize=12)
    ax1.set_ylabel('h(s) ', fontsize=12)
    ax1.set_title('Film Thickness Profile', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 10)
    
    # Add text box with parameters
    textstr = f'Ca = {Ca}\nλ_slip = {lambda_slip:.0e}\nμ_r = {mu_r:.0e}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Plot theta(s)
    ax2.plot(s_values_local, theta_values_deg, '-', 
             color=solver_color, linewidth=2.5)
    ax2.set_xlabel('s [μm]', fontsize=12)
    ax2.set_ylabel('θ(s) [degrees]', fontsize=12)
    ax2.set_title('Contact Angle Profile', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 10)
    
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
