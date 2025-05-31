import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import os
import sys

#Parameters
Ca = 1.0  # Capillary number
lambda_slip = 1e-5  # Slip length
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
    denominator = 3 * (mu_r * f1(theta) * f2(np.pi - theta) - f1(np.pi - theta) * f2(theta))
    return numerator / denominator

# Initial conditions
h0 = lambda_slip  # h at s = 0
theta0 = np.pi/6  # theta at s = 0
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
def boundary_conditions(ya, yb):
    # ya corresponds to s = 0, yb corresponds to s = 4*Delta
    h_a, theta_a, w_a = ya # boundary conditions at s = 0
    h_b, theta_b, w_b = yb # boundary conditions at s = Delta
    return [
        theta_a - theta0,      # theta(0) = pi/6, this forces theta_a to be essentially theta0. We set. 
        h_a - lambda_slip,      # h(0) = lambda_slip, this forces h_a to be essentially lambda_slip. We set.
        w_b - w         # w(Delta) = w (curvature at s=Delta), this forces w_b (curvature at s=Delta) to be essentially w, comes from the DNS.
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
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    # Initial guess for the solution
    s_range_local = np.linspace(0, 4*Delta, 10000)  # Define the range of s
    y_guess_local = np.zeros((3, s_range_local.size))  # Initial guess for [theta, w, h]
    y_guess_local[0, :] = np.linspace(lambda_slip, Delta, s_range_local.size)  # Linear guess for h
    y_guess_local[1, :] = np.pi / 6  # Initial guess for theta
    y_guess_local[2, :] = 0          # Initial guess for dTheta/ds

    # Solve the ODEs
    solution = solve_bvp(GLE, boundary_conditions, s_range_local, y_guess_local)

    # Extract the solution
    s_values_local = solution.x
    h_values_local, theta_values_local, w_values_local = solution.y
    theta_values_deg = theta_values_local*180/np.pi

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(s_values_local, h_values_local, label=r'$h(s)$')
    plt.ylabel(r'$h(s)$')
    plt.xlabel('s')
    plt.grid()
    if GUI:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, 'GLE_h_profile.png'), dpi=150, bbox_inches='tight')
        plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(s_values_local, theta_values_deg, label=r'$\theta(s)$')
    plt.xlabel('s')
    plt.ylabel(r'$\theta(s)$ (degrees)')
    plt.grid()
    if GUI:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, 'GLE_theta_profile.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
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
        print("Plots saved in 'output' directory")


# Note: difference between this code and the ones from our [coalleauges](https://doi.org/10.1140/epjs/s11734-024-01443-5) is that we are solving for a specific control parameter whereas they use continuation method to track solution branches as parameters vary.