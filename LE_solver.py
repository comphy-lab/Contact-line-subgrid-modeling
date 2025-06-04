import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import os
import sys
from functools import partial

#Parameters
Ca = 1e-2  # Capillary number
lambda_slip = 1e-7  # Slip length
mu_r = 1e-3 # \mu_g/\mu_l

# Initial conditions
h0 = lambda_slip  # h at s = 0
slope0 = 1  # initial slope (dh/dx) at x = 0
omega_bc = 0  # curvature boundary condition at s = Delta, this needs to be not remain constant, but fed back from the DNS

# Define the coupled ODEs system for Lubrication Equation
def LE(x, y):
    h, alpha, omega = y
    dh_dx = alpha
    dalpha_dx = omega # omega = d²h/dx²
    dw_dx = - Ca/(h*(h+1))
    return [dh_dx, dalpha_dx, dw_dx]

# Set up the solver parameters
# Need to set the initial conditions for the ODEs. Since we are setting them at different points, we need 3 as fixed, 3 as guesses
# slope at x=0, h at x=0, curvature at x=Delta
# The guesses follow the known BCs when solved
# The 3rd "known" BC is the curvature at x=Delta, which is not known, but can be fed back from the DNS

Delta = 10  # Domain size, adjust as needed

# Boundary conditions
def boundary_conditions(ya, yb, omega_bc):
    # ya corresponds to x = 0, yb corresponds to x = Delta
    h_a, alpha_a, omega_a = ya # boundary conditions at x = 0
    h_b, alpha_b, omega_b = yb # boundary conditions at x = Delta
    return [
        alpha_a - slope0,      # dh/dx(0) = slope0
        h_a - h0,             # h(0) = lambda_slip
        omega_b - omega_bc    # d²h/dx²(Delta) = omega_bc (curvature at x=Delta)
    ]

def run_solver_and_plot(GUI=False, output_dir='output'):
    """Run the solver and either display or save plots

    Args:
        GUI (bool): If True, display plots. If False, save to files.
        output_dir (str): Directory to save plots when GUI=False

    Returns:
        tuple: (solution, x_values, h_values, alpha_values, omega_values)
    """
    # Set matplotlib backend based on GUI parameter
    if not GUI:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
    
    # Create output directory if it doesn't exist (always create for CSV)
    os.makedirs(output_dir, exist_ok=True)

    # Initial guess for the solution
    x_range_local = np.linspace(0, Delta, 100000)  # Define the range of x
    y_guess_local = np.zeros((3, x_range_local.size))  # Initial guess for [h, alpha, omega]
    y_guess_local[0, :] = np.linspace(h0, Delta, x_range_local.size)  # Linear guess for h
    y_guess_local[1, :] = slope0  # Initial guess for dh/dx
    y_guess_local[2, :] = 0      # Initial guess for d²h/dx²

    # Solve the ODEs
    # Use partial to pass omega_bc as a parameter to boundary_conditions
    bc_with_omega = partial(boundary_conditions, omega_bc=omega_bc)
    solution = solve_bvp(LE, bc_with_omega, x_range_local, y_guess_local, max_nodes=1000000)

    # Extract the solution
    x_values_local = solution.x
    h_values_local, alpha_values_local, omega_values_local = solution.y

    # Plot the results with nice styling
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Define color
    solver_color = '#1f77b4'  # Blue
    
    # First create the combined plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot h(x)
    ax1.plot(x_values_local, h_values_local, '-', 
             color=solver_color, linewidth=2.5)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('h(x)', fontsize=12)
    ax1.set_title('Film Thickness Profile (Lubrication Equation)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, Delta)
    
    # Add text box with parameters
    textstr = f'Ca = {Ca}\\nλ_slip = {lambda_slip:.0e}\\nμ_r = {mu_r:.0e}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.02, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    # Plot dh/dx
    ax2.plot(x_values_local, alpha_values_local, '-', 
             color=solver_color, linewidth=2.5)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('dh/dx', fontsize=12)
    ax2.set_title('Film Thickness Gradient', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, Delta)
    
    # Add initial condition text
    ax2.text(0.02, 0.05, f'dh/dx(0) = {slope0}', transform=ax2.transAxes, fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    
    if GUI:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, 'LE_profiles.png'), dpi=300, bbox_inches='tight')
        plt.close()

    # Save data to CSV file
    csv_data = np.column_stack((x_values_local, h_values_local, alpha_values_local, omega_values_local))
    csv_path = os.path.join(output_dir, 'LE_data-python.csv')
    np.savetxt(csv_path, csv_data, delimiter=',', header='x,h,dh_dx,d2h_dx2', comments='')
    print(f"Data saved to: {csv_path}")

    return solution, x_values_local, h_values_local, alpha_values_local, omega_values_local

# Main execution
if __name__ == "__main__":
    # Check for command line argument
    gui_mode = False  # Default is no GUI
    if len(sys.argv) > 1 and sys.argv[1] == '--gui':
        gui_mode = True

    solution, x_values_final, h_values_final, alpha_values_final, omega_values_final = run_solver_and_plot(GUI=gui_mode)

    print(f"Solution converged: {solution.success}")
    print(f"Number of iterations: {solution.niter}")
    
    if solution.success:
        print("Solver converged successfully.")
    else:
        print("Solver did not converge. Message:", solution.message)

    if not gui_mode:
        print("Plot saved to: output/LE_profiles.png")