import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from functools import partial
import pandas as pd

#Parameters
Ca = 1e-2  # Capillary number
lambda_slip = 1e-7  # Slip length
mu_r = 1e-3 # \mu_g/\mu_l

# Initial conditions
h0 = lambda_slip  # h at s = 0
slope0 = 1  # theta at s = 0
omega_bc = 0  # curvature boundary condition at s = \Delta, this needs to be not remain constant, but fed back from the DNS

# Define the coupled ODEs system
def GLE(x, y):
    h, alpha, omega = y
    dh_dx = alpha
    dalpha_dx = omega # omega = dtheta/ds
    dw_dx = - Ca/(h*(h+1))
    return [dh_dx, dalpha_dx, dw_dx]

# Set up the solver parameters
# Need to set the initial conditions for the ODEs. Since we are setting them at different points, we need 3 as fixed, 3 as guesses
# \Theta at s=0, h at s=0, dTheta/ds at h=\Delta
# The guesses follow the known BCs when solved
# The 3rd "known" BC is the curvature at h=\Delta, which is not known, but can be fed back from the DNS

# Boundary conditions
def boundary_conditions(ya, yb, omega_bc):
    # ya corresponds to s = 0, yb corresponds to s = 4*Delta
    h_a, alpha_a, omega_a = ya # boundary conditions at s = 0
    h_b, alpha_b, omega_b = yb # boundary conditions at s = Delta
    return [
        alpha_a - slope0,
        h_a - h0,      # h(0) = lambda_slip, this forces h_a to be essentially lambda_slip. We set.
        omega_b - omega_bc         # w(Delta) = w_bc (curvature at s=Delta), this forces w_b (curvature at s=Delta) to be essentially w_bc, comes from the DNS.
    ]

# def run_solver_and_plot(GUI=False, output_dir='output'):
    """Run the solver and either display or save plots

    Args:
        GUI (bool): If True, display plots. If False, save to files.
        output_dir (str): Directory to save plots when GUI=False

    Returns:
        tuple: (solution, s_values, h_values, theta_values, w_values)
    """
    # # Set matplotlib backend based on GUI parameter
    # if not GUI:
    #     import matplotlib
    #     matplotlib.use('Agg')  # Use non-interactive backend
    
    # # Create output directory if it doesn't exist (always create for CSV)
    # os.makedirs(output_dir, exist_ok=True)

    # Initial guess for the solution
s_max = 10 # adjust as needed for your problem
x_range_local = np.linspace(0, s_max, 100000)  # Define the range of s with finite values
y_guess_local = np.zeros((3, x_range_local.size))  # Initial guess for [h, theta, w]
y_guess_local[0, :] = np.linspace(h0, s_max, x_range_local.size)  # Linear guess for h
y_guess_local[1, :] = slope0  # Initial guess for theta
y_guess_local[2, :] = 0          # Initial guess for dTheta/ds

    # Solve the ODEs
    # Use partial to pass omega_bc as a parameter to boundary_conditions
bc_with_w = partial(boundary_conditions, omega_bc=omega_bc)
solution = solve_bvp(GLE, bc_with_w, x_range_local, y_guess_local, max_nodes=1000000)

    # Extract the solution
x_values_local = solution.x
h_values_local, alpha_values_local, omega_values_local = solution.y

try:
        data_path = 'Minkush data.csv'
        df = pd.read_csv(data_path, delimiter=', ', engine='python')
        x_csv = df.iloc[:, 0].values
        h_csv = df.iloc[:, 1].values
except Exception as e:
        print(f"Failed to load CSV: {e}")
        x_csv = h_csv = None    
    
    # Plot h(s)
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x_values_local, h_values_local, '-')
if x_csv is not None and h_csv is not None:
    ax.plot(x_csv, h_csv, 'o', label='Minkush Data', markersize=4)

    # Only plot h(s) vs s (no subplots)

ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('h', fontsize=12)
ax.set_xlim(0, s_max)
ax.set_ylim(0, s_max)
plt.grid(True, alpha=0.3)

plt.show()

if solution.success:
    print("Solver converged successfully.")
else:
    print("Solver did not converge. Message:", solution.message)
#     # Add text box with parameters
#     textstr = f'Ca = {Ca}\nλ_slip = {lambda_slip:.0e}\nμ_r = {mu_r:.0e}'
#     props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#     ax.text(0.02, 0.95, textstr, transform=ax.transAxes, fontsize=10,
#             verticalalignment='top', bbox=props)
#     ax1.plot(s_values_local * 1e6, h_values_local * 1e6, '-', 
#              color=solver_color, linewidth=2.5)
#     ax1.set_xlabel('s [μm]', fontsize=12)
#     ax1.set_ylabel('h(s) [μm]', fontsize=12)
#     ax1.set_title('Film Thickness Profile', fontsize=14, fontweight='bold')
#     ax1.grid(True, alpha=0.3)
#     ax1.set_xlim(0, 4)
    
#     # Add text box with parameters
#     textstr = f'Ca = {Ca}\nλ_slip = {lambda_slip:.0e}\nμ_r = {mu_r:.0e}'
#     props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
#     ax1.text(0.02, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
#              verticalalignment='top', bbox=props)
    
#     # Plot theta(s)
#     ax2.plot(s_values_local * 1e6, theta_values_deg, '-', 
#              color=solver_color, linewidth=2.5)
#     ax2.set_xlabel('s [μm]', fontsize=12)
#     ax2.set_ylabel('θ(s) [degrees]', fontsize=12)
#     ax2.set_title('Contact Angle Profile', fontsize=14, fontweight='bold')
#     ax2.grid(True, alpha=0.3)
#     ax2.set_xlim(0, 4)
    
#     # Add initial condition text
#     ax2.text(0.02, 0.05, f'θ(0) = {np.pi/4 * 180/np.pi:.0f}°', transform=ax2.transAxes, fontsize=10,
#              bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
#     plt.tight_layout()
    
#     if GUI:
#         plt.show()
#     else:
#         plt.savefig(os.path.join(output_dir, 'GLE_profiles.png'), dpi=300, bbox_inches='tight')
#         plt.close()

#     # Save data to CSV file
#     csv_data = np.column_stack((s_values_local, h_values_local, theta_values_local))
#     csv_path = os.path.join(output_dir, 'data-python.csv')
#     np.savetxt(csv_path, csv_data, delimiter=',', header='s,h,theta', comments='')
#     print(f"Data saved to: {csv_path}")

#     return solution, s_values_local, h_values_local, theta_values_local, w_values_local

# Main execution
# if __name__ == "__main__":
#     # Check for command line argument
#     gui_mode = False  # Default is no GUI
#     if len(sys.argv) > 1 and sys.argv[1] == '--gui':
#         gui_mode = True

#     solution, s_values_final, h_values_final, theta_values_final, w_values_final = run_solver_and_plot(GUI=gui_mode)

#     print(f"Solution converged: {solution.success}")
#     print(f"Number of iterations: {solution.niter}")

#     if not gui_mode:
#         print("Plot saved to: output/GLE_profiles.png")


# # Note: difference between this code and the ones from our [coalleauges](https://doi.org/10.1140/epjs/s11734-024-01443-5) is that we are solving for a specific control parameter whereas they use continuation method to track solution branches as parameters vary.
