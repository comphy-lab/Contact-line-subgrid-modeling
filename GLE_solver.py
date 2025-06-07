import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
import os
import sys
from functools import partial
import argparse

# Default parameters (can be overridden via command line)
DEFAULT_DELTA = 1e0  # Minimum dimensionless grid cell size (for the DNS) and the maximum s-value for the solver.
DEFAULT_CA = 0.0246  # Capillary number
DEFAULT_LAMBDA_SLIP = 1e-4  # Slip length
DEFAULT_MU_R = 1e-6  # \mu_g/\mu_l
DEFAULT_THETA0 = np.pi/2  # theta at s = 0
DEFAULT_W = 0  # curvature boundary condition at s = \Delta
DEFAULT_NGRID_INIT = 10000  # Initial number of grid points

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
    # Add small epsilon to prevent division by zero
    epsilon = 1e-10
    denominator = np.where(np.abs(denominator) < epsilon, 
                          np.sign(denominator) * epsilon + (denominator == 0) * epsilon, 
                          denominator)
    return numerator / denominator

# Define the coupled ODEs system
def GLE(s, y, Ca, mu_r, lambda_slip):
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
def boundary_conditions(ya, yb, w_bc, theta0, lambda_slip):
    # ya corresponds to s = 0, yb corresponds to s = Delta
    h_a, theta_a, w_a = ya # boundary conditions at s = 0
    h_b, theta_b, w_b = yb # boundary conditions at s = Delta
    return [
        theta_a - theta0,      # theta(0), this forces theta_a to be essentially theta0. We set.
        h_a - lambda_slip,      # h(0) = lambda_slip, this forces h_a to be essentially lambda_slip. We set.
        w_b - w_bc         # w(Delta) = w_bc (curvature at s=Delta), this forces w_b (curvature at s=Delta) to be essentially w_bc, comes from the DNS.
    ]

def plot_x0_vs_ca(Ca_values, x0_values, output_dir='output'):
    """Plot x0 (position of minimum theta) vs Ca"""
    plt.figure(figsize=(10, 8))
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Create scatter plot
    plt.scatter(Ca_values, x0_values, s=60, alpha=0.7, color='darkblue', edgecolors='black', linewidth=1)
    
    # Add a trend line
    plt.plot(Ca_values, x0_values, '-', color='crimson', linewidth=2, alpha=0.8)
    
    plt.xlabel('Ca (Capillary Number)', fontsize=14)
    plt.ylabel('$x_0$ (Position at $\\theta_{min}$)', fontsize=14)
    plt.title('Position of Minimum Contact Angle vs Capillary Number', fontsize=16, fontweight='bold')
    
    # Use log scale for Ca if range is large
    if max(Ca_values) / min(Ca_values) > 100:
        plt.xscale('log')
        plt.xlabel('Ca (Capillary Number) [log scale]', fontsize=14)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'x0_vs_Ca.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"x0 vs Ca plot saved to: {plot_path}")
    
    # Also save the data to CSV
    csv_data = np.column_stack((Ca_values, x0_values))
    csv_path = os.path.join(output_dir, 'x0_vs_Ca_data.csv')
    np.savetxt(csv_path, csv_data, delimiter=',', header='Ca,x0', comments='')
    print(f"x0 vs Ca data saved to: {csv_path}")

def find_critical_ca_continuation(Ca_target, mu_r, lambda_slip, theta0, w_bc, Delta, s_range_initial, y_guess_initial, output_dir='output'):
    """
    Find the critical Ca using continuation method
    
    Returns:
        solution: The BVP solution at the largest successful Ca
        Ca_critical: The largest Ca for which a solution exists
    """
    print("Direct solve failed. Using continuation method to find critical Ca...")
    # Use continuation on Ca to find the maximum Ca for which a solution exists
    # Start with small Ca and increase logarithmically
    Ca_steps = np.logspace(-4, np.log10(max(1.0, Ca_target*2)), 50)  # Logarithmic spacing
    current_guess = y_guess_initial
    s_range_local = s_range_initial
    Ca_critical = 0
    last_good_solution = None
    
    # Lists to store x0 vs Ca data
    Ca_values = []
    x0_values = []
    
    for Ca_step in Ca_steps:
        if Ca_step > Ca_target:
            # If we've exceeded target Ca and have a solution, use it
            if last_good_solution is not None:
                print(f"\nReached target Ca = {Ca_target}")
                # Plot x0 vs Ca if we collected data
                if len(Ca_values) > 1:
                    plot_x0_vs_ca(Ca_values, x0_values, output_dir)
                return last_good_solution, Ca_target
        
        # Try this Ca value
        GLE_with_params = partial(GLE, Ca=Ca_step, mu_r=mu_r, lambda_slip=lambda_slip)
        bc_with_params = partial(boundary_conditions, w_bc=w_bc, theta0=theta0, lambda_slip=lambda_slip)
        temp_solution = solve_bvp(GLE_with_params, bc_with_params, s_range_local, current_guess, 
                                 max_nodes=1000000, tol=1e-6, verbose=0)
        
        if temp_solution.success:
            # Update for next iteration
            current_guess = temp_solution.y
            s_range_local = temp_solution.x
            Ca_critical = Ca_step
            last_good_solution = temp_solution
            
            # Calculate x0 (position where theta is minimum)
            s_vals = temp_solution.x
            h_vals, theta_vals, w_vals = temp_solution.y
            
            # Find minimum theta
            min_theta_idx = np.argmin(theta_vals)
            
            # Calculate x values
            x_vals = np.zeros_like(s_vals)
            x_vals[1:] = np.cumsum(np.diff(s_vals) * np.cos(theta_vals[:-1]))
            
            # Store x0 (x at minimum theta)
            x0 = x_vals[min_theta_idx]
            Ca_values.append(Ca_step)
            x0_values.append(x0)
            
            # If we reached target Ca exactly, we're done
            if Ca_step == Ca_target:
                print(f"\nSuccessfully reached target Ca = {Ca_target}")
                # Plot x0 vs Ca if we collected data
                if len(Ca_values) > 1:
                    plot_x0_vs_ca(Ca_values, x0_values, output_dir)
                return temp_solution, Ca_target
        else:
            # Failed - we've found the critical Ca
            print(f"\nCritical Ca found: Ca_cr = {Ca_critical:.6f}")
            print(f"(Failed at Ca = {Ca_step:.6f})")
            if last_good_solution is not None:
                # Plot x0 vs Ca if we collected data
                if len(Ca_values) > 1:
                    plot_x0_vs_ca(Ca_values, x0_values, output_dir)
                return last_good_solution, Ca_critical
            else:
                print("No solution found in continuation method.")
                return None, 0
    
    # If we went through all steps successfully but didn't reach target Ca
    if last_good_solution is not None and Ca_critical < Ca_target:
        print(f"\nMaximum Ca reached: Ca_cr = {Ca_critical:.6f} < target Ca = {Ca_target}")
    
    # Plot x0 vs Ca if we collected data
    if len(Ca_values) > 1:
        plot_x0_vs_ca(Ca_values, x0_values, output_dir)
    
    return last_good_solution, Ca_critical

def run_solver_and_plot(Delta, Ca, lambda_slip, mu_r, theta0, w, nGridInit=DEFAULT_NGRID_INIT, GUI=False, output_dir='output'):
    """Run the solver and either display or save plots

    Args:
        Delta (float): Maximum s-value for the solver
        Ca (float): Capillary number
        lambda_slip (float): Slip length
        mu_r (float): Viscosity ratio
        theta0 (float): Initial contact angle
        w (float): Curvature boundary condition at s=Delta
        nGridInit (int): Initial number of grid points
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
    s_range_local = np.linspace(0, Delta, nGridInit)  # Define the range of s
    y_guess_local = np.zeros((3, s_range_local.size))  # Initial guess for [h, theta, omega]
    y_guess_local[0, :] = np.linspace(lambda_slip, Delta, s_range_local.size)  # Linear guess for h
    y_guess_local[1, :] = theta0  # Initial guess for theta
    y_guess_local[2, :] = 0          # Initial guess for omega (dTheta/ds)

    # Solve the ODEs with continuation method on Ca if direct solve fails
    # First try direct solve
    print(f"\nAttempting direct solve with Ca = {Ca}, mu_r = {mu_r}...")
    GLE_with_params = partial(GLE, Ca=Ca, mu_r=mu_r, lambda_slip=lambda_slip)
    bc_with_params = partial(boundary_conditions, w_bc=w, theta0=theta0, lambda_slip=lambda_slip)
    solution = solve_bvp(GLE_with_params, bc_with_params, s_range_local, y_guess_local, 
                        max_nodes=1000000, tol=1e-6, verbose=0)
    
    Ca_actual = Ca  # The actual Ca used in the solution (may be Ca_critical if Ca > Ca_cr)
    
    if not solution.success:
        # Use continuation method to find critical Ca
        solution, Ca_actual = find_critical_ca_continuation(
            Ca, mu_r, lambda_slip, theta0, w, Delta, s_range_local, y_guess_local, output_dir
        )
        
        if solution is None:
            # Create a dummy failed solution for compatibility
            solution = type('obj', (object,), {
                'success': False, 
                'x': s_range_local, 
                'y': y_guess_local,
                'niter': 0
            })

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
    
    # Add overall title if using critical Ca
    if Ca_actual < Ca:
        fig.suptitle(f'GLE Solution at Critical Ca = {Ca_actual:.6f} (requested Ca = {Ca:.6f})', 
                     fontsize=16, fontweight='bold', color='darkred')
    
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
    if Ca_actual < Ca:
        textstr = f'Ca = {Ca_actual:.6f} (Ca_cr)\n$\\lambda_\\text{{slip}}$ = {lambda_slip:.0e}\n$\\mu_r$ = {mu_r:.0e}'
        facecolor = 'salmon'  # Red-ish to indicate critical Ca
    else:
        textstr = f'Ca = {Ca_actual:.6f}\n$\\lambda_\\text{{slip}}$ = {lambda_slip:.0e}\n$\\mu_r$ = {mu_r:.0e}'
        facecolor = 'wheat'
    props = dict(boxstyle='round', facecolor=facecolor, alpha=0.5)
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
    
    # Plot 4: theta(s) vs x(s)
    ax4.plot(x_values_local, theta_values_deg, '-', 
             color=solver_color, linewidth=2.5)
    ax4.set_xlabel('x(s)', fontsize=12)
    ax4.set_ylabel('$\\theta(s)$ [degrees]', fontsize=12)
    ax4.set_title('Contact Angle vs Horizontal Position', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1.01*max(x_values_local))
    ax4.set_ylim(0.99*min(theta_values_deg), 1.01*max(theta_values_deg))
    
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

    return solution, s_values_local, h_values_local, theta_values_local, w_values_local, Ca_actual

# Main execution
if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Solve Generalized Lubrication Equations (GLE)')
    
    # Add arguments for parameters
    parser.add_argument('--delta', type=float, default=DEFAULT_DELTA,
                        help=f'Maximum s-value for the solver (default: {DEFAULT_DELTA})')
    parser.add_argument('--ca', type=float, default=DEFAULT_CA,
                        help=f'Capillary number (default: {DEFAULT_CA})')
    parser.add_argument('--lambda-slip', type=float, default=DEFAULT_LAMBDA_SLIP,
                        help=f'Slip length (default: {DEFAULT_LAMBDA_SLIP})')
    parser.add_argument('--mu-r', type=float, default=DEFAULT_MU_R,
                        help=f'Viscosity ratio mu_g/mu_l (default: {DEFAULT_MU_R})')
    parser.add_argument('--theta0', type=float, default=DEFAULT_THETA0*180/np.pi,
                        help=f'Initial contact angle in degrees (default: {DEFAULT_THETA0*180/np.pi:.0f})')
    parser.add_argument('--w', type=float, default=DEFAULT_W,
                        help=f'Curvature boundary condition at s=Delta (default: {DEFAULT_W})')
    parser.add_argument('--ngrid-init', type=int, default=DEFAULT_NGRID_INIT,
                        help=f'Initial number of grid points (default: {DEFAULT_NGRID_INIT})')
    parser.add_argument('--gui', action='store_true',
                        help='Display plots in GUI mode')
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory for plots and data (default: output)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Convert theta0 from degrees to radians
    theta0_rad = args.theta0 * np.pi / 180
    
    # Run solver with parsed parameters
    solution, s_values_final, h_values_final, theta_values_final, w_values_final, Ca_actual = run_solver_and_plot(
        Delta=args.delta,
        Ca=args.ca,
        lambda_slip=args.lambda_slip,
        mu_r=args.mu_r,
        theta0=theta0_rad,
        w=args.w,
        nGridInit=args.ngrid_init,
        GUI=args.gui,
        output_dir=args.output_dir
    )

    print(f"Solution converged: {solution.success}")
    print(f"Number of iterations: {solution.niter}")
    
    # Print parameters used
    print("\nParameters used:")
    print(f"  Delta: {args.delta}")
    print(f"  Ca_requested: {args.ca}")
    if Ca_actual < args.ca:
        print(f"  Ca_actual: {Ca_actual:.6f} (Ca_critical - maximum Ca for convergence)")
    else:
        print(f"  Ca_actual: {Ca_actual:.6f}")
    print(f"  lambda_slip: {args.lambda_slip}")
    print(f"  mu_r: {args.mu_r}")
    print(f"  theta0: {args.theta0}° ({theta0_rad:.4f} rad)")
    print(f"  w: {args.w}")

    if not args.gui:
        print(f"\nPlot saved to: {args.output_dir}/GLE_profiles.png")


# Note: difference between this code and the ones from our [coalleauges](https://doi.org/10.1140/epjs/s11734-024-01443-5) is that we are solving for a specific control parameter whereas they use continuation method to track solution branches as parameters vary.
