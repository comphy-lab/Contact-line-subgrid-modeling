# PLotting the difference in Huh and Scriven velocity at the grid size and the plate velocity over all \theta and \phi
# to see how significantly the boundary conditions change

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Parameters
U_p = 1  # Plate velocity
Theta = np.linspace(np.pi*1/180, np.pi, 100)  # Contact angle
Phi = np.linspace(0, Theta, 100)  # Angle between the interface and the plate

# Define velocities

# The common term in both U_r and U_phi
def term(U_p, Theta):
    return U_p/(Theta - np.cos(Theta)*np.sin(Theta)) 

# Radial component of the velocity
def Ur(Theta, Phi):
    return term(U_p, Theta) * ((np.cos(Phi) - Phi*np.sin(Phi))*np.sin(Theta) - Theta*np.cos(Theta)*np.cos(Phi))

# Azimuthal component of the velocity
def Uphi(Theta, Phi):
    return term(U_p, Theta) * (Theta*np.sin(Phi)*np.cos(Theta) - Phi*np.cos(Phi)*np.sin(Theta))

# x component of the velocity
def Ux(Theta, Phi):
    return Ur(Theta, Phi) * np.cos(Theta-Phi) - Uphi(Theta, Phi) * np.sin(Theta-Phi)

# y component of the velocity
def Uy(Theta, Phi):
    return Ur(Theta, Phi) * np.sin(Theta-Phi) + Uphi(Theta, Phi) * np.cos(Theta-Phi)

def compute_and_plot(GUI=False, output_dir='plots'):
    """Compute velocity fields and either display or save plots
    
    Args:
        GUI (bool): If True, display plots. If False, save to files.
        output_dir (str): Directory to save plots when GUI=False
    
    Returns:
        tuple: (Theta_grid, Phi_grid, Ux_rel_grid, Uy_rel_grid)
    """
    # Set matplotlib backend based on GUI parameter
    if not GUI:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    # Create a meshgrid for Theta and Phi
    Theta_grid = []
    Phi_grid = []
    for theta in Theta:
        phi_values = np.linspace(0, theta, 100)
        Theta_grid.extend([theta] * len(phi_values))
        Phi_grid.extend(phi_values)

    Theta_grid = np.array(Theta_grid)
    Phi_grid = np.array(Phi_grid)

    # Compute Ux, Uy
    Ux_grid = Ux(Theta_grid, Phi_grid)
    Uy_grid = Uy(Theta_grid, Phi_grid)

    # Compute Ux_rel and Uy_rel
    Ux_rel_grid = -U_p - Ux_grid
    Uy_rel_grid = -Uy_grid

    # Convert Theta and Phi to degrees
    Theta_grid_deg = np.degrees(Theta_grid)
    Phi_grid_deg = np.degrees(Phi_grid)

    # Plot Ux_rel as a scatter plot
    plt.figure(figsize=(10, 5))
    plt.scatter(Theta_grid_deg, Phi_grid_deg, c=Ux_rel_grid, cmap='viridis', s=10, vmin=-0.1, vmax = 0.1)
    plt.colorbar(label='Ux_rel')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Phi (degrees)')
    plt.title('Scatter Plot of Ux_rel')
    plt.grid()
    if GUI:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, 'huh_scriven_Ux_rel.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # Plot Uy_rel as a scatter plot
    plt.figure(figsize=(10, 5))
    plt.scatter(Theta_grid_deg, Phi_grid_deg, c=Uy_rel_grid, cmap='plasma', s=10, vmin=-0.1, vmax = 0.1)
    plt.colorbar(label='Uy_rel')
    plt.xlabel('Theta (degrees)')
    plt.ylabel('Phi (degrees)')
    plt.title('Scatter Plot of Uy_rel')
    plt.grid()
    if GUI:
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, 'huh_scriven_Uy_rel.png'), dpi=150, bbox_inches='tight')
        plt.close()
    
    return Theta_grid, Phi_grid, Ux_rel_grid, Uy_rel_grid

# Main execution
if __name__ == "__main__":
    # Check for command line argument
    gui_mode = False  # Default is no GUI
    if len(sys.argv) > 1 and sys.argv[1] == '--gui':
        gui_mode = True
    
    Theta_grid, Phi_grid, Ux_rel_grid, Uy_rel_grid = compute_and_plot(GUI=gui_mode)
    
    if not gui_mode:
        print(f"Plots saved in 'plots' directory")
        print(f"Ux_rel range: [{np.min(Ux_rel_grid):.4f}, {np.max(Ux_rel_grid):.4f}]")
        print(f"Uy_rel range: [{np.min(Uy_rel_grid):.4f}, {np.max(Uy_rel_grid):.4f}]")
