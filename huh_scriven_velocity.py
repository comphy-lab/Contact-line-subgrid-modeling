# PLotting the difference in Huh and Scriven velocity at the grid size and the plate velocity over all \theta and \phi
# to see how significantly the boundary conditions change

import numpy as np
import matplotlib.pyplot as plt

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

# Create a meshgrid for Theta and Phi
Theta_grid = []
Phi_grid = []
for theta in Theta:
    phi_values = np.linspace(0, Theta, 100)
    Theta_grid.extend([Theta] * len(phi_values))
    Phi_grid.extend(phi_values)

Theta_grid = np.array(Theta_grid)
Phi_grid = np.array(Phi_grid)

# Compute Ur, Uphi, Ux, Uy
Ur_grid = Ur(Theta_grid, Phi_grid)
Uphi_grid = Uphi(Theta_grid, Phi_grid)
Ux_grid = Ux(Theta_grid, Phi_grid)
Uy_grid = Uy(Theta_grid, Phi_grid)

# Compute Ux_rel and Uy_rel
Ux_rel_grid = -U_p - Ux_grid
Uy_rel_grid = -Uy_grid

# Convert Theta and Phi to degrees
Theta_grid = np.degrees(Theta_grid)
Phi_grid = np.degrees(Phi_grid)

# Plot Ux_rel as a scatter plot
plt.figure(figsize=(10, 5))
plt.scatter(Theta_grid, Phi_grid, c=Ux_rel_grid, cmap='viridis', s=10, vmin=-0.1, vmax = 0.1)
plt.colorbar(label='Ux_rel')
plt.xlabel('Theta')
plt.ylabel('Phi')
plt.title('Scatter Plot of Ux_rel')
plt.grid()

# Plot Uy_rel as a scatter plot
plt.figure(figsize=(10, 5))
plt.scatter(Theta_grid, Phi_grid, c=Uy_rel_grid, cmap='plasma', s=10, vmin=-0.1, vmax = 0.1)
plt.colorbar(label='Uy_rel')
plt.xlabel('Theta')
plt.ylabel('Phi')
plt.title('Scatter Plot of Uy_rel')
plt.grid()

# Show the plots
plt.show()
