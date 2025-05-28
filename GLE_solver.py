import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

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

# Define f(theta, R) function
def f(theta, mu_r):
    numerator = 2 * np.sin(theta)**3 * (mu_r**2 * f1(theta) + 2 * mu_r * f3(theta) + f1(np.pi - theta))
    denominator = 3 * (mu_r * f1(theta) * f2(np.pi - theta) - f1(np.pi - theta) * f2(theta))
    return numerator / denominator

# Initial conditions
h0 = lambda_slip  # h at s = 0
theta0 = np.pi/6  # theta at s = 0
kappa0 = 1 #curvature at h=\Delta
w = 0  # curvature boundary condition at s = \Delta, this needs to be not remain constant, but fed back from the DNS



# Define the coupled ODEs system
def GLE(s, y):
    h, t, w = y
    dh_ds = np.sin(t) # dh/ds = sin(theta)
    dt_ds = w # omega = dtheta/ds
    dw_ds = 3 * Ca * f(t, mu_r) / (h * (h + 3 * lambda_slip))
    return [dh_ds, dt_ds, dw_ds]

# Set up the solver parameters
s_range = (0, 1e-3)  # Integration limits for s
# Need to set the initial conditions for the ODEs. Since we are setting them at different points, we need 3 as fixed, 3 as guesses
# \Theta at s=0, h at s=0, dTheta/ds at h=\Delta
# The guesses follow the known BCs when solved
# The 3rd "known" BC is the curvature at h=\Delta, which is not known, but can be fed back from the DNS

Delta = 1e-4  # Miminum grid cell size

# Boundary conditions
def boundary_conditions(ya, yb):
    # ya corresponds to s = 0, yb corresponds to s = Delta
    h_a, theta_a, w_a = ya
    h_b, theta_b, w_b = yb
    return [
        theta_a - theta0,      # theta(0) = pi/6
        h_a - lambda_slip,      # h(0) = lambda_slip
        w_b - w         # w(Delta) = w (curvature at s=Delta)
    ]

# Initial guess for the solution
s_range = np.linspace(0, 4*Delta, 10000)  # Define the range of s
y_guess = np.zeros((3, s_range.size))  # Initial guess for [theta, w, h]
y_guess[0, :] = np.linspace(lambda_slip, Delta, s_range.size)  # Linear guess for h
y_guess[1, :] = np.pi / 6  # Initial guess for theta
y_guess[2, :] = 0          # Initial guess for dTheta/ds


# Solve the ODEs
solution = solve_bvp(GLE, boundary_conditions, s_range, y_guess)

# Extract the solution
s_values = solution.x
h_values, theta_values, w_values = solution.y
theta_values = theta_values*180/np.pi

# Plot the results
plt.figure(figsize=(10, 5))
# plt.plot(s_values, theta_values, label=r'$\theta(s)$')
plt.plot(s_values, h_values, label=r'$h(s)$')
# plt.ylim(0, 1e-4)
plt.ylabel(r'$h(s)$')
plt.xlabel('s')
plt.grid()
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(s_values, theta_values, label=r'$\theta(s)$')
plt.xlabel('s')
plt.ylabel(r'$\theta(s)$')
plt.grid()
plt.show()