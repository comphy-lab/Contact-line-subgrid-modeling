import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

#Parameters
Ca = 1.0  # Capillary number
lambda_slip = 1.0  # Slip length

# Define f1, f2, and f3 functions
def f1(theta):
    return theta**2 - np.sin(theta)**2

def f2(theta):
    return theta - np.sin(theta) * np.cos(theta)

def f3(theta):
    return theta * (np.pi - theta) + np.sin(theta)**2

# Define f(theta, R) function
def f(theta, R):
    numerator = 2 * np.sin(theta)**3 * (R**2 * f1(theta) + 2 * R * f3(theta) + f1(np.pi - theta))
    denominator = 3 * (R * f1(theta) * f2(np.pi - theta) - f1(np.pi - theta) * f2(theta))
    return numerator / denominator

# Initial conditions
h0 = lambda_slip  # h at s = 0
theta0 = np.pi/6  # theta at s = 0
kappa0 = 1 #curvature at h=\Delta
R = 1 # \mu_g/\mu_l

# Define the coupled ODEs system
def GLE(y):
    t, dt_ds = y
    dh_ds = np.sin(t)
    d2t_ds2 = 3 * Ca * f(t, R) / (h * (h + 3 * lambda_slip))
    return [dt_ds, d2t_ds2]

# Set up the solver parameters
s_range = (0, 10)  # Integration limits for s
y0 = [h0, np.sin(theta0)]  # Initial conditions

# Solve the ODE
solution = odeint(GLE, s_range, y0, method='RK45', t_eval=np.linspace(s_range[0], s_range[1], 500))

# Check if the solver was successful
if solution.status != 0:
    print("Warning: The solver failed to integrate properly.")

# Extract the results
if not isinstance(solution.y, np.ndarray) or len(solution.y) < 2:
    print("Error: Solver did not return the expected number of outputs.")
    s_values = []
    h_values = []
    dh_ds_values = []
    theta_values = []
else:
    s_values = solution.t
    h_values = solution.y[0]
    dh_ds_values = solution.y[1]
    theta_values = np.arcsin(dh_ds_values)
    theta_values = theta_values*(180/np.pi)

# Plot the results if data is available
if len(s_values) > 0:
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot(s_values, h_values, label='h(s)')
    plt.xlabel('s')
    plt.ylabel('h')
    plt.title('h(s)')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(s_values, theta_values, label='${\\theta(s)}$')
    plt.xlabel('s')
    plt.ylabel(r'$\theta$')
    plt.title('$\\theta(s)$')
    plt.legend()

    plt.tight_layout()
    plt.show()
else:
    print("No solution to plot.")
