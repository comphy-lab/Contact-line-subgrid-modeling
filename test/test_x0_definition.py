"""
Test x0 Definition in GLE Solver

This script tests that x0 is correctly identified as the x-position where theta reaches
its minimum value across different Ca values.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../src-local')
from GLE_solver import solve_single_ca
from find_x0_utils import find_x0_from_solution, find_x0_and_theta_min

# Test parameters
mu_r = 1e-6
lambda_slip = 1e-4
theta0 = np.pi/3  # 60 degrees
w_bc = 0
Delta = 10.0

# Test at different Ca values
Ca_values = [0.0, 0.01, 0.02, 0.04]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for i, Ca in enumerate(Ca_values):
    print(f"\nTesting Ca = {Ca}")
    
    # Initial guess
    s_range = np.linspace(0, Delta, 1000)
    y_guess = np.zeros((3, len(s_range)))
    y_guess[0, :] = np.linspace(lambda_slip, Delta, len(s_range))
    y_guess[1, :] = theta0
    y_guess[2, :] = 0
    
    # Solve
    result = solve_single_ca(
        Ca=Ca, mu_r=mu_r, lambda_slip=lambda_slip, theta0=theta0,
        w_bc=w_bc, Delta=Delta, s_range=s_range, y_guess=y_guess
    )
    
    if result.success:
        h_vals, theta_vals, w_vals = result.solution.y
        s_vals = result.solution.x
        
        # Calculate x manually to verify
        x_vals = np.zeros_like(s_vals)
        x_vals[1:] = np.cumsum(np.diff(s_vals) * np.cos(theta_vals[:-1]))
        
        # Find x0 using our function
        x0, x0_idx, theta_min = find_x0_from_solution(s_vals, theta_vals)
        
        # Also get theta_max
        theta_max = np.max(theta_vals)
        
        print(f"  theta range: [{theta_min*180/np.pi:.2f}°, {theta_max*180/np.pi:.2f}°]")
        print(f"  x0 from solver: {result.x0}")
        print(f"  x0 from function: {x0}")
        print(f"  theta_min position: s = {s_vals[x0_idx]:.4f}, x = {x0:.4f}")
        
        # Plot
        ax = axes[i]
        ax.plot(s_vals, theta_vals * 180/np.pi, 'b-', linewidth=2)
        ax.axhline(y=theta_min*180/np.pi, color='g', linestyle='--', alpha=0.5, 
                   label=f'θ_min = {theta_min*180/np.pi:.1f}°')
        
        # Mark the position where theta is minimum (x0)
        if x0_idx is not None:
            ax.plot(s_vals[x0_idx], theta_vals[x0_idx] * 180/np.pi, 'go', 
                    markersize=10, label=f'x₀ = {x0:.3f}')
        
        ax.set_xlabel('s')
        ax.set_ylabel('θ [degrees]')
        ax.set_title(f'Ca = {Ca}')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(50, 100)
    else:
        print(f"  Solution failed!")

plt.suptitle('Testing x₀ Definition: Position where θ reaches minimum')
plt.tight_layout()
plt.savefig('test_x0_definition.png', dpi=150)
plt.close()

print("\nPlot saved to test_x0_definition.png")

# Test edge case: solution with varying theta
print("\n\nTesting case with varying theta profile:")
s_test = np.linspace(0, 10, 100)
# Create a theta profile that has a clear minimum
theta_test = np.pi/3 + 0.2 * np.sin(s_test) - 0.1 * s_test

x0_test, idx_test, theta_min_test = find_x0_from_solution(s_test, theta_test)
theta_max_test = np.max(theta_test)
print(f"  theta range: [{theta_min_test*180/np.pi:.2f}°, {theta_max_test*180/np.pi:.2f}°]")
print(f"  x0 index: {idx_test}")
if x0_test is not None:
    print(f"  x0 value: {x0_test:.4f}")
    print(f"  theta_min position: s = {s_test[idx_test]:.4f}")