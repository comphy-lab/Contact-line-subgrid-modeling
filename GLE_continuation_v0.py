#!/usr/bin/env python3
"""
GLE_continuation_simple.py

Simplified pseudo-arclength continuation for the Generalized Lubrication Equations (GLE).
Tracks contact line displacement δX_cl as a function of Capillary number (Ca).

This version uses a more robust approach with consistent mesh handling.

Author: Vatsal Sanjay
Date: January 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp
from scipy.interpolate import interp1d
import argparse
import time
import sys
from pathlib import Path

# Add src-local to path for utilities
sys.path.append('src-local')
from find_x0_utils import find_x0_and_theta_min
from gle_utils import GLE as GLE_func, boundary_conditions


class SimpleContinuation:
    """Simplified continuation solver for GLE system"""
    
    def __init__(self, mu_r: float, lambda_slip: float, theta0: float,
                 Delta: float = 10.0, N_points: int = 10000):
        self.mu_r = mu_r
        self.lambda_slip = lambda_slip
        self.theta0 = theta0
        self.Delta = Delta
        self.N_points = N_points
        
        # Fixed mesh for consistency
        self.s_mesh = np.linspace(0, Delta, N_points)
        
        # Storage for results
        self.Ca_vals = []
        self.theta_min_vals = []
        self.delta_x_cl_vals = []
        self.solutions = []
        self.x_cl_ref = None
        
    def solve_at_Ca(self, Ca: float, initial_guess=None):
        """Solve BVP at given Ca"""
        if initial_guess is None:
            # Default initial guess
            h_guess = self.lambda_slip + self.s_mesh * np.sin(self.theta0)
            theta_guess = self.theta0 * np.ones_like(self.s_mesh)
            omega_guess = np.zeros_like(self.s_mesh)
            initial_guess = np.vstack([h_guess, theta_guess, omega_guess])
        
        # Define ODE and BC functions
        def ode_fun(s, y):
            return GLE_func(s, y, Ca, self.mu_r, self.lambda_slip)
        
        def bc_fun(ya, yb):
            return boundary_conditions(ya, yb, 0.0, self.theta0, self.lambda_slip)
        
        # Solve BVP
        sol = solve_bvp(ode_fun, bc_fun, self.s_mesh, initial_guess,
                       tol=1e-6, max_nodes=100000, verbose=0)
        
        if not sol.success:
            raise RuntimeError(f"BVP failed at Ca={Ca}: {sol.message}")
        
        # Always interpolate solution back to fixed mesh
        if len(sol.x) != self.N_points:
            sol_interp = np.zeros((3, self.N_points))
            for i in range(3):
                f_interp = interp1d(sol.x, sol.y[i, :], kind='cubic',
                                   bounds_error=False, fill_value='extrapolate')
                sol_interp[i, :] = f_interp(self.s_mesh)
            return sol_interp
        else:
            return sol.y
    
    def compute_x_cl(self, theta):
        """Compute contact line position"""
        # Simple estimate based on far-field angle
        far_field_idx = int(0.9 * len(self.s_mesh))
        theta_macro = np.mean(theta[far_field_idx:])
        X_cl = self.Delta * (theta_macro - self.theta0) / np.pi
        return X_cl
    
    def run_continuation(self, Ca_max=0.1, dCa_init=0.001):
        """Run simple continuation with adaptive Ca stepping"""
        print(f"\nStarting continuation to Ca_max={Ca_max}")
        
        # Initialize at Ca=0
        try:
            sol0 = self.solve_at_Ca(0.0)
            theta_min = np.min(sol0[1, :])
            x_cl = self.compute_x_cl(sol0[1, :])
            self.x_cl_ref = x_cl
            
            self.Ca_vals.append(0.0)
            self.theta_min_vals.append(theta_min)
            self.delta_x_cl_vals.append(0.0)
            self.solutions.append(sol0)
            
            print(f"Step 0: Ca=0.0000, θ_min={np.degrees(theta_min):.2f}°")
            
        except Exception as e:
            print(f"Failed to initialize at Ca=0: {e}")
            return
        
        # Continuation loop
        Ca = 0.0
        dCa = dCa_init
        sol_prev = sol0
        step = 0
        
        while Ca < Ca_max and step < 500:
            # Predict next Ca
            Ca_new = Ca + dCa
            
            try:
                # Solve at new Ca using previous solution as guess
                sol_new = self.solve_at_Ca(Ca_new, sol_prev)
                
                # Extract properties
                theta_min = np.min(sol_new[1, :])
                x_cl = self.compute_x_cl(sol_new[1, :])
                delta_x_cl = x_cl - self.x_cl_ref
                
                # Check if solution is still physical
                if theta_min < np.radians(1.0):  # Less than 1 degree
                    print(f"\nApproaching θ_min=0 at Ca={Ca_new:.6f}")
                    break
                
                # Store results
                self.Ca_vals.append(Ca_new)
                self.theta_min_vals.append(theta_min)
                self.delta_x_cl_vals.append(delta_x_cl)
                self.solutions.append(sol_new)
                
                step += 1
                print(f"Step {step}: Ca={Ca_new:.4f}, θ_min={np.degrees(theta_min):.2f}°, "
                      f"δX_cl={delta_x_cl:.6f}")
                
                # Adaptive step size
                theta_change = abs(theta_min - np.min(sol_prev[1, :]))
                if theta_change < np.radians(0.1):  # Small change
                    dCa = min(dCa * 1.5, 0.01)  # Increase step
                elif theta_change > np.radians(1.0):  # Large change
                    dCa = max(dCa * 0.7, 0.0001)  # Decrease step
                
                # Update for next iteration
                Ca = Ca_new
                sol_prev = sol_new
                
            except Exception as e:
                print(f"\nSolver failed at Ca={Ca_new:.6f}")
                print(f"Error: {e}")
                
                # Try smaller step
                dCa *= 0.5
                if dCa < 1e-6:
                    print("Step size too small, stopping")
                    break
                else:
                    print(f"Reducing step size to {dCa:.6f}")
        
        print(f"\nContinuation completed with {len(self.Ca_vals)} points")
        print(f"Final Ca: {self.Ca_vals[-1]:.6f}")
        
    def plot_results(self, save_dir='output'):
        """Plot continuation results"""
        if len(self.Ca_vals) < 2:
            print("Not enough data to plot")
            return
        
        Path(save_dir).mkdir(exist_ok=True)
        
        # Convert to arrays
        Ca_arr = np.array(self.Ca_vals)
        theta_min_deg = np.degrees(self.theta_min_vals)
        delta_x_cl_arr = np.array(self.delta_x_cl_vals)
        
        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: δX_cl vs Ca
        ax1.plot(Ca_arr, delta_x_cl_arr, 'b-', linewidth=2)
        ax1.set_xlabel('Capillary number (Ca)', fontsize=12)
        ax1.set_ylabel('Contact line displacement (δX_cl)', fontsize=12)
        ax1.set_title('Contact Line Displacement vs Ca', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: θ_min vs Ca
        ax2.plot(Ca_arr, theta_min_deg, 'g-', linewidth=2)
        ax2.set_xlabel('Capillary number (Ca)', fontsize=12)
        ax2.set_ylabel('Minimum contact angle (degrees)', fontsize=12)
        ax2.set_title('Minimum Contact Angle vs Ca', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        # Add parameters
        param_text = (
            f"μ_r = {self.mu_r:.2e}\n"
            f"λ_slip = {self.lambda_slip:.2e}\n"
            f"θ₀ = {np.degrees(self.theta0):.1f}°"
        )
        fig.text(0.02, 0.95, param_text, transform=fig.transFigure, 
                fontsize=10, bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        plt.tight_layout()
        
        # Save figure
        filename = f"simple_continuation_mu{self.mu_r}_lambda{self.lambda_slip}.png"
        filepath = Path(save_dir) / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {filepath}")
        plt.close()
        
        # Also save data
        import pandas as pd
        df = pd.DataFrame({
            'Ca': Ca_arr,
            'theta_min_deg': theta_min_deg,
            'delta_x_cl': delta_x_cl_arr
        })
        csv_file = Path(save_dir) / f"simple_continuation_mu{self.mu_r}_lambda{self.lambda_slip}.csv"
        df.to_csv(csv_file, index=False)
        print(f"Data saved to {csv_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Simple continuation for GLE system'
    )
    
    parser.add_argument('--mu_r', type=float, default=1.0,
                       help='Viscosity ratio (default: 1.0)')
    parser.add_argument('--lambda_slip', type=float, default=1e-4,
                       help='Slip length (default: 1e-4)')
    parser.add_argument('--theta0', type=float, default=10.0,
                       help='Initial contact angle in degrees (default: 10)')
    parser.add_argument('--Ca_max', type=float, default=0.1,
                       help='Maximum Ca (default: 0.1)')
    parser.add_argument('--Delta', type=float, default=10.0,
                       help='Domain size (default: 10.0)')
    parser.add_argument('--N_points', type=int, default=10000,
                       help='Number of mesh points (default: 10000)')
    
    args = parser.parse_args()
    
    # Convert theta0 to radians
    theta0_rad = np.radians(args.theta0)
    
    # Create solver
    solver = SimpleContinuation(
        mu_r=args.mu_r,
        lambda_slip=args.lambda_slip,
        theta0=theta0_rad,
        Delta=args.Delta,
        N_points=args.N_points
    )
    
    print(f"\nSimple continuation solver")
    print(f"Parameters:")
    print(f"  mu_r = {args.mu_r}")
    print(f"  lambda_slip = {args.lambda_slip}")
    print(f"  theta0 = {args.theta0}°")
    print(f"  Ca_max = {args.Ca_max}")
    
    # Run continuation
    start_time = time.time()
    solver.run_continuation(Ca_max=args.Ca_max)
    elapsed = time.time() - start_time
    
    print(f"\nCompleted in {elapsed:.2f} seconds")
    
    # Plot results
    solver.plot_results()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())