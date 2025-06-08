"""
/**
# Simple Continuation with δX_cl vs Ca Plotting

Natural parameter continuation that tracks δX_cl vs Ca up to the fold point.
This is simpler than full pseudo-arclength but still shows the key physics.

Author: Assistant
Date: 2025
*/
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Optional
import sys
import os

sys.path.append('src-local')
from gle_utils import solve_single_ca
from find_x0_utils import find_x0_and_theta_min


@dataclass  
class SolutionData:
    Ca: float
    X_cl: float
    delta_X_cl: float
    theta_min: float
    converged: bool = True


def trace_branch_simple(mu_r: float, lambda_slip: float, theta0: float, 
                       Delta: float = 10.0, w_bc: float = 0.0,
                       Ca_start: float = 0.001, Ca_step_init: float = 0.005,
                       Ca_max: float = 0.1) -> List[SolutionData]:
    """
    Simple continuation in Ca parameter space.
    """
    print(f"\nStarting continuation with:")
    print(f"  mu_r = {mu_r}")
    print(f"  lambda_slip = {lambda_slip}")
    print(f"  theta0 = {theta0*180/np.pi:.1f}°\n")
    
    # Get reference X_cl at Ca ≈ 0
    s_init = np.linspace(0, Delta, 10000)
    y_init = np.zeros((3, len(s_init)))
    y_init[0, :] = lambda_slip + s_init * np.sin(theta0)
    y_init[1, :] = theta0
    y_init[2, :] = 0.0
    
    result_ref = solve_single_ca(1e-6, mu_r, lambda_slip, theta0, w_bc, Delta, s_init, y_init)
    X_cl_ref = result_ref.x0 if result_ref.success else 0
    print(f"Reference X_cl at Ca≈0: {X_cl_ref:.6f}\n")
    
    # Initialize
    branch = []
    Ca = Ca_start
    Ca_step = Ca_step_init
    y_guess = y_init
    s_guess = s_init
    
    # Main loop
    while Ca < Ca_max:
        result = solve_single_ca(Ca, mu_r, lambda_slip, theta0, w_bc, Delta, s_guess, y_guess)
        
        if result.success:
            # Store solution
            delta_X = result.x0 - X_cl_ref
            sol = SolutionData(
                Ca=Ca,
                X_cl=result.x0,
                delta_X_cl=delta_X,
                theta_min=result.theta_min,
                converged=True
            )
            branch.append(sol)
            
            print(f"Ca={Ca:.6f}, X_cl={result.x0:.4f}, "
                  f"δX_cl={delta_X:.4f}, θ_min={result.theta_min*180/np.pi:.1f}°")
            
            # Update guess for next step
            s_guess = result.s_range
            y_guess = result.solution.y
            
            # Adaptive step size
            if result.theta_min < 0.2:  # Near critical Ca
                Ca_step *= 0.5
            else:
                Ca_step = min(Ca_step * 1.2, 0.01)
                
            # Check if approaching critical Ca
            if result.theta_min < 0.05:
                print(f"\nApproaching critical Ca - θ_min = {result.theta_min*180/np.pi:.1f}°")
                if Ca_step < 1e-5:
                    print("Step size too small, stopping")
                    break
                    
        else:
            # Failed - reduce step size
            Ca_step *= 0.5
            if Ca_step < 1e-5:
                print(f"\nSolver failed at Ca={Ca:.6f}, stopping")
                break
            continue
            
        Ca += Ca_step
        
    return branch


def plot_results(branch: List[SolutionData], mu_r: float, lambda_slip: float, 
                theta0: float, output_dir: str = 'output'):
    """Plot δX_cl vs Ca curve."""
    os.makedirs(output_dir, exist_ok=True)
    
    Ca_vals = [s.Ca for s in branch]
    delta_X_vals = [s.delta_X_cl for s in branch]
    theta_min_vals = [s.theta_min * 180/np.pi for s in branch]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot 1: δX_cl vs Ca
    ax1.plot(Ca_vals, delta_X_vals, 'b.-', linewidth=2.5, markersize=8)
    ax1.set_xlabel('Ca', fontsize=14)
    ax1.set_ylabel('δX_cl', fontsize=14)
    ax1.set_title('Contact Line Displacement vs Capillary Number', fontsize=16)
    ax1.grid(True, alpha=0.3)
    
    # Add parameters
    textstr = (f'μ_r = {mu_r:.1e}\n'
              f'λ_slip = {lambda_slip:.1e}\n'
              f'θ₀ = {theta0*180/np.pi:.0f}°')
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    # Highlight near-critical region
    if min(theta_min_vals) < 10:
        critical_idx = next(i for i, t in enumerate(theta_min_vals) if t < 10)
        ax1.axvline(x=Ca_vals[critical_idx], color='r', linestyle='--', alpha=0.5)
        ax1.text(Ca_vals[critical_idx], max(delta_X_vals)*0.9, 
                f'Ca_cr ≈ {Ca_vals[-1]:.4f}', rotation=90, 
                verticalalignment='bottom', fontsize=10)
    
    # Plot 2: θ_min vs Ca  
    ax2.plot(Ca_vals, theta_min_vals, 'g.-', linewidth=2.5, markersize=8)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Ca', fontsize=14)
    ax2.set_ylabel('θ_min [degrees]', fontsize=14)
    ax2.set_title('Minimum Contact Angle vs Capillary Number', fontsize=16)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=-5)
    
    plt.tight_layout()
    
    # Save
    plot_file = os.path.join(output_dir, 'delta_X_cl_vs_Ca.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved to: {plot_file}")
    
    # Save data
    data_file = os.path.join(output_dir, 'delta_X_cl_vs_Ca.txt')
    with open(data_file, 'w') as f:
        f.write("# Simple continuation results\n")
        f.write(f"# mu_r={mu_r}, lambda_slip={lambda_slip}, theta0={theta0*180/np.pi}°\n")
        f.write("# Ca, X_cl, delta_X_cl, theta_min_deg\n")
        for s in branch:
            f.write(f"{s.Ca:.8f} {s.X_cl:.8f} {s.delta_X_cl:.8f} {s.theta_min*180/np.pi:.4f}\n")
    print(f"Data saved to: {data_file}")


def main():
    """Run simple continuation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Simple continuation for GLE with δX_cl tracking'
    )
    parser.add_argument('--mu_r', type=float, default=1e-6)
    parser.add_argument('--lambda_slip', type=float, default=1e-4)
    parser.add_argument('--theta0', type=float, default=60,
                        help='Initial angle in degrees')
    parser.add_argument('--output-dir', type=str, default='output')
    
    args = parser.parse_args()
    
    # Run continuation
    branch = trace_branch_simple(
        mu_r=args.mu_r,
        lambda_slip=args.lambda_slip,
        theta0=np.radians(args.theta0)
    )
    
    # Plot results
    plot_results(branch, args.mu_r, args.lambda_slip, 
                np.radians(args.theta0), args.output_dir)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Traced {len(branch)} points on solution branch")
    if branch:
        print(f"Ca range: {branch[0].Ca:.6f} to {branch[-1].Ca:.6f}")
        print(f"Final θ_min: {branch[-1].theta_min*180/np.pi:.2f}°")
        print(f"Estimated Ca_critical ≈ {branch[-1].Ca:.6f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()