# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains Python implementations for contact line subgrid modeling, focusing on solving the Generalized Lubrication Equations (GLE) and analyzing Huh-Scriven velocity fields near moving contact lines in fluid dynamics. The project includes advanced bifurcation analysis tools to study both stable and unstable solution branches.

## Recent Updates

### x0 and X_cl Definitions (Updated)
- **x0**: The x-position where θ reaches its minimum value
- **X_cl**: The contact line position, defined as the maximum x value (x at s=Delta)
- These definitions are implemented in `src-local/find_x0_utils.py`
- δX_cl = X_cl(Ca) - X_cl(Ca≈0) represents the contact line displacement
- When θ_min approaches 0, we are at the critical capillary number (Ca_cr)

## Key Components

### GLE_solver.py
- Simple solver for the Generalized Lubrication Equations
- Attempts to solve for a given Capillary number (Ca)
- If solution fails (Ca > Ca_critical), suggests using critical Ca finders
- Key functions:
  - `solve_gle()`: Main solver function
  - `create_solution_plots()`: Visualization of successful solutions
- Uses scipy's solve_bvp with standard settings

### GLE_criticalCa.py
- Finds the critical Capillary number (Ca_cr) where solutions cease to exist
- Uses a hybrid refinement approach: IQI/Newton/bisection methods
- Key functions:
  - `f1(theta)`, `f2(theta)`, `f3(theta)`: Helper functions for the GLE formulation
  - `f(theta, mu_r)`: Main function combining the helpers with viscosity ratio mu_r
  - `GLE(s, y, Ca, mu_r, lambda_slip)`: System of ODEs defining the contact line shape evolution
  - `find_critical_ca_lower_branch()`: Two-stage method to find critical Capillary number
  - `hybrid_iqi_newton_refinement()`: Advanced refinement using multiple methods
- Features adaptive tolerance, solution interpolation, and automatic convergence detection

### GLE_criticalCa_advanced.py
- Advanced critical Ca finder with enhanced algorithms
- More robust and accurate than the basic finder in GLE_criticalCa.py
- Features:
  - Adaptive mesh refinement based on solution properties
  - Optimized search strategy
- Key functions:
  - `find_critical_ca_advanced()`: Main function with adaptive refinement
  - `analyze_solution_properties()`: Solution analysis for mesh adaptation
  - Uses enhanced Phase 0 from GLE_criticalCa with improvements
- NOTE: This is NOT a continuation method - it specifically finds Ca_critical

### GLE_continuation_v4.py
- Unified continuation solver supporting both pseudo-arclength and natural parameter methods
- Key features:
  - Tracks contact line displacement δX_cl = X_cl(Ca) - X_cl(Ca≈0)
  - Natural parameter method: Simple Ca stepping for stable branches
  - Simplified pseudo-arclength: More robust near folds but cannot trace unstable branch
  - Automatic fold detection and branch classification
  - Adaptive step size control
- Key functions:
  - `GLEContinuation`: Main class with method selection
  - `trace_branch()`: Main continuation loop
  - `_predictor_arclength()` and `_predictor_natural()`: Different prediction strategies
  - `_corrector()`: Simplified correction step
  - `analyze_branch()`: Branch analysis including fold identification
- Best for: Finding critical Ca and stable branch analysis

### GLE_continuation_v4.5.py
- Extended pseudo-arclength continuation attempting to trace through folds
- Key features:
  - Implements arc-length constraint (simplified scalar approach)
  - Attempts to trace through fold bifurcations
  - Gets closer to Ca_critical than v4
  - Currently fails near fold point (cannot capture unstable branch)
- Key functions:
  - `GLEContinuationExtended`: Main class for extended method
  - `newton_extended()`: Scalar Newton solver for arc-length constraint
  - `compute_tangent()`: Normalized tangent using Ca and X_cl only
- Current limitations: Cannot successfully trace past fold to unstable branch
- See issues #14 and #16 on GitHub for details

### huh_scriven_velocity.py
- Analyzes the Huh-Scriven velocity field near moving contact lines
- Computes velocity components in polar (Ur, Uphi) and Cartesian (Ux, Uy) coordinates
- Visualizes relative velocities between the fluid and moving plate

## Development Commands

```bash
# Run the simple GLE solver (for Ca < Ca_critical)
python GLE_solver.py --ca 0.01 --mu_r 1e-6 --lambda_slip 1e-4

# Find the critical Ca using standard method
python GLE_criticalCa.py --ca 0.05 --mu_r 1e-6 --lambda_slip 1e-4

# Find the critical Ca using advanced method with adaptive mesh
python GLE_criticalCa_advanced.py --mu_r 1e-6 --lambda_slip 1e-4

# Run continuation methods to track δX_cl vs Ca
# Method 1: Unified solver with natural parameter (finds critical Ca on stable branch)
python GLE_continuation_v4.py --method natural --mu_r 1e-6 --lambda_slip 1e-3 --theta0 60

# Method 2: Unified solver with simplified pseudo-arclength (more robust near folds)
python GLE_continuation_v4.py --method arclength --mu_r 1e-6 --lambda_slip 1e-3 --theta0 60

# Method 3: Extended pseudo-arclength (traces through folds, captures unstable branch)
python GLE_continuation_v4.5.py --mu_r 1e-6 --lambda_slip 1e-3 --theta0 60

# Run the Huh-Scriven velocity analysis
python huh_scriven_velocity.py

# Run tests
pytest test/
```

## Mathematical Context

The code implements solutions to the contact line dynamics problem where:
- Ca: Capillary number (ratio of viscous to surface tension forces)
- lambda_slip: Slip length parameter (λ)
- mu_r: Viscosity ratio (μ_g/μ_l)
- theta: Contact angle
- h: Film thickness
- s: Arc length coordinate along the interface
- x0: Position where θ reaches its minimum value
- X_cl: Contact line position (x at s=Delta, the end of the domain)
- δX_cl: Contact line displacement relative to Ca≈0
- omega (ω): Curvature (dθ/ds)

The system exhibits a bifurcation structure with:
- Lower branch: Stable solutions (physical)
- Upper branch: Unstable solutions (mathematical but unphysical)
- Turning point: Critical Ca where branches meet

## Important Notes

- GLE_solver.py is for solving at a specific Ca value (when Ca < Ca_critical)
- GLE_criticalCa.py finds Ca_critical using hybrid IQI/Newton/bisection refinement
- GLE_criticalCa_advanced.py provides more robust critical Ca finding with adaptive mesh refinement
- GLE_continuation_v4.py offers two methods:
  - Natural parameter: Fast, simple, finds critical Ca on stable branch
  - Simplified pseudo-arclength: More robust near folds but cannot trace unstable branch
- GLE_continuation_v4.5.py implements true pseudo-arclength continuation:
  - Solves extended system with arc-length constraint
  - Can trace through fold bifurcations
  - Captures complete S-shaped bifurcation curve
- Default parameters (Delta=10, ngrid=10000) work well for most cases
- For very small slip lengths (<1e-6), may need to increase grid resolution

## File Organization

### src-local/
Contains shared utility modules:
- `find_x0_utils.py`: Functions for finding x0 (position at θ_min) and θ_min from solutions
- `__init__.py`: Makes src-local a Python package

### Importing from src-local
Both solver files use:
```python
import sys
sys.path.append('src-local')
from find_x0_utils import find_x0_and_theta_min
```

## Known Issues & Debugging

### Running Large Computations
- Both GLE_solver.py and GLE_critical_ca_advanced.py can take significant time to run
- For testing, use the test scripts in test/ folder instead of running directly
- Monitor output for progress indicators
- The advanced finder shows detailed progress during execution

### Mesh Refinement
- If solutions are not converging well, GLE_critical_ca_advanced.py will automatically increase mesh density
- You can disable adaptive mesh with `--no-adaptive-mesh` flag