# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains Python implementations for contact line subgrid modeling, focusing on solving the Generalized Lubrication Equations (GLE) and analyzing Huh-Scriven velocity fields near moving contact lines in fluid dynamics. The project includes advanced bifurcation analysis tools to study both stable and unstable solution branches.

## Recent Updates

### x0 Definition (Updated)
- x0 is now consistently defined as the x-position where θ reaches its minimum value
- This definition is implemented in `src-local/find_x0_utils.py`
- Both GLE_solver.py and GLE_continuation_hybrid.py import this shared utility
- When θ_min approaches 0, we are at the critical capillary number (Ca_cr)

## Key Components

### GLE_solver.py
- Solves coupled ODEs for contact line dynamics using the Generalized Lubrication Equations
- Uses scipy's solve_bvp with optimized parallel bisection refinement
- Key functions:
  - `f1(theta)`, `f2(theta)`, `f3(theta)`: Helper functions for the GLE formulation
  - `f(theta, mu_r)`: Main function combining the helpers with viscosity ratio mu_r
  - `GLE(s, y, Ca, mu_r, lambda_slip)`: System of ODEs defining the contact line shape evolution
  - `find_critical_ca_lower_branch()`: Continuation method to find critical Capillary number
  - `adaptive_bisection_refinement()`: Parallel bisection with adaptive tolerances
- Features adaptive mesh refinement, solution caching, and automatic Ca_cr detection

### GLE_continuation_hybrid.py
- Advanced bifurcation analysis tool that captures both solution branches
- Uses hybrid approach combining two methods:
  - Phase 1: x0 parameterization with parallel processing for lower branch
  - Phase 2: Pseudo-arclength continuation for upper branch
- Key classes and functions:
  - `PseudoArclengthContinuation`: Implements arc-length continuation
  - `solve_for_x0_newton()`: Newton's method for x0-based continuation
  - `worker_solve_x0()`: Parallel worker for efficient computation
  - `trace_both_branches_hybrid()`: Main function orchestrating the hybrid approach
- Generates complete bifurcation diagrams showing stable and unstable branches

### huh_scriven_velocity.py
- Analyzes the Huh-Scriven velocity field near moving contact lines
- Computes velocity components in polar (Ur, Uphi) and Cartesian (Ux, Uy) coordinates
- Visualizes relative velocities between the fluid and moving plate

## Development Commands

```bash
# Run the GLE solver (finds critical Ca for lower branch)
python GLE_solver.py --ca 0.05 --mu-r 1e-6 --lambda-slip 1e-4

# Run the continuation solver (traces both branches)
python GLE_continuation_hybrid.py --mu-r 1e-6 --lambda-slip 1e-4

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
- omega (ω): Curvature (dθ/ds)

The system exhibits a bifurcation structure with:
- Lower branch: Stable solutions (physical)
- Upper branch: Unstable solutions (mathematical but unphysical)
- Turning point: Critical Ca where branches meet

## Important Notes

- The GLE_solver.py automatically finds Ca_critical when requested Ca exceeds it
- GLE_continuation_hybrid.py can trace through turning points to capture upper branch
- Default parameters (Delta=10, ngrid=10000) work well for most cases
- For very small slip lengths (<1e-6), may need to increase grid resolution
- The hybrid solver uses parallel processing - adjust --workers based on your system

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

### Phase 2 in GLE_continuation_hybrid.py
If Phase 2 (pseudo-arclength continuation) appears stuck:
- The turning point is likely very close to the critical Ca
- Solutions near the turning point are difficult to compute
- Consider adjusting the initial step size (ds) or tolerance
- Debug output has been added to track progress:
  - Tangent computation status
  - Predictor step values
  - Corrector iteration progress

### Running Large Computations
- Both GLE_solver.py and GLE_continuation_hybrid.py can take significant time to run
- For testing, use the test scripts in test/ folder instead of running directly
- Monitor output for progress indicators
- Use `--workers` flag to adjust parallel processing (default: 4 or CPU count)