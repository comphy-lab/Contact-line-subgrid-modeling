# GLE Continuation Solvers Documentation

This document provides comprehensive documentation for the continuation methods implemented in this project for solving the Generalized Lubrication Equations (GLE) and tracking contact line displacement.

## Overview

The continuation solvers track the relationship between the Capillary number (Ca) and the contact line displacement (δX_cl), revealing the bifurcation structure of the system. Two main implementations are provided:

1. **GLE_continuation_v4.py**: Unified continuation solver with both pseudo-arclength and natural parameter methods
2. **GLE_continuation_v4.5.py**: Extended pseudo-arclength solver with true arc-length constraints for tracing through folds

## Key Concepts

### Physical Parameters
- **Ca**: Capillary number (ratio of viscous to surface tension forces)
- **X_cl**: Contact line position, defined as x-coordinate at s=Δ (end of domain)
- **δX_cl**: Contact line displacement = X_cl(Ca) - X_cl(Ca≈0)
- **θ_min**: Minimum contact angle along the interface
- **x0**: Position where θ reaches its minimum value

### Bifurcation Structure
The system exhibits a fold bifurcation with:
- **Lower branch**: Stable solutions (physical)
- **Upper branch**: Unstable solutions (mathematical but unphysical)
- **Fold point**: Critical Ca where branches meet

## GLE_continuation_v4.py

### Features
- **Dual methods**: Supports both pseudo-arclength and natural parameter continuation
- **Adaptive step size**: Automatically adjusts step size based on convergence
- **Fold detection**: Identifies turning points in the solution branch
- **Simplified implementation**: Uses standard BVP solver at each continuation step

### Methods

#### Natural Parameter Continuation
- Increments Ca directly
- Simple and fast for stable branches
- Cannot trace through fold bifurcations
- Suitable for finding critical Ca on the stable branch

#### Simplified Pseudo-Arclength Continuation
- Uses tangent prediction but simplified correction
- More robust near fold points than natural parameter
- Cannot fully trace through folds to unstable branch
- Good balance of robustness and simplicity

### Usage

```bash
# Natural parameter method (default)
python GLE_continuation_v4.py --method natural --mu_r 1e-6 --lambda_slip 1e-3 --theta0 60

# Simplified pseudo-arclength method
python GLE_continuation_v4.py --method arclength --mu_r 1e-6 --lambda_slip 1e-3 --theta0 60
```

### Algorithm Details

The simplified pseudo-arclength in v4 uses:
1. **Predictor**: `Ca_new = Ca_old + ds * tangent_Ca`
2. **Corrector**: Solve BVP at predicted Ca (not the full extended system)
3. **Tangent update**: Computed from last two points

This approach is more stable than natural parameter but cannot trace the full S-curve.

## GLE_continuation_v4.5.py

### Features
- **True pseudo-arclength**: Solves extended system with arc-length constraint
- **Traces through folds**: Can follow unstable branch after fold bifurcation
- **S-shaped curves**: Captures complete bifurcation diagram
- **Newton iteration**: Solves coupled system for Ca and solution simultaneously

### Extended System

The extended pseudo-arclength method solves:
```
F(Ca, y) = 0  (BVP equations)
g(Ca, y) = 0  (Arc-length constraint)
```

Where the arc-length constraint is:
```
(Ca - Ca_ref)·τ_Ca + (X_cl - X_cl_ref)·τ_X = ds
```

### Algorithm Details

1. **Extended system construction**:
   - ODE residuals from GLE
   - Boundary condition residuals
   - Arc-length constraint equation

2. **Newton iteration**:
   - Solves for Ca and solution y simultaneously
   - Uses finite differences for Jacobian approximation
   - Converges to solution satisfying arc-length constraint

3. **Tangent computation**:
   - Includes Ca, X_cl, and solution norm components
   - Properly normalized for arc-length parameterization

### Usage

```bash
# Extended pseudo-arclength for full bifurcation diagram
python GLE_continuation_v4.5.py --mu_r 1e-6 --lambda_slip 1e-3 --theta0 60 --max-steps 200
```

### Expected Output

The v4.5 solver should produce:
- S-shaped bifurcation curve in δX_cl vs Ca plot
- Stable branch (solid line) up to fold point
- Unstable branch (dashed line) after fold point
- Clear fold point marking

## Comparison: v4 vs v4.5

| Feature | v4 (Simplified) | v4.5 (Extended) |
|---------|-----------------|-----------------|
| Implementation complexity | Simple | Complex |
| Computational cost | Low | High |
| Traces through folds | No | Yes |
| Captures unstable branch | No | Yes |
| Robustness | High | Moderate |
| Best use case | Finding critical Ca | Complete bifurcation analysis |

## Output Files

Both solvers generate:
- **Bifurcation diagram**: PNG plot showing δX_cl vs Ca
- **Data file**: Text file with Ca, X_cl, δX_cl, θ_min values
- **Branch pickle**: Complete solution data for post-processing

## Physical Interpretation

### Contact Line Displacement
- δX_cl > 0: Contact line advances (meniscus stretches)
- δX_cl increases with Ca on stable branch
- At fold: Maximum sustainable Ca before solution ceases to exist
- Unstable branch: Mathematical solutions, not physically realizable

### Critical Capillary Number
- Ca_cr: Maximum Ca on the lower (stable) branch
- Beyond Ca_cr: No steady-state solution exists
- System transitions to dynamic behavior (not captured by steady solver)

## Troubleshooting

### v4 Issues
- **Stuck at fold**: Expected behavior - use v4.5 for full curve
- **Convergence failures**: Reduce initial_ds or increase tolerance

### v4.5 Issues
- **Newton iteration failures**: Extended system is more sensitive
- **Slow convergence**: Normal due to coupled system size
- **Memory issues**: Reduce max_nodes or use coarser initial mesh

## Recommendations

1. **For critical Ca determination**: Use v4 with natural parameter method
2. **For publication-quality bifurcation diagrams**: Use v4.5
3. **For parameter studies**: Start with v4, use v4.5 for selected cases
4. **For debugging**: v4 is easier to understand and modify

## References

The continuation methods are based on:
- Keller, H. B. (1977). "Numerical solution of bifurcation and nonlinear eigenvalue problems"
- Doedel, E. J. (1981). "AUTO: A program for the automatic bifurcation analysis of autonomous systems"