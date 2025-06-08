# GLE Continuation Solvers Documentation

This document provides comprehensive documentation for the continuation methods implemented in this project for solving the Generalized Lubrication Equations (GLE) and tracking contact line displacement.

## Overview

The continuation solvers track the relationship between the Capillary number (Ca) and the contact line displacement (δX_cl), revealing the bifurcation structure of the system. Two main implementations are provided:

1. **GLE_continuation_v4.py**: Unified continuation solver with both pseudo-arclength and natural parameter methods
2. **GLE_continuation_v4.5.py**: Extended pseudo-arclength solver with arc-length constraints (attempts to trace through folds)

### Visual Comparison

```
v4 behavior:              v4.5 intended behavior:

δX_cl                     δX_cl
  ^                         ^
  |    ___X (stops)         |    _____ (fold)
  |   /                     |   /     \
  | /                       | /       \ (unstable)
  |/                        |/         \
  +---------> Ca            +-----------> Ca
       Ca_cr                      Ca_cr

v4: Approaches fold         v4.5: Should trace S-curve
    then stops              (but currently fails near fold)
```

### Quick Decision Tree

```
What do you need?
│
├─ Just find critical Ca? ──────────────→ Use v4 with --method natural
│
├─ See approach to fold? ───────────────→ Use v4 with --method arclength
│
└─ Want full S-curve? ──────────────────→ Try v4.5 (see limitations below)
                                           └─ Currently fails near fold
```

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

**Note**: The current implementation uses a simplified approach:

1. **Simplified arc-length constraint**:
   - Uses scalar constraint on Ca only (not full extended system)
   - Solves: `(Ca - Ca_ref)·τ_Ca + (X_cl - X_cl_ref)·τ_X = ds`
   - Uses brentq/newton methods for robustness

2. **Newton iteration**:
   - Finds Ca satisfying arc-length constraint
   - Then solves BVP at that Ca
   - Avoids mesh compatibility issues

3. **Tangent computation**:
   - Uses only Ca and X_cl components (no solution norms)
   - Simplified but more stable

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
  - v4: `bifurcation_diagram_arclength.png` or `bifurcation_diagram_natural.png`
  - v4.5: `bifurcation_diagram_extended.png`
- **Data file**: Text file with Ca, X_cl, δX_cl, θ_min values
  - v4: `bifurcation_data_arclength.txt` or `bifurcation_data_natural.txt`
  - v4.5: `bifurcation_data_extended.txt`
- **Branch pickle**: Complete solution data for post-processing (in .gitignore)

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

### v4.5 Issues and Limitations
- **Cannot trace through fold**: Currently fails near the fold point
- **No unstable branch capture**: Stops before reaching the upper branch
- **Newton iteration failures**: Arc-length constraint becomes ill-conditioned near fold
- **Known issues**: See [Issue #14](https://github.com/comphy-lab/Contact-line-subgrid-modeling/issues/14) and [Issue #16](https://github.com/comphy-lab/Contact-line-subgrid-modeling/issues/16)
- **Current status**: v4.5 gets closer to Ca_cr than v4 but cannot continue past the fold

**Note**: Full pseudo-arclength continuation through folds remains an open challenge for this system

## Recommendations

1. **For critical Ca determination**: Use v4 with natural parameter method
2. **For approaching the fold**: Use v4 with arclength method
3. **For attempting full curve**: Try v4.5 but expect it to fail near fold
4. **For parameter studies**: Use v4 (more reliable)
5. **For debugging**: v4 is easier to understand and modify

## Computation Time Expectations

- **v4 natural parameter**: Typically completes in 10-30 seconds
- **v4 pseudo-arclength**: Typically completes in 30-60 seconds
- **v4.5 extended**: May take 1-5 minutes before failing near fold
- Times increase with smaller slip lengths or finer meshes

## References

The continuation methods are based on:
- Keller, H. B. (1977). "Numerical solution of bifurcation and nonlinear eigenvalue problems"
- Doedel, E. J. (1981). "AUTO: A program for the automatic bifurcation analysis of autonomous systems"