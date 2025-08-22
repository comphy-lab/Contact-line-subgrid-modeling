# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements dual-language (Python and C) solutions for contact line subgrid modeling in thin liquid films. The project addresses the fundamental "contact line singularity" problem in fluid mechanics through a multiscale approach that incorporates molecular-scale physics into continuum models.

## Architecture Overview

The codebase consists of two complete implementations that solve the same mathematical problem using different numerical approaches:

### Python Implementation (Development/Prototyping)
- **Primary files**: `GLE_solver.py` (vertical plate), `GLE_solver_v3.py` (horizontal plate), `GLE_solver_v4.py` (h-based BC)
- **Method**: Boundary Value Problem (BVP) using scipy's `solve_bvp`
- **Purpose**: Rapid prototyping, parameter exploration, visualization
- **Variants**: Different geometries and boundary condition formulations

### C Implementation (Performance/Production) 
- **Primary file**: `GLE_solver-GSL.c` 
- **Method**: IVP + shooting method with GSL (GNU Scientific Library)
- **Architecture**: Modular header-only design in `src-local/`:
  - `GLE_solver-GSL.h`: Main constants and structures
  - `gle_physics.h`: Physical functions and viscous dissipation calculations
  - `gle_ode_systems.h`: ODE system definitions for GSL
  - `gle_shooting.h`: Shooting method with gradient descent fallback
  - `gle_io.h`: File I/O operations
- **Features**: Enhanced convergence, adaptive integration, robust error handling

## Development Commands

### Python Environment Setup
```bash
# Install dependencies
pip install -r requirements-python.txt

# Run Python solvers
python GLE_solver.py           # Vertical plate withdrawal (BVP)
python GLE_solver_v3.py        # Horizontal plate immersion (BVP)
python GLE_solver_v4.py        # H-based boundary condition (iterative BVP)
python huh_scriven_velocity.py # Velocity field analysis
```

### C Build System
```bash
# Build C solver (requires GSL)
make                # or make all

# Run C solver
make run            # Runs ./gle_solver_gsl

# Compare implementations
make compare        # Runs both C and Python, generates comparison plots

# Clean build artifacts
make clean
```


## Key Components

### Core Solvers

**GLE_solver.py**: Vertical plate withdrawal
- Physical setup: Plate withdrawn vertically from liquid bath
- Gravity term: `-cos(θ)/l_cap²` in momentum equation
- Boundary conditions: `θ(s_max) = 90°`, `ω(s_max) = 0`
- Uses `scipy.integrate.solve_bvp` for boundary value problem

**GLE_solver_v3.py**: Horizontal plate immersion  
- Physical setup: Horizontal plate immersed into liquid bath
- Gravity term: `+sin(θ)/l_cap²` in momentum equation (different geometry)
- Boundary conditions: `θ(s_max) = 90°` (vertical interface at far field)
- Different parameter set optimized for horizontal geometry

**GLE_solver_v4.py**: H-based boundary condition
- Same horizontal geometry as v3
- Novel BC: `θ(h=h_end) = θ_end` instead of fixed arc length
- Uses iterative shooting method with root finding
- More physically meaningful for coating applications where film thickness is controlled

**GLE_solver-GSL.c**: High-performance C implementation  
- Adaptive Runge-Kutta-Fehlberg integration via GSL
- Enhanced shooting method with bracketing and gradient descent fallback
- Robust handling of singularities and boundary conditions
- Outputs compatible CSV files for comparison

### Analysis Tools

**huh_scriven_velocity.py**: Velocity field analysis
- Computes Huh-Scriven velocity components (polar and Cartesian)
- Visualizes relative velocities between fluid and moving plate
- Essential for understanding flow structure near contact lines

**compare_results.py**: Cross-validation tool
- Generates side-by-side comparison plots between C and Python
- Validates numerical accuracy across implementations
- Outputs comparison plots to `output/` directory

### Modular C Architecture

The C implementation uses a header-only design pattern for modularity:

- **Physics layer** (`gle_physics.h`): Mathematical functions, dissipation calculations
- **Integration layer** (`gle_ode_systems.h`): ODE definitions for GSL integration  
- **Algorithm layer** (`gle_shooting.h`): Shooting method, optimization algorithms
- **I/O layer** (`gle_io.h`): File operations, data output formatting

## Mathematical Context

### Physical Problem
The Generalized Lubrication Equations (GLE) system:
```
dh/ds = sin(θ)                                    
dθ/ds = ω                                         
dω/ds = 3Ca·f(θ,μᵣ)/(h(h+3λ)) ± gravity_term          
```

Where:
- `s`: Arc length coordinate along interface
- `h(s)`: Film thickness profile  
- `θ(s)`: Interface angle with substrate
- `ω(s)`: Interface curvature (dθ/ds)
- `Ca`: Capillary number (viscous/surface tension forces)
- `λ`: Slip length (molecular scale parameter)
- `μᵣ`: Viscosity ratio (gas/liquid)
- `f(θ,μᵣ)`: Viscous dissipation function from wedge flow analysis
- `gravity_term`: `-cos(θ)/l_cap²` (vertical) or `+sin(θ)/l_cap²` (horizontal)

### Boundary Condition Variants
**Standard (v1, v3)**: 
- Contact line: `h(λ_slip) = λ_slip`, `θ(λ_slip) = θ₀`
- Far field: `ω(s_max) = 0` or `θ(s_max) = θ_end`

**H-based (v4)**:
- Contact line: `h(λ_slip) = λ_slip`, `θ(λ_slip) = θ₀` 
- Target thickness: `θ(h=h_end) = θ_end` (requires iterative domain finding)

## File Organization

```
├── GLE_solver.py                     # Vertical plate withdrawal
├── GLE_solver_v3.py                  # Horizontal plate immersion  
├── GLE_solver_v4.py                  # H-based boundary conditions
├── GLE_solver-GSL.c                  # C main program
├── huh_scriven_velocity.py           # Velocity field analysis
├── compare_results.py                # Cross-validation tool
├── src-local/                        # C header library
│   ├── GLE_solver-GSL.h             # Main definitions
│   ├── gle_physics.h                # Physical functions
│   ├── gle_ode_systems.h            # ODE systems
│   ├── gle_shooting.h               # Numerical algorithms
│   └── gle_io.h                     # I/O operations
├── output/                          # Generated results
└── build/                           # C build artifacts
```

## Dependencies

**Python**: numpy, scipy, matplotlib
**C**: GSL (≥2.5), OpenBLAS, standard C99 compiler

## Solver Selection Guide

**Choose based on physical problem:**
- **GLE_solver.py**: Vertical coating processes (dip coating, plate withdrawal)
- **GLE_solver_v3.py**: Horizontal wetting processes (immersion, advancing contact lines)  
- **GLE_solver_v4.py**: When film thickness `h_end` is the controlled parameter
- **GLE_solver-GSL.c**: High-performance computations or parameter studies

**Parameter considerations:**
- Different solvers use different parameter sets (Ca, λ_slip, l_cap)
- v4 requires careful choice of `h_end` and `s_max_initial` for convergence
- Gravity terms have opposite signs for vertical vs horizontal geometries

## Known Issues and Limitations

- **Parameter sensitivity**: Solutions sensitive to initial guesses, especially in v4's iterative method
- **Integration domain**: s_range may need adjustment for different physical regimes  
- **Convergence**: v4's root-finding may fail if h_end is too large or domain bounds are inappropriate
- **Validation**: Results need comparison against analytical/experimental benchmarks

## Common Development Patterns

- **Cross-validation**: Always compare Python and C results using `make compare`
- **Parameter studies**: Modify physical constants in headers/global variables
- **Debugging**: Use direct solver runs to debug numerical issues
- **Performance**: C implementation for production, Python for rapid prototyping