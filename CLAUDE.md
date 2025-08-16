# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements dual-language (Python and C) solutions for contact line subgrid modeling in thin liquid films. The project addresses the fundamental "contact line singularity" problem in fluid mechanics through a multiscale approach that incorporates molecular-scale physics into continuum models.

## Architecture Overview

The codebase consists of two complete implementations that solve the same mathematical problem using different numerical approaches:

### Python Implementation (Development/Prototyping)
- **Primary files**: `GLE_solver.py`, `GLE_solver_v2.py` (uses different numerical method)
- **Method**: Initial Value Problem (IVP) using scipy's `odeint` and `solve_bvp`
- **Purpose**: Rapid prototyping, parameter exploration, visualization

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
python GLE_solver.py           # Original IVP approach
python GLE_solver_v2.py        # BVP approach with solve_bvp
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

# Run tests
make test           # C unit tests only
./test/run_tests.sh # Both Python and C tests

# Clean build artifacts
make clean
```

### Testing Framework
```bash
# Python tests with pytest
pytest test/                    # From project root
cd test && pytest -v          # Verbose output

# All tests (comprehensive)
./test/run_tests.sh            # Runs both Python and C test suites
```

## Key Components

### Core Solvers

**GLE_solver.py**: Original Python implementation
- Uses `scipy.integrate.odeint` for initial value problems
- Functions: `f1(theta)`, `f2(theta)`, `f3(theta)`, `f(theta, R)`, `GLE(y)`
- Known convergence issues for certain parameter regimes

**GLE_solver-GSL.c**: High-performance C implementation  
- Adaptive Runge-Kutta-Fehlberg integration via GSL
- Enhanced shooting method with bracketing and gradient descent fallback
- Robust handling of singularities and boundary conditions
- Outputs compatible CSV files for comparison

**GLE_solver_v2.py**: Alternative Python approach
- Uses `scipy.integrate.solve_bvp` for boundary value problems
- Different parameter set (Ca=0.0246 vs Ca=1.0)
- May have better convergence properties

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
dω/ds = 3Ca·f(θ,μᵣ)/(h(h+3λ)) - cos(θ)          
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

### Boundary Conditions
- Contact line: `θ(0) = θ₀`, `h(0) = λ`
- Far field: `ω(s_max) = 0` (curvature vanishes)

## File Organization

```
├── GLE_solver.py, GLE_solver_v2.py    # Python implementations
├── GLE_solver-GSL.c                   # C main program
├── huh_scriven_velocity.py            # Velocity field analysis
├── compare_results.py                 # Cross-validation tool
├── src-local/                         # C header library
│   ├── GLE_solver-GSL.h              # Main definitions
│   ├── gle_physics.h                 # Physical functions
│   ├── gle_ode_systems.h             # ODE systems
│   ├── gle_shooting.h                # Numerical algorithms
│   └── gle_io.h                      # I/O operations
├── test/                             # Test suites
│   ├── test_*.py                     # Python unit tests
│   ├── test_*.c                      # C unit tests  
│   └── run_tests.sh                  # Test orchestration
├── output/                           # Generated results
└── build/                            # C build artifacts
```

## Dependencies

**Python**: numpy, scipy, matplotlib, pytest
**C**: GSL (≥2.5), OpenBLAS, standard C99 compiler

## Known Issues and Limitations

- **Python solver convergence**: Original `GLE_solver.py` has convergence problems for certain parameter combinations
- **Integration domain**: Current s_range may need adjustment for different physical regimes  
- **Parameter sensitivity**: Solutions sensitive to initial guesses in shooting method
- **Validation**: Results need comparison against analytical/experimental benchmarks

## Common Development Patterns

- **Cross-validation**: Always compare Python and C results using `make compare`
- **Parameter studies**: Modify physical constants in headers/global variables
- **Debugging**: Use test suites to isolate issues before full solver runs
- **Performance**: C implementation for production, Python for rapid prototyping