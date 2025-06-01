# Contact-line-subgrid-modeling

This repository contains implementations for modeling contact line dynamics, including Python and C versions of a Generalized Lubrication Equation (GLE) solver. The solvers use shooting methods to solve the coupled ODEs that describe the interface shape near a moving contact line.

## Python Implementation

The Python implementation provides tools to solve the GLE and analyze related phenomena:
- `GLE_solver.py` - Main solver using scipy's odeint
- `huh_scriven_velocity.py` - Analyzes Huh-Scriven velocity fields near contact lines
- `compare_results.py` - Compares outputs between Python and C implementations

### Dependencies
- Python 3.x
- NumPy
- SciPy
- Matplotlib
- pytest (for testing)

Install dependencies using:
```bash
pip install -r requirements-python.txt
```

### Running Tests (Python)
To run the Python unit tests:
```bash
# From the project root directory
cd test
sh run_tests.sh
```

Or run pytest directly:
```bash
pytest test/
```

## C Implementation (`GLE_solver-GSL`)

A C implementation of the GLE solver using the GNU Scientific Library (GSL) is provided. The solver uses an Initial Value Problem (IVP) approach with an enhanced shooting method that includes gradient descent optimization for robust convergence.

### Dependencies (C)
- A C compiler (e.g., GCC)
- GNU Scientific Library (GSL) version 2.5 or later
- OpenBLAS (or another BLAS implementation)

#### Installing GSL on macOS with Anaconda:
```bash
conda install -c conda-forge gsl
```

#### Installing GSL on Debian/Ubuntu:
```bash
sudo apt-get update
sudo apt-get install libgsl-dev
```

#### Installing GSL on Fedora/RHEL:
```bash
sudo dnf install gsl-devel
```

### Building the C Solver

The provided `Makefile` handles compilation with proper library paths and runtime linking:

```bash
# Build the solver
make

# Or explicitly
make all
```

This creates the `gle_solver_gsl` executable.

### Running the C Solver

```bash
# Run with default parameters (verbose output)
./gle_solver_gsl

# Run in quiet mode (minimal output)
./gle_solver_gsl --quiet
```

The solver uses the following default parameters:
- Capillary number (Ca): 1.00
- Slip length (λ): 1.00e-05
- Viscosity ratio (μ_r): 1.00e-03
- Initial contact angle (θ₀): 30 degrees
- Domain: s ∈ [0, 4.00e-04]

### Output Files

Both Python and C solvers generate output files in the `output/` directory:

**C Solver outputs:**
- `output/data-c-gsl.csv`: Contains columns for s (arc length), h (film height), and theta (contact angle)
- `output/GLE_h_profile_c.csv`: Contains s and h data
- `output/GLE_theta_profile_c.csv`: Contains s and theta_deg (angle in degrees)

**Python Solver outputs:**
- `output/data-python.csv`: Contains columns for s, h, and theta
- `output/GLE_h_profile.png`: Visualization of film height profile
- `output/GLE_theta_profile.png`: Visualization of contact angle profile
- `output/comparison_python_vs_c.png`: Side-by-side comparison (when using `make compare`)

### Solver Algorithm

The C implementation uses an IVP+shooting method approach:

**Initial Value Problem (IVP):**
- Solves the coupled ODEs: dh/ds = sin(θ), dθ/ds = ω, dω/ds = 3Ca·f(θ,μᵣ)/(h(h+3λ)) - cos(θ)
- Initial conditions: h(0) = λ (slip length), θ(0) = π/6 (30°)
- Uses GSL's adaptive Runge-Kutta-Fehlberg (4,5) integrator with tight tolerances

**Shooting Method:**
1. Searches for the correct initial value ω₀ such that ω(s_max) = 0
2. First attempts exponential search to bracket the solution
3. If bracketing succeeds, uses Brent's method for root finding
4. If bracketing fails, automatically switches to gradient descent optimization
5. Gradient descent uses adaptive learning rates and line search
6. Achieves convergence tolerance of 1e-8 for the boundary condition residual

**Note:** The code originally included provisions for a Boundary Value Problem (BVP) solver, but since GSL doesn't provide BVP functionality, only the IVP+shooting method is implemented.

### Comparing C and Python Results

To run both solvers and compare outputs:
```bash
make compare
```

This will:
1. Run the C solver
2. Run the Python solver
3. Execute `compare_results.py` to generate comparison plots
4. Place all outputs in the `output/` directory

The comparison script generates visualizations showing the agreement between the two implementations.

### Running Tests

```bash
# Run C tests only
make test

# Run all tests (Python and C)
sh test/run_tests.sh
```

### Troubleshooting

#### Library Loading Errors on macOS
If you encounter errors like:
```
dyld: Library not loaded: @rpath/libgsl.25.dylib
```

The Makefile already includes runtime path fixes, but if issues persist:
1. Ensure GSL is installed via conda or homebrew
2. Check that the library paths in the Makefile match your installation
3. For conda users, ensure your environment is activated
4. Consider creating symbolic links for missing libraries (e.g., libcblas.3.dylib)

#### Build Messages
During compilation, you may see:
```
GSL BVP support not detected - solver will use fallback implementation
```
This is normal and can be ignored - the solver uses the IVP+shooting method which doesn't require BVP functionality.

### Cleaning Build Artifacts

```bash
make clean
```

This removes:
- Compiled executables
- Object files in `build/`
- Output CSV files

### Available Make Targets

```bash
make help
```

Shows all available targets:
- `make all` - Build the solver
- `make test` - Build and run tests
- `make run` - Run the solver
- `make compare` - Run both C and Python solvers
- `make clean` - Clean build artifacts
- `make help` - Show help message

## Mathematical Background

The solvers implement the Generalized Lubrication Equations (GLE) for contact line dynamics:
- The system describes the interface shape h(s) and angle θ(s) along arc length s
- Key parameters: Capillary number (Ca), slip length (λ), viscosity ratio (μ_r)
- Boundary conditions enforce matching to the macroscopic contact angle
- Both solvers use shooting methods to find the correct initial conditions