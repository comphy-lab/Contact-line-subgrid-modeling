# Contact-line-subgrid-modeling

This repository contains implementations for modeling contact line dynamics, including Python and C versions of a Generalized Lubrication Equation (GLE) solver.

## Python Implementation

The Python implementation (`GLE_solver.py`, `huh_scriven_velocity.py`) provides tools to solve the GLE and analyze related phenomena.

### Dependencies
- Python 3.x
- NumPy
- SciPy
- Matplotlib (for plotting, if used)

Install dependencies using:
```bash
pip install -r requirements-python.txt
```

### Running Tests (Python)
To run the Python unit tests:
```bash
sh test/run_tests.sh
```
(This will also attempt to run C tests if compiled).

## C Implementation (`GLE_solver-GSL`)

A C implementation of the GLE solver using the GNU Scientific Library (GSL) is provided. The solver uses an enhanced shooting method with gradient descent optimization for robust convergence.

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

The solver generates a CSV file in the `output/` directory:
- `output/data-c-gsl.csv`: Contains columns for s (arc length), h (film height), and theta (contact angle)

Additionally, two separate files are created for compatibility:
- `output/GLE_h_profile_c.csv`: Contains s and h data
- `output/GLE_theta_profile_c.csv`: Contains s and theta_deg (angle in degrees)

### Solver Algorithm

The C implementation uses an enhanced shooting method that:
1. First attempts traditional bracketing to find the initial value ω₀
2. If bracketing fails, automatically switches to gradient descent optimization
3. Uses adaptive learning rates and line search for robust convergence
4. Achieves convergence tolerance of 1e-8 for the boundary condition residual

### Comparing C and Python Results

To run both solvers and compare outputs:
```bash
make compare
```

This will:
1. Run the C solver
2. Run the Python solver
3. Place outputs in the `output/` directory for comparison

### Running Tests

```bash
# Run C tests only
make test

# Run all tests (Python and C)
sh test/run_tests.sh
```

### Troubleshooting

If you encounter library loading errors on macOS:
```
dyld: Library not loaded: @rpath/libgsl.25.dylib
```

The Makefile already includes runtime path fixes, but if issues persist:
1. Ensure GSL is installed via conda or homebrew
2. Check that the library paths in the Makefile match your installation
3. Consider creating symbolic links for missing libraries (e.g., libcblas.3.dylib)

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