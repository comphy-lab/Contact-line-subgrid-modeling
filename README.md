# Contact-line-subgrid-modeling

This repository contains implementations for modeling contact line dynamics in thin liquid films, including Python and C versions of a Generalized Lubrication Equation (GLE) solver. The code addresses a fundamental problem in fluid dynamics: how to model the motion of a contact line (where liquid, gas, and solid meet) across multiple length scales.

## Scientific Background

The contact line problem is a classic challenge in fluid mechanics. Standard continuum models predict infinite viscous dissipation at a moving contact line - the famous "contact line singularity." This solver implements a multiscale approach that resolves this singularity by incorporating molecular-scale physics (slip) into the continuum description.

### Physical Problem

Consider a thin liquid film spreading on a solid substrate:
- **Microscale** (~nm): Molecular interactions dominate, allowing slip at the wall
- **Mesoscale** (~μm): Viscous forces balance surface tension in a wedge flow
- **Macroscale** (~mm): Lubrication approximation describes the thin film

The GLE provide the mathematical framework to connect these scales seamlessly.

## Project Structure

### Core Implementation Files

- **`GLE_solver.py`**: Python implementation of the GLE solver using scipy's odeint
- **`GLE_solver-GSL.c`**: Main C program entry point with comprehensive documentation
- **`huh_scriven_velocity.py`**: Analyzes Huh-Scriven velocity fields near contact lines
- **`compare_results.py`**: Compares outputs between Python and C implementations

### Header Files (src-local/)

All modules are implemented as header-only libraries with static inline functions:

- **`GLE_solver-GSL.h`**: Main header with constants, structures, and function declarations
- **`gle_physics.h`**: Physical functions including the viscous dissipation function f(θ,μᵣ)
- **`gle_ode_systems.h`**: ODE system definitions for GSL integration
- **`gle_shooting.h`**: Shooting method implementation with optimization algorithms
- **`gle_io.h`**: Input/output operations for data files
- **`gle_optimization.h`**: Stub header for backward compatibility

### Test Files

- **`test/test_GLE_solver.py`**: Python unit tests
- **`test/test_GLE_solver-GSL.c`**: C unit tests with colorful output
- **`test/test_integration.py`**: Integration tests comparing C and Python results
- **`test/test_huh_scriven_velocity.py`**: Tests for velocity field analysis
- **`test/run_tests.sh`**: Shell script to run all tests

### Build System

- **`Makefile`**: Build configuration for C implementation
- **`requirements-python.txt`**: Python dependencies
- **`pytest.ini`**: Pytest configuration

## Python Implementation

The Python implementation provides tools to solve the GLE and analyze related phenomena:
- `GLE_solver.py` - Main solver using scipy's solve_bvp with continuation method for extreme parameter values
- `huh_scriven_velocity.py` - Analyzes Huh-Scriven velocity fields near contact lines
- `compare_results.py` - Compares outputs between Python and C implementations

### Command-line Usage

The Python solver accepts command-line parameters:
```bash
python GLE_solver.py --ca 0.01 --theta0 60 --delta 2.0 --lambda-slip 1e-5 --mu-r 1e-6
```

Parameters:
- `--ca`: Capillary number (default: 0.0246)
- `--theta0`: Initial contact angle in degrees (default: 90)
- `--delta`: Maximum s-value for solver (default: 1.0)
- `--lambda-slip`: Slip length (default: 1e-4)
- `--mu-r`: Viscosity ratio μ_g/μ_l (default: 1e-6)
- `--w`: Curvature boundary condition at s=Δ (default: 0)
- `--ngrid-init`: Initial number of grid points (default: 10000)

### Solver Details

The Python solver uses `scipy.solve_bvp` with the following features:

1. **Grid Resolution**: The default grid uses 10000 points to properly capture steep gradients near the contact line, especially for small slip lengths. This can be adjusted via `--ngrid-init`.

2. **Critical Capillary Number**: The solver automatically detects the critical Capillary number (Ca_cr) - the maximum Ca for which a steady-state solution exists. This critical value depends on:
   - Initial contact angle (θ₀)
   - Slip length (λ_slip) 
   - Viscosity ratio (μ_r)
   
   When the requested Ca exceeds Ca_cr:
   - The solver finds Ca_cr using a continuation method
   - Returns the solution at Ca_cr (not at the requested Ca)
   - Clearly indicates in the output and plots that Ca_cr is being used
   - The plot title and parameter box show Ca_cr with a red highlight

3. **Continuation Method**: The continuation method is implemented as a modular function `find_critical_ca_continuation()`:
   - Activated automatically when direct solve fails
   - Starts with small Ca (0.0001) and increases logarithmically through 50 steps
   - Keeps μ_r and λ_slip fixed during continuation
   - Stops when convergence fails, identifying Ca_cr
   - Returns both the solution and the actual Ca used (either target Ca or Ca_cr)

4. **Convergence Tips**: For challenging parameter regimes:
   - The default 10000 grid points usually ensures convergence for Ca < Ca_cr
   - For very small λ_slip (< 1e-6), consider increasing --ngrid-init
   - The critical Ca typically ranges from 0.001 to 0.1 depending on parameters

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

### Mathematical Formulation

The solver addresses the Generalized Lubrication Equations (GLE), a system of coupled ODEs:

```
dh/ds = sin(θ)                                    ... (1)
dθ/ds = ω                                         ... (2)
dω/ds = 3Ca·f(θ,μᵣ)/(h(h+3λ)) - cos(θ)          ... (3)
```

Where:
- `s`: Arc length coordinate along the liquid-gas interface
- `h(s)`: Film thickness profile
- `θ(s)`: Local interface angle with the substrate
- `ω(s)`: Interface curvature (dθ/ds)
- `Ca`: Capillary number (ratio of viscous to surface tension forces)
- `λ`: Navier slip length (molecular-scale parameter)
- `μᵣ`: Viscosity ratio (gas/liquid)
- `f(θ,μᵣ)`: Viscous dissipation function from wedge flow analysis

### Boundary Conditions

1. **At the contact line (s = 0)**:
   - `θ(0) = θ₀` (microscopic contact angle, default 30°)
   - `h(0) = λ` (film height equals slip length)

2. **In the far field (s = s_max)**:
   - `ω(s_max) = 0` (curvature vanishes, matching outer solution)

### Numerical Algorithm

The implementation uses an IVP+shooting method approach:

#### 1. Initial Value Problem (IVP) Setup
- Convert the boundary value problem to an IVP by guessing ω₀ = ω(0)
- Integrate from s = 0 to s = s_max using the guessed ω₀
- Check if the far-field boundary condition ω(s_max) = 0 is satisfied

#### 2. Shooting Method
The shooting method finds the correct ω₀ by solving R(ω₀) = 0, where R(ω₀) = ω(s_max; ω₀):

**Phase 1: Bracketing**
- Start with initial guess ω₀ ≈ 82150 (empirically determined)
- Use exponential search to find [ω₀_low, ω₀_high] where R changes sign
- Double the search width iteratively until bracket is found

**Phase 2: Root Finding**
- If bracketing succeeds: Use Brent's method (guaranteed convergence)
- If bracketing fails: Switch to gradient descent optimization
- Convergence criterion: |R(ω₀)| < 1e-8

#### 3. Integration Details
- Method: Adaptive Runge-Kutta-Fehlberg (4,5) from GSL
- Tolerances: 1e-10 (relative), 1e-12 (absolute)
- Step size: Automatically adapted based on local error estimates

#### 4. Solution Generation
Once ω₀ is found:
- Re-integrate the ODEs with the converged ω₀
- Store solution at 1000 points uniformly distributed in s ∈ [0, s_max]
- Output: Arrays of s, h(s), and θ(s)

### Physical Constraints and Singularity Handling

1. **Film thickness**: h > 0 (enforced during integration)
2. **Interface angle**: 0 < θ < π (clamped to avoid singularities)
3. **Dissipation function**: f(θ,μᵣ) has singularities at θ = 0, π
   - Handled by clamping: θ ∈ [1e-10, π - 1e-10]
   - Returns large finite values near singularities

### Gradient Descent Fallback

When bracketing fails, the solver uses gradient descent to minimize |R(ω₀)|²:
- Numerical gradient: ∇R ≈ [R(ω₀ + ε) - R(ω₀)]/ε
- Adaptive learning rate with line search
- Maximum 200 iterations
- Typically converges within 50-100 iterations

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

## Physical Interpretation of Results

The solution profiles h(s) and θ(s) reveal the multiscale structure:

1. **Near contact line (s → 0)**:
   - h ~ s (linear growth from slip length)
   - θ remains close to θ₀ (microscopic angle)
   - Strong curvature ω indicates rapid adjustment

2. **Intermediate region**:
   - Transition from molecular to continuum physics
   - Interface angle θ increases toward macroscopic value
   - Curvature ω decreases

3. **Far field (s → s_max)**:
   - h grows more slowly
   - θ approaches asymptotic value
   - ω → 0 (matches outer lubrication solution)

The parameter f(θ,μᵣ) encodes how viscous dissipation depends on the local interface configuration and viscosity contrast.

## Applications

This model is relevant for:
- Coating and printing processes
- Microfluidics and lab-on-chip devices
- Enhanced oil recovery
- Understanding dewetting and film stability
- Biomedical applications (tear films, lung surfactants)

## References

The mathematical formulation follows:
- Huh & Scriven (1971): Original wedge flow analysis
- Cox (1986): Asymptotic matching for moving contact lines
- Snoeijer & Andreotti (2013): Review of moving contact line physics