# C Implementation of GLE Solver using SUNDIALS

This directory contains a C implementation of the Generalized Lubrication Equation (GLE) solver using the SUNDIALS library, addressing issue #3.

## Overview

The C implementation provides:
- Modular mathematical functions (`gle_math.h/c`)
- SUNDIALS-based BVP solver (`gle_solver.h/c`, `GLE_solver-SUNDIALS.c`)
- CSV output functionality (`csv_output.h/c`)
- Comprehensive unit tests (`test/c_tests/`)
- Main executable (`main.c`)

## Dependencies

### Required Libraries
- **SUNDIALS**: Scientific computing library for ODE/DAE solving
  ```bash
  # Ubuntu/Debian
  sudo apt-get install libsundials-dev
  
  # Or build from source: https://sundials.readthedocs.io/
  ```

- **CMake**: Build system (version 3.10 or higher)
  ```bash
  sudo apt-get install cmake
  ```

## Building

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build the project
make

# This creates:
# - gle_solver_main: Main executable
# - test_gle_math_c: Mathematical function tests
# - test_gle_solver_c: Solver integration tests
```

## Usage

### Running the Solver
```bash
# Basic usage with default parameters
./gle_solver_main

# Custom parameters
./gle_solver_main --output-dir results --ca 2.0 --mu-r 1e-2 --theta0 45 --points 2000

# See all options
./gle_solver_main --help
```

### Output Files
The solver generates CSV files instead of PNG images:
- `gle_h_profile.csv`: Film thickness h(s) vs arc length s
- `gle_theta_profile.csv`: Contact angle θ(s) vs arc length s  
- `gle_complete_solution.csv`: Complete solution data

### Running Tests
```bash
# Run all tests (Python + C)
bash test/run_tests.sh

# Run only C tests
bash test/run_c_tests.sh

# Run individual test executables
./test_gle_math_c
./test_gle_solver_c
```

## Implementation Details

### Mathematical Functions
- Direct C translations of Python functions f1, f2, f3, and f_combined
- Robust error handling for division by zero and invalid parameters
- Comprehensive unit tests against known values

### SUNDIALS Integration
- Uses CVODE for initial value problems in multiple shooting approach
- KINSOL for boundary condition enforcement (framework prepared)
- Configurable tolerances and solver parameters

### Modular Design
- Separate headers for mathematical functions, solver, and CSV output
- Clean separation of concerns for maintainability
- Consistent error handling and memory management

## Known Limitations

1. **Simplified BVP Implementation**: Current implementation uses basic integration rather than full multiple shooting. This is a starting framework that can be extended.

2. **SUNDIALS Dependency**: Requires SUNDIALS library installation. Build system provides fallback for missing dependencies.

3. **Convergence**: Like the Python implementation, convergence depends on problem parameters and may require adjustment.

## Comparison with Python Implementation

| Feature | Python | C Implementation |
|---------|--------|------------------|
| Mathematical functions | ✓ | ✓ (direct translation) |
| BVP solving | scipy.solve_bvp | SUNDIALS framework |
| Output format | PNG plots | CSV data files |
| Performance | Moderate | Potentially faster |
| Dependencies | scipy, numpy, matplotlib | SUNDIALS, CMake |
| Testing | pytest | Custom C tests + pytest integration |

## Future Enhancements

1. **Complete Multiple Shooting**: Implement full multiple shooting with KINSOL
2. **Advanced Boundary Conditions**: Support for more complex boundary conditions
3. **Performance Optimization**: Leverage C performance for large-scale problems
4. **Parameter Studies**: Batch processing capabilities
5. **Visualization Tools**: Optional plotting utilities using external libraries

## Troubleshooting

### Build Issues
- **SUNDIALS not found**: Install libsundials-dev or build from source
- **CMake errors**: Ensure CMake 3.10+ is installed
- **Compilation errors**: Check compiler supports C99 standard

### Runtime Issues
- **Solver convergence**: Adjust tolerances or domain parameters
- **Memory errors**: Check input parameters for validity
- **File I/O errors**: Ensure output directory exists and is writable

## Contributing

When modifying the C implementation:
1. Maintain consistency with Python mathematical formulation
2. Add corresponding unit tests for new functionality
3. Update this documentation
4. Ensure memory management is correct (no leaks)
5. Follow existing code style and naming conventions
