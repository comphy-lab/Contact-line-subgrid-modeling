# Test Suite for Contact Line Subgrid Modeling

This directory contains the comprehensive test suite for the GLE (Generalized Lubrication Equations) solver and related components.

## Test Organization

### Unit Tests

1. **test_GLE_solver.py**
   - Tests for mathematical helper functions (f1, f2, f3, f)
   - Tests for ODE system (GLE function)
   - Tests for boundary conditions
   - Tests for x0 finding utilities
   - Coverage: Core mathematical functions and solver components

2. **test_huh_scriven_velocity.py**
   - Tests for Huh-Scriven velocity field calculations
   - Tests for polar coordinates (Ur, Uphi)
   - Tests for Cartesian coordinates (Ux, Uy)
   - Coverage: Velocity field computations

3. **test_GLE_solver-GSL.c**
   - C unit tests for GSL (GNU Scientific Library) implementation
   - Tests for C versions of mathematical functions
   - Tests for ODE integration
   - Coverage: C implementation of the solver

### Integration Tests

4. **test_integration.py**
   - End-to-end tests running complete solver workflows
   - Tests for file I/O operations
   - Tests for plot generation
   - Coverage: Full solver pipeline from input to output

### Validation Tests

5. **validate_LE_solver.py**
   - Validation against known analytical solutions
   - Comparison with published results
   - Coverage: Physical correctness of solutions

6. **test_theta_min_continuation.py**
   - Tests for theta_min-based continuation method
   - Tests for bifurcation tracking
   - Coverage: Advanced continuation algorithms

7. **test_x0_definition.py**
   - Tests for the new x0 definition (position where theta is minimum)
   - Validates x0 computation across different parameter regimes
   - Coverage: Critical parameter tracking

## Running Tests

### Quick Start
```bash
./run_tests.sh
```

This will run both Python and C test suites.

### Python Tests Only
```bash
pytest -v
```

### C Tests Only
```bash
cd .. && make test
```

### With Coverage
```bash
pytest -v --cov=.. --cov-report=html
```

## Test Data

The following CSV file contains reference data for validation:
- **Minkush data.csv**: Reference data for comparison with published results

## Adding New Tests

When adding new tests:
1. Follow the existing naming convention: `test_*.py` for Python tests
2. Use pytest fixtures for common setup
3. Group related tests in classes
4. Include docstrings explaining what each test validates
5. Add integration tests for new features

## Test Coverage Goals

Current focus areas:
- ✅ Mathematical functions
- ✅ ODE solver
- ✅ Boundary conditions
- ✅ File I/O
- ✅ Continuation methods
- ✅ Critical parameter detection
- ✅ x0 computation with new definition
- 🔄 Performance benchmarks (future)
- 🔄 Edge cases and error handling (ongoing)