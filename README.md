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

A C implementation of the GLE solver using the GNU Scientific Library (GSL) is provided in the `src-local/` directory.

### Dependencies (C)
- A C compiler (e.g., GCC)
- GNU Scientific Library (GSL)

To install GSL on Debian/Ubuntu-based systems:
```bash
sudo apt-get update
sudo apt-get install libgsl-dev
```
For other systems, please refer to the GSL installation documentation.

### Compilation (C)

A `Makefile` is provided to manage the compilation process.

- **To build the main C solver executable (`gle_solver_gsl`):**
  ```bash
  make
  ```
  or
  ```bash
  make all
  ```
  This compiles `src-local/GLE_solver-GSL.c`. The `Makefile` includes the `-DHAVE_GSL_BVP_H` flag by default, which enables the BVP solving capabilities. If your GSL installation is missing `gsl/gsl_bvp.h` (as observed in some limited environments), you may need to remove this flag from the `CFLAGS` in the `Makefile`, which will disable BVP-specific code paths and data output.

- **To build the C unit test executable (`run_c_tests`):**
  The test executable is built as part of the `make test_c` command or can be built standalone if needed (though typically not directly).

### Running the C Solver

After compilation, the main solver executable can be run from the root directory:
```bash
./gle_solver_gsl
```
If the BVP solver converges successfully (and was compiled with `HAVE_GSL_BVP_H`), it will produce two CSV files in the root directory:
- `output_h.csv`: Contains `s` (dimensionless arc length) and `h` (film height) data.
- `output_theta.csv`: Contains `s` and `theta` (contact angle) data.

If BVP functionalities were disabled during compilation, the program will indicate this and will not produce output files.

### Running C Unit Tests

To compile and run the C unit tests:
```bash
make test_c
```
This command will:
1. Compile `src-local/GLE_solver-GSL.c` as a library object (excluding its `main` function).
2. Compile `test/test_GLE_solver-GSL.c` and link it against the library object and GSL.
3. Run the compiled test executable (`./run_c_tests`).

The C tests will also be run if you execute the main test script:
```bash
sh test/run_tests.sh
```

### Cleaning Compiled Files

To remove all compiled C executables, object files, and output CSVs:
```bash
make clean
```