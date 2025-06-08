# GLE Continuation v4 - Unified Solver

## Overview

`GLE_continuation_v4.py` is a production-ready continuation solver that unifies the capabilities of the previous three implementations:
- Pseudo-arclength continuation (from `GLE_continuation_arclength.py`) 
- Natural parameter continuation (from `GLE_continuation_simple.py`)
- Clean architecture and plotting (from all versions)

## Key Features

### Method Selection
- **Pseudo-arclength** (default): Tracks solution branches through fold bifurcations, capturing both stable and unstable branches
- **Natural parameter**: Simpler and faster for tracking stable branches up to the critical Ca

### Capabilities
- Tracks contact line displacement δX_cl = X_cl(Ca) - X_cl(Ca=0)
- Automatic fold detection and branch stability classification
- Adaptive step size control
- Comprehensive plotting and data export
- Clean command-line interface

## Usage

### Basic Examples

```bash
# Pseudo-arclength continuation (default)
python GLE_continuation_v4.py --mu_r 1e-6 --lambda_slip 1e-4 --theta0 60

# Natural parameter continuation (simpler, faster)
python GLE_continuation_v4.py --method natural --mu_r 1e-6 --lambda_slip 1e-4 --theta0 60

# With custom parameters
python GLE_continuation_v4.py --mu_r 1e-3 --lambda_slip 1e-3 --theta0 45 \
  --initial-ds 0.005 --max-steps 100 --output-dir results/
```

### Command Line Options

```
--method {arclength,natural}  # Continuation method (default: arclength)
--mu_r                       # Viscosity ratio μ_g/μ_l
--lambda_slip                # Slip length parameter λ  
--theta0                     # Initial contact angle in degrees
--Delta                      # Domain size (default: 10.0)
--initial-ds                 # Initial step size (default: 0.01)
--max-ds                     # Maximum step size (default: 0.05)
--min-ds                     # Minimum step size (default: 1e-5)
--max-steps                  # Maximum continuation steps (default: 200)
--Ca-max                     # Maximum Ca for natural method (default: 0.2)
--output-dir                 # Output directory (default: output)
--quiet                      # Suppress progress output
```

## Output Files

The solver generates several output files in the specified directory:

1. **bifurcation_diagram_{method}.png**: Main visualization showing δX_cl vs Ca and θ_min vs Ca
2. **bifurcation_data_{method}.txt**: Text data file with all solution points
3. **branch_{method}.pkl**: Python pickle file containing complete branch data and analysis

## Method Comparison

### Pseudo-arclength Method
- **Pros**: Tracks complete bifurcation diagram including unstable branches
- **Cons**: More computationally intensive, can be slower
- **Use when**: You need to understand the full solution structure or track past fold points

### Natural Parameter Method  
- **Pros**: Simple, fast, robust for stable branches
- **Cons**: Cannot track past fold points, stops at critical Ca
- **Use when**: You only need the stable branch or want quick results

## Implementation Details

The solver imports utilities from `src-local/`:
- `gle_utils.py`: Core BVP solver
- `find_x0_utils.py`: x0 and θ_min finding functions

Key classes:
- `ContinuationPoint`: Data structure for solution points
- `ContinuationParams`: Algorithm parameters
- `GLEContinuation`: Main solver class

## Testing

Run the test script to verify both methods work correctly:

```bash
python test_continuation_v4.py
```

This will run several test cases and report success/failure for each method.

## Migration from Previous Versions

To migrate from the previous implementations:

1. **From GLE_continuation_arclength.py**: Use default settings
2. **From GLE_continuation_simple.py**: Add `--method natural`
3. **From GLE_continuation_hybrid.py**: Update import statements and class names

The output format is backward compatible with analysis scripts expecting the previous data formats.