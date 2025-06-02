# Plan of Action: Implementing Shooting Method with Gradient Descent for GLE Solver

## Overview
The current C implementation uses an IVP solver (CVODE) which cannot enforce boundary conditions at both ends of the domain. We need to implement a shooting method that:
1. Guesses the initial value of ω(0)
2. Integrates forward using CVODE
3. Checks if the boundary condition ω(Δ) = 0 is satisfied
4. Updates the guess using gradient descent
5. Repeats until convergence

## Mathematical Formulation

### Problem Statement
- **Known BCs:** h(0) = λ_slip, θ(0) = π/6, ω(4Δ) = 0
- **Unknown:** ω(0) = ?
- **Objective:** Find ω(0) such that integrating the IVP yields ω(4Δ) = 0

### Shooting Method
Define the shooting function:
```
F(ω₀) = ω(4Δ; ω₀) - 0
```
where ω(4Δ; ω₀) is the value of ω at s=4Δ when starting with initial condition ω(0)=ω₀.

We want to find ω₀* such that F(ω₀*) = 0.

### Gradient Descent Update
Instead of Newton's method, use gradient descent:
```
ω₀^(k+1) = ω₀^(k) - α * ∇F(ω₀^(k))
```
where:
- α is the learning rate
- ∇F is approximated using finite differences

## Implementation Plan

### 1. New Module Structure
Create new files in the modular architecture:
```
include/
  shooting_method.h     # Shooting method interface
  gradient_descent.h    # Gradient descent optimizer
src/
  shooting_method.c     # Shooting method implementation
  gradient_descent.c    # Gradient descent implementation
```

### 2. Core Components

#### A. Shooting Method Module (`shooting_method.h/c`)
```c
typedef struct {
    GLEParams* gle_params;
    double target_omega;      // Target value of omega at s_end (usually 0)
    double s_end;            // End point of integration
    double tolerance;        // Convergence tolerance
    int max_iterations;      // Maximum shooting iterations
} ShootingParams;

typedef struct {
    double omega_0_guess;    // Current guess for omega(0)
    double residual;         // F(omega_0) = omega(s_end) - target
    int iterations;          // Number of iterations performed
    bool converged;          // Convergence status
} ShootingResult;

// Main shooting method function
ShootingResult solve_shooting_method(ShootingParams* params);

// Evaluate the shooting function F(omega_0)
double evaluate_shooting_function(double omega_0, ShootingParams* params);

// Compute gradient using finite differences
double compute_gradient(double omega_0, double epsilon, ShootingParams* params);
```

#### B. Gradient Descent Module (`gradient_descent.h/c`)
```c
typedef struct {
    double learning_rate;        // Step size α
    double adaptive_factor;      // Factor for adaptive learning rate
    double min_learning_rate;    // Minimum allowed learning rate
    double max_learning_rate;    // Maximum allowed learning rate
    bool use_adaptive;          // Use adaptive learning rate
    double momentum;            // Momentum coefficient (0 for no momentum)
} GradientDescentParams;

typedef struct {
    double* iterates;           // History of omega_0 values
    double* residuals;          // History of F(omega_0) values
    double* gradients;          // History of gradients
    int n_iterations;           // Number of iterations performed
} GradientDescentHistory;

// Perform one gradient descent step
double gradient_descent_step(
    double current_value,
    double gradient,
    double* velocity,  // For momentum
    GradientDescentParams* params
);

// Adaptive learning rate adjustment
double adjust_learning_rate(
    double current_lr,
    double current_residual,
    double previous_residual,
    GradientDescentParams* params
);
```

### 3. Integration with Existing Code

#### Modify `gle_solver.h`:
```c
// Add shooting method option
typedef enum {
    GLE_SOLVER_IVP,      // Current approach
    GLE_SOLVER_SHOOTING  // New shooting method
} GLESolverMethod;

// Update GLEParams to include solver method
typedef struct {
    // ... existing fields ...
    GLESolverMethod method;
    ShootingParams* shooting_params;  // Optional, only for shooting method
} GLEParams;
```

#### Update `gle_solver.c`:
- Add a wrapper function that chooses between IVP and shooting method
- Integrate shooting method results into existing solution structure

### 4. Algorithm Flow

```
1. Initialize shooting parameters
   - Set initial guess ω₀ (e.g., 0 or small negative value)
   - Set gradient descent parameters (learning rate, etc.)

2. Main shooting loop:
   a. Evaluate F(ω₀) by:
      - Set IC: h(0)=λ_slip, θ(0)=π/6, ω(0)=ω₀
      - Integrate using CVODE from s=0 to s=4Δ
      - Extract ω(4Δ) and compute residual F(ω₀) = ω(4Δ) - 0
   
   b. Check convergence: |F(ω₀)| < tolerance
   
   c. Compute gradient ∇F(ω₀):
      - Use finite differences: ∇F ≈ [F(ω₀+ε) - F(ω₀-ε)]/(2ε)
      - Each evaluation requires a full CVODE integration
   
   d. Update ω₀ using gradient descent:
      - ω₀_new = ω₀ - α * ∇F(ω₀)
      - Optionally use momentum or adaptive learning rate
   
   e. Check for divergence or max iterations

3. Return final solution with converged ω₀
```

### 5. Key Design Decisions

#### A. Gradient Computation
- Use central finite differences for better accuracy
- Choose ε adaptively based on machine precision and ω₀ magnitude
- Cache function evaluations to avoid redundant integrations

#### B. Learning Rate Strategy
- Start with conservative learning rate (e.g., 0.1)
- Implement adaptive learning rate:
  - Increase if residual decreases consistently
  - Decrease if residual increases or oscillates
- Add momentum to accelerate convergence in consistent directions

#### C. Initial Guess Strategy
- Start with ω₀ = 0 (physical intuition: flat interface)
- If that fails, try a bracket search to find sign change
- Consider multiple starting points for robustness

#### D. Convergence Criteria
- Primary: |F(ω₀)| < tolerance (e.g., 1e-8)
- Secondary: |ω₀^(k+1) - ω₀^(k)| < tolerance
- Maximum iterations safeguard

### 6. Error Handling and Robustness

#### A. Integration Failures
- If CVODE fails during shooting function evaluation:
  - Return large penalty value
  - Try smaller integration steps
  - Adjust CVODE tolerances

#### B. Gradient Descent Issues
- Detect oscillations and reduce learning rate
- Implement gradient clipping for stability
- Add bounds on ω₀ based on physical constraints

#### C. Diagnostics
- Log iteration history for debugging
- Plot convergence history
- Save intermediate solutions for analysis

### 7. Testing Strategy

#### A. Unit Tests
- Test gradient computation against analytical cases
- Test gradient descent on simple functions
- Verify shooting method on problems with known solutions

#### B. Integration Tests
- Compare with Python solve_bvp results
- Test parameter sensitivity
- Verify physical constraints are satisfied

#### C. Performance Tests
- Measure computation time vs Python
- Profile to identify bottlenecks
- Optimize gradient computation (parallel evaluations?)

### 8. Future Enhancements

1. **Parallel Gradient Computation**: Evaluate F(ω₀+ε) and F(ω₀-ε) in parallel
2. **Higher-Order Methods**: Implement Newton-Raphson as alternative to gradient descent
3. **Multiple Shooting**: Divide domain into segments for better stability
4. **Continuation Methods**: Track solution branches as parameters vary
5. **GPU Acceleration**: For parameter studies requiring many solutions

## Implementation Priority

1. **Phase 1**: Basic shooting method with simple gradient descent
2. **Phase 2**: Add adaptive learning rate and momentum
3. **Phase 3**: Implement robust error handling and diagnostics
4. **Phase 4**: Optimize performance and add advanced features

## Success Metrics

- Achieve same accuracy as Python solve_bvp (within 1e-6)
- Converge in < 100 shooting iterations for standard parameters
- Handle edge cases gracefully (high Ca, small λ_slip, etc.)
- Maintain modular, testable code structure