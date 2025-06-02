/**
 * gradient_descent.h
 * 
 * Gradient descent optimizer for numerical optimization
 * 
 * This module provides gradient descent optimization with
 * adaptive learning rate and momentum options for improved
 * convergence in shooting method applications.
 */

#ifndef GRADIENT_DESCENT_H
#define GRADIENT_DESCENT_H

#include <stdbool.h>

typedef struct {
  double learning_rate;        // Step size alpha
  double adaptive_factor;      // Factor for adaptive learning rate
  double min_learning_rate;    // Minimum allowed learning rate
  double max_learning_rate;    // Maximum allowed learning rate
  bool use_adaptive;          // Use adaptive learning rate
  double momentum;            // Momentum coefficient (0 for no momentum)
  double gradient_clip;       // Maximum gradient magnitude (0 for no clipping)
} GradientDescentParams;

typedef struct {
  double* iterates;           // History of parameter values
  double* residuals;          // History of function values
  double* gradients;          // History of gradients
  double* learning_rates;     // History of learning rates used
  int n_iterations;           // Number of iterations performed
  int capacity;               // Allocated size of arrays
} GradientDescentHistory;

/**
 * Perform one gradient descent step
 * 
 * @param current_value Current parameter value
 * @param gradient Current gradient value
 * @param velocity Momentum velocity (updated in-place)
 * @param params Gradient descent parameters
 * @return Updated parameter value
 */
double gradient_descent_step(
  double current_value,
  double gradient,
  double* velocity,
  GradientDescentParams* params
);

/**
 * Adjust learning rate based on progress
 * 
 * @param current_lr Current learning rate
 * @param current_residual Current function value
 * @param previous_residual Previous function value
 * @param params Gradient descent parameters
 * @return Adjusted learning rate
 */
double adjust_learning_rate(
  double current_lr,
  double current_residual,
  double previous_residual,
  GradientDescentParams* params
);

/**
 * Create default gradient descent parameters
 * 
 * @return GradientDescentParams with reasonable defaults
 */
GradientDescentParams create_default_gd_params(void);

/**
 * Initialize gradient descent history
 * 
 * @param capacity Maximum number of iterations to store
 * @return Initialized GradientDescentHistory
 */
GradientDescentHistory* create_gd_history(int capacity);

/**
 * Add iteration to history
 * 
 * @param history History structure to update
 * @param iterate Current parameter value
 * @param residual Current function value
 * @param gradient Current gradient
 * @param learning_rate Learning rate used
 */
void add_to_history(
  GradientDescentHistory* history,
  double iterate,
  double residual,
  double gradient,
  double learning_rate
);

/**
 * Free gradient descent history
 * 
 * @param history History structure to free
 */
void free_gd_history(GradientDescentHistory* history);

/**
 * Check if gradient descent has stalled
 * 
 * @param history Gradient descent history
 * @param window Number of recent iterations to check
 * @param tolerance Tolerance for detecting stall
 * @return true if progress has stalled
 */
bool is_stalled(GradientDescentHistory* history, int window, double tolerance);

/**
 * Clip gradient magnitude if needed
 * 
 * @param gradient Gradient value
 * @param max_magnitude Maximum allowed magnitude
 * @return Clipped gradient
 */
double clip_gradient(double gradient, double max_magnitude);

#endif // GRADIENT_DESCENT_H