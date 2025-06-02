/**
 * gradient_descent.c
 * 
 * Implementation of gradient descent optimization algorithms
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include "src-local/gradient_descent.h"

GradientDescentParams create_default_gd_params(void) {
  GradientDescentParams params;
  params.learning_rate = 0.01;
  params.adaptive_factor = 1.1;
  params.min_learning_rate = 1e-6;
  params.max_learning_rate = 1.0;
  params.use_adaptive = true;
  params.momentum = 0.0;
  params.gradient_clip = 0.0;  // No clipping by default
  return params;
}

double gradient_descent_step(
  double current_value,
  double gradient,
  double* velocity,
  GradientDescentParams* params
) {
  // Apply momentum if enabled
  if (params->momentum > 0.0 && velocity != NULL) {
    *velocity = params->momentum * (*velocity) - params->learning_rate * gradient;
    return current_value + *velocity;
  } else {
    // Standard gradient descent step
    return current_value - params->learning_rate * gradient;
  }
}

double adjust_learning_rate(
  double current_lr,
  double current_residual,
  double previous_residual,
  GradientDescentParams* params
) {
  if (!params->use_adaptive) {
    return current_lr;
  }
  
  double new_lr = current_lr;
  
  // If residual decreased, increase learning rate
  if (current_residual < previous_residual) {
    new_lr = current_lr * params->adaptive_factor;
  } else {
    // If residual increased, decrease learning rate
    new_lr = current_lr / params->adaptive_factor;
  }
  
  // Enforce bounds
  if (new_lr > params->max_learning_rate) {
    new_lr = params->max_learning_rate;
  } else if (new_lr < params->min_learning_rate) {
    new_lr = params->min_learning_rate;
  }
  
  return new_lr;
}

GradientDescentHistory* create_gd_history(int capacity) {
  GradientDescentHistory* history = (GradientDescentHistory*)malloc(sizeof(GradientDescentHistory));
  if (history == NULL) {
    return NULL;
  }
  
  history->capacity = capacity;
  history->n_iterations = 0;
  
  history->iterates = (double*)malloc(capacity * sizeof(double));
  history->residuals = (double*)malloc(capacity * sizeof(double));
  history->gradients = (double*)malloc(capacity * sizeof(double));
  history->learning_rates = (double*)malloc(capacity * sizeof(double));
  
  // Check for allocation failures
  if (history->iterates == NULL || history->residuals == NULL ||
      history->gradients == NULL || history->learning_rates == NULL) {
    free_gd_history(history);
    return NULL;
  }
  
  return history;
}

void add_to_history(
  GradientDescentHistory* history,
  double iterate,
  double residual,
  double gradient,
  double learning_rate
) {
  if (history == NULL || history->n_iterations >= history->capacity) {
    return;
  }
  
  int idx = history->n_iterations;
  history->iterates[idx] = iterate;
  history->residuals[idx] = residual;
  history->gradients[idx] = gradient;
  history->learning_rates[idx] = learning_rate;
  history->n_iterations++;
}

void free_gd_history(GradientDescentHistory* history) {
  if (history == NULL) {
    return;
  }
  
  if (history->iterates != NULL) free(history->iterates);
  if (history->residuals != NULL) free(history->residuals);
  if (history->gradients != NULL) free(history->gradients);
  if (history->learning_rates != NULL) free(history->learning_rates);
  
  free(history);
}

bool is_stalled(GradientDescentHistory* history, int window, double tolerance) {
  if (history == NULL || history->n_iterations < window + 1) {
    return false;
  }
  
  // Check if residuals have changed significantly in the last 'window' iterations
  int start_idx = history->n_iterations - window - 1;
  double max_change = 0.0;
  
  for (int i = start_idx + 1; i < history->n_iterations; i++) {
    double change = fabs(history->residuals[i] - history->residuals[i-1]);
    if (change > max_change) {
      max_change = change;
    }
  }
  
  return max_change < tolerance;
}

double clip_gradient(double gradient, double max_magnitude) {
  if (max_magnitude <= 0.0) {
    return gradient;  // No clipping
  }
  
  if (fabs(gradient) > max_magnitude) {
    return (gradient > 0) ? max_magnitude : -max_magnitude;
  }
  
  return gradient;
}