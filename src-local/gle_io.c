/**
 * gle_io.c
 * 
 * Input/Output operations for the GLE solver
 * Handles file creation, data writing, and directory management
 * 
 * Date: 2025-06-02
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <gsl/gsl_errno.h>
#include "GLE_solver-GSL.h"

/**
 * Create output directory if it doesn't exist
 * Returns 0 on success, -1 on failure
 */
int create_output_directory(const char *dirname) {
    struct stat st = {0};
    
    if (stat(dirname, &st) == -1) {
        if (mkdir(dirname, 0755) != 0) {
            fprintf(stderr, "Error creating output directory '%s': %s\n", 
                    dirname, strerror(errno));
            return -1;
        }
    }
    
    return 0;
}

/**
 * Write solution data to CSV file
 * Format: s,h,theta with high precision
 */
int write_solution_to_csv(const char *filename, 
                         double *s_data, double *h_data, double *theta_data,
                         int n_points) {
    FILE *file = fopen(filename, "w");
    
    if (!file) {
        fprintf(stderr, "Error opening output file '%s': %s\n", 
                filename, strerror(errno));
        return -1;
    }
    
    // Write header
    fprintf(file, "s,h,theta\n");
    
    // Write data with high precision
    for (int i = 0; i < n_points; i++) {
        fprintf(file, "%.12e,%.12e,%.12e\n", 
                s_data[i], h_data[i], theta_data[i]);
    }
    
    fclose(file);
    return 0;
}

/**
 * Main solver function - orchestrates the shooting method solution
 * 
 * ALGORITHM STEPS:
 * 1. Set up physical parameters (Ca, λ, μᵣ)
 * 2. Call shooting method to find ω₀
 * 3. Integrate with found ω₀ to generate full solution
 * 4. Save results to CSV file for analysis/plotting
 */
int solve_gle_shooting_and_save(size_t num_nodes, int verbose) {
    (void)num_nodes;  // Not used - we use fixed number in shooting method
    
    // Set up parameters
    gle_parameters params = {
        .Ca = CA,
        .lambda_slip = LAMBDA_SLIP,
        .mu_r = MU_R,
        .Delta = DELTA
    };

    double *s_out, *h_out, *theta_out;
    int n_points;

    if (verbose) {
        printf("Using shooting method to solve GLE...\n");
    }

    // Solve the GLE system
    int status = solve_gle_shooting_method(&params, S_MAX, 
                                         &s_out, &h_out, &theta_out, &n_points);

    if (status == 0) {
        // Create output directory
        if (create_output_directory("output") != 0) {
            free(s_out);
            free(h_out);
            free(theta_out);
            return GSL_EFAILED;
        }

        // Write results to CSV file
        if (write_solution_to_csv("output/data-c-gsl.csv", 
                                 s_out, h_out, theta_out, n_points) == 0) {
            if (verbose) {
                printf("Results saved to: output/data-c-gsl.csv\n");
            }
        }

        // Free allocated memory
        free(s_out);
        free(h_out);
        free(theta_out);

        return GSL_SUCCESS;
    }

    return GSL_EFAILED;
}