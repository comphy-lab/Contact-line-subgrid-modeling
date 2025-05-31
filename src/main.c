#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gle_solver.h"
#include "csv_output.h"

/**
 * @file main.c
 * @brief Main executable for the GLE solver using SUNDIALS
 * 
 * This program solves the Generalized Lubrication Equation boundary value
 * problem and outputs results in CSV format.
 */

void print_usage(const char *program_name) {
    printf("Usage: %s [options]\n", program_name);
    printf("Options:\n");
    printf("  --output-dir DIR    Output directory for CSV files (default: output)\n");
    printf("  --ca VALUE          Capillary number (default: 1.0)\n");
    printf("  --mu-r VALUE        Viscosity ratio (default: 1e-3)\n");
    printf("  --theta0 VALUE      Initial contact angle in degrees (default: 30)\n");
    printf("  --delta VALUE       Domain length parameter (default: 1e-4)\n");
    printf("  --w VALUE           Curvature boundary condition (default: 0.0)\n");
    printf("  --points N          Number of solution points (default: 1000)\n");
    printf("  --help              Show this help message\n");
}

int parse_arguments(int argc, char *argv[], GLEParams *params, 
                   char *output_dir, int *n_points) {
    /* Set defaults */
    gle_params_init(params);
    strcpy(output_dir, "output");
    *n_points = 1000;
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            return -1; /* Signal to show help */
        } else if (strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
            strcpy(output_dir, argv[++i]);
        } else if (strcmp(argv[i], "--ca") == 0 && i + 1 < argc) {
            params->Ca = atof(argv[++i]);
        } else if (strcmp(argv[i], "--mu-r") == 0 && i + 1 < argc) {
            params->mu_r = atof(argv[++i]);
        } else if (strcmp(argv[i], "--theta0") == 0 && i + 1 < argc) {
            params->theta0 = atof(argv[++i]) * M_PI / 180.0; /* Convert to radians */
        } else if (strcmp(argv[i], "--delta") == 0 && i + 1 < argc) {
            params->Delta = atof(argv[++i]);
        } else if (strcmp(argv[i], "--w") == 0 && i + 1 < argc) {
            params->w = atof(argv[++i]);
        } else if (strcmp(argv[i], "--points") == 0 && i + 1 < argc) {
            *n_points = atoi(argv[++i]);
        } else {
            printf("Unknown option: %s\n", argv[i]);
            return -1;
        }
    }
    
    return 0;
}

int main(int argc, char *argv[]) {
    GLEParams params;
    char output_dir[256];
    int n_points;
    
    printf("=== GLE Solver using SUNDIALS ===\n");
    
    /* Parse command line arguments */
    if (parse_arguments(argc, argv, &params, output_dir, &n_points) != 0) {
        print_usage(argv[0]);
        return 1;
    }
    
    /* Validate parameters */
    if (n_points <= 0) {
        printf("Error: Number of points must be positive\n");
        return 1;
    }
    
    if (params.Ca <= 0 || params.mu_r <= 0 || params.Delta <= 0) {
        printf("Error: Physical parameters must be positive\n");
        return 1;
    }
    
    if (params.theta0 <= 0 || params.theta0 >= M_PI) {
        printf("Error: Initial contact angle must be between 0 and 180 degrees\n");
        return 1;
    }
    
    /* Print problem parameters */
    printf("\nProblem Parameters:\n");
    printf("  Capillary number (Ca): %g\n", params.Ca);
    printf("  Viscosity ratio (mu_r): %g\n", params.mu_r);
    printf("  Slip length (lambda_slip): %g\n", params.lambda_slip);
    printf("  Initial contact angle: %g rad = %g deg\n", 
           params.theta0, params.theta0 * 180.0 / M_PI);
    printf("  Domain length (Delta): %g\n", params.Delta);
    printf("  Curvature BC (w): %g\n", params.w);
    printf("  Solution points: %d\n", n_points);
    printf("  Output directory: %s\n", output_dir);
    
    /* Create output directory */
    if (create_output_directory(output_dir) != 0) {
        printf("Error: Failed to create output directory\n");
        return 1;
    }
    
    /* Create SUNDIALS context */
    if (gle_create_context(&params) != 0) {
        printf("Error: Failed to create SUNDIALS context\n");
        return 1;
    }
    
    /* Allocate solution structure */
    GLESolution *solution = gle_solution_alloc(n_points);
    if (!solution) {
        printf("Error: Failed to allocate solution structure\n");
        gle_destroy_context(&params);
        return 1;
    }
    
    /* Solve the BVP */
    printf("\nSolving GLE boundary value problem...\n");
    int solve_flag = gle_solve_bvp(&params, solution);
    
    /* Print solver statistics */
    gle_print_stats(solution);
    
    if (solve_flag != 0) {
        printf("Warning: Solver did not converge properly\n");
        printf("Proceeding with available solution data...\n");
    }
    
    /* Save results to CSV file */
    printf("\nSaving results to CSV file...\n");
    
    char filename[512];
    
    /* Save data as requested: s, h, theta */
    snprintf(filename, sizeof(filename), "%s/data-c-sundials.csv", output_dir);
    FILE *file = fopen(filename, "w");
    if (!file) {
        printf("Error: Cannot open file '%s' for writing\n", filename);
        gle_solution_free(solution);
        gle_destroy_context(&params);
        return 1;
    }
    
    /* Write CSV header */
    fprintf(file, "s,h,theta\n");
    
    /* Write data rows (theta in radians) */
    for (int i = 0; i < solution->n_points; i++) {
        fprintf(file, "%.12e,%.12e,%.12e\n", 
                solution->s_values[i], 
                solution->h_values[i],
                solution->theta_values[i]);
    }
    
    fclose(file);
    printf("Data saved to: %s\n", filename);
    
    /* Clean up */
    gle_solution_free(solution);
    gle_destroy_context(&params);
    
    printf("\n=== GLE Solver completed ===\n");
    
    if (solve_flag == 0) {
        printf("✅ Solution converged successfully\n");
        return 0;
    } else {
        printf("⚠️  Solution may not have converged\n");
        return 2; /* Non-zero exit code but not complete failure */
    }
}
