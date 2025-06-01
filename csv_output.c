#include "src-local/csv_output.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <errno.h>

/**
 * @file csv_output.c
 * @brief Implementation of CSV output functionality for GLE solver results
 * 
 * This file implements functions for saving GLE solution data to CSV files,
 * replacing the PNG output from the Python implementation.
 */

int create_output_directory(const char *dir_path) {
    struct stat st = {0};
    
    /* Check if directory already exists */
    if (stat(dir_path, &st) == 0) {
        return 0; /* Directory exists */
    }
    
    /* Create directory */
    if (mkdir(dir_path, 0755) == 0) {
        return 0; /* Success */
    }
    
    /* Check if error is because directory already exists */
    if (errno == EEXIST) {
        return 0; /* Directory was created by another process */
    }
    
    printf("Error: Failed to create directory '%s': %s\n", dir_path, strerror(errno));
    return -1; /* Failure */
}

int save_h_profile_csv(const char *filename, GLESolution *solution) {
    FILE *file;
    
    if (!solution || !filename) {
        printf("Error: Invalid arguments to save_h_profile_csv\n");
        return -1;
    }
    
    if (solution->n_points <= 0) {
        printf("Error: No solution data to save\n");
        return -1;
    }
    
    /* Open file for writing */
    file = fopen(filename, "w");
    if (!file) {
        printf("Error: Cannot open file '%s' for writing: %s\n", filename, strerror(errno));
        return -1;
    }
    
    /* Write CSV header */
    fprintf(file, "s,h\n");
    
    /* Write data rows */
    for (int i = 0; i < solution->n_points; i++) {
        fprintf(file, "%.12e,%.12e\n", 
                solution->s_values[i], 
                solution->h_values[i]);
    }
    
    fclose(file);
    printf("Film thickness profile saved to: %s\n", filename);
    return 0;
}

int save_theta_profile_csv(const char *filename, GLESolution *solution) {
    FILE *file;
    
    if (!solution || !filename) {
        printf("Error: Invalid arguments to save_theta_profile_csv\n");
        return -1;
    }
    
    if (solution->n_points <= 0) {
        printf("Error: No solution data to save\n");
        return -1;
    }
    
    /* Open file for writing */
    file = fopen(filename, "w");
    if (!file) {
        printf("Error: Cannot open file '%s' for writing: %s\n", filename, strerror(errno));
        return -1;
    }
    
    /* Write CSV header */
    fprintf(file, "s,theta_degrees\n");
    
    /* Write data rows (convert theta from radians to degrees) */
    for (int i = 0; i < solution->n_points; i++) {
        double theta_degrees = solution->theta_values[i] * 180.0 / M_PI;
        fprintf(file, "%.12e,%.12e\n", 
                solution->s_values[i], 
                theta_degrees);
    }
    
    fclose(file);
    printf("Contact angle profile saved to: %s\n", filename);
    return 0;
}

int save_complete_solution_csv(const char *filename, GLESolution *solution) {
    FILE *file;
    
    if (!solution || !filename) {
        printf("Error: Invalid arguments to save_complete_solution_csv\n");
        return -1;
    }
    
    if (solution->n_points <= 0) {
        printf("Error: No solution data to save\n");
        return -1;
    }
    
    /* Open file for writing */
    file = fopen(filename, "w");
    if (!file) {
        printf("Error: Cannot open file '%s' for writing: %s\n", filename, strerror(errno));
        return -1;
    }
    
    /* Write CSV header */
    fprintf(file, "s,h,theta_radians,theta_degrees,omega\n");
    
    /* Write data rows */
    for (int i = 0; i < solution->n_points; i++) {
        double theta_degrees = solution->theta_values[i] * 180.0 / M_PI;
        fprintf(file, "%.12e,%.12e,%.12e,%.12e,%.12e\n", 
                solution->s_values[i], 
                solution->h_values[i],
                solution->theta_values[i],
                theta_degrees,
                solution->w_values[i]);
    }
    
    fclose(file);
    printf("Complete solution saved to: %s\n", filename);
    return 0;
}
