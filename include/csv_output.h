#ifndef CSV_OUTPUT_H
#define CSV_OUTPUT_H

#include "gle_solver.h"

/**
 * @file csv_output.h
 * @brief CSV output functionality for GLE solver results
 * 
 * This header defines functions for saving GLE solution data to CSV files,
 * replacing the PNG output from the Python implementation.
 */

/**
 * @brief Save film thickness profile to CSV file
 * @param filename Output filename
 * @param solution Solution structure containing data
 * @return Success flag (0 = success)
 */
int save_h_profile_csv(const char *filename, GLESolution *solution);

/**
 * @brief Save contact angle profile to CSV file
 * @param filename Output filename
 * @param solution Solution structure containing data
 * @return Success flag (0 = success)
 */
int save_theta_profile_csv(const char *filename, GLESolution *solution);

/**
 * @brief Save complete solution data to CSV file
 * @param filename Output filename
 * @param solution Solution structure containing data
 * @return Success flag (0 = success)
 */
int save_complete_solution_csv(const char *filename, GLESolution *solution);

/**
 * @brief Create output directory if it doesn't exist
 * @param dir_path Directory path to create
 * @return Success flag (0 = success)
 */
int create_output_directory(const char *dir_path);

#endif /* CSV_OUTPUT_H */
