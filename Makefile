# Compiler and Flags
CC = gcc
CFLAGS = -Wall -g -std=c99 -Isrc-local
# Add -DHAVE_GSL_BVP_H if gsl/gsl_bvp.h is expected to be available and BVP features are desired.
# If gsl_bvp.h is confirmed missing, remove -DHAVE_GSL_BVP_H from CFLAGS.
CFLAGS += -DHAVE_GSL_BVP_H
LDFLAGS = -lm -lgsl -lgslcblas

# Source Files and Output Names
SRC_DIR = src-local
TEST_DIR = test
MAIN_SRC = $(SRC_DIR)/GLE_solver-GSL.c
MAIN_EXEC = gle_solver_gsl
TEST_SRC = $(TEST_DIR)/test_GLE_solver-GSL.c
TEST_EXEC = run_c_tests

# Object file for the library part of GLE_solver-GSL.c (without its main)
LIB_OBJ = $(SRC_DIR)/GLE_solver-GSL_lib.o

# Default target
all: $(MAIN_EXEC)

# Rule to build the main executable
$(MAIN_EXEC): $(MAIN_SRC) $(SRC_DIR)/GLE_solver-GSL.h
	$(CC) $(CFLAGS) $(MAIN_SRC) -o $(MAIN_EXEC) $(LDFLAGS)

# Rule to build the library object file (GLE_solver-GSL.c without its main)
# We define COMPILING_TESTS to exclude main from GLE_solver-GSL.c
$(LIB_OBJ): $(MAIN_SRC) $(SRC_DIR)/GLE_solver-GSL.h
	$(CC) $(CFLAGS) -DCOMPILING_TESTS -c $(MAIN_SRC) -o $(LIB_OBJ)

# Rule to build the test executable
$(TEST_EXEC): $(TEST_SRC) $(LIB_OBJ) $(SRC_DIR)/GLE_solver-GSL.h
	$(CC) $(CFLAGS) $(TEST_SRC) $(LIB_OBJ) -o $(TEST_EXEC) $(LDFLAGS)

# Rule to run C tests
test_c: $(TEST_EXEC)
	@echo "Running C unit tests..."
	./$(TEST_EXEC)

# Clean rule
clean:
	@echo "Cleaning up compiled files..."
	rm -f $(MAIN_EXEC) $(TEST_EXEC) $(LIB_OBJ)
	rm -f output_h.csv output_theta.csv

.PHONY: all test_c clean
