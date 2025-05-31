# Makefile for the GLE solver implementation
# Author: Claude
# Date: 2025-05-31

CC = gcc
CFLAGS = -Wall -Wextra -O2 -g -std=c99 -Isrc-local $(shell pkg-config --cflags gsl)
LDFLAGS = $(shell pkg-config --libs gsl)

# Check if GSL BVP is available
GSL_BVP_CHECK := $(shell echo '#include <gsl/gsl_bvp.h>' | $(CC) -E -x c - >/dev/null 2>&1 && echo "yes" || echo "no")

ifeq ($(GSL_BVP_CHECK),yes)
    CFLAGS += -DHAVE_GSL_BVP_H
    $(info GSL BVP support detected)
else
    $(warning GSL BVP support not detected - solver will use fallback implementation)
endif

# Source files
SRC_DIR = .
TEST_DIR = test
BUILD_DIR = build

# Targets
SOLVER = gle_solver_gsl
TEST_EXEC = test_gle_solver_gsl

# Object files
SOLVER_OBJ = $(BUILD_DIR)/GLE_solver-GSL.o
TEST_OBJ = $(BUILD_DIR)/test_GLE_solver-GSL.o

.PHONY: all clean test run

all: $(BUILD_DIR) $(SOLVER)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build the solver executable
$(SOLVER): $(SOLVER_OBJ)
	$(CC) $^ -o $@ $(LDFLAGS)

# Build the solver object file
$(BUILD_DIR)/GLE_solver-GSL.o: $(SRC_DIR)/GLE_solver-GSL.c $(SRC_DIR)/src-local/GLE_solver-GSL.h
	$(CC) $(CFLAGS) -c $< -o $@

# Build the test executable
$(TEST_EXEC): $(TEST_OBJ) $(BUILD_DIR)/GLE_solver-GSL-test.o
	$(CC) $^ -o $@ $(LDFLAGS)

# Build test object file
$(BUILD_DIR)/test_GLE_solver-GSL.o: $(TEST_DIR)/test_GLE_solver-GSL.c $(SRC_DIR)/src-local/GLE_solver-GSL.h
	$(CC) $(CFLAGS) -c $< -o $@

# Build solver object for tests (with COMPILING_TESTS defined)
$(BUILD_DIR)/GLE_solver-GSL-test.o: $(SRC_DIR)/GLE_solver-GSL.c $(SRC_DIR)/src-local/GLE_solver-GSL.h
	$(CC) $(CFLAGS) -DCOMPILING_TESTS -c $< -o $@

test: $(TEST_EXEC)
	./$(TEST_EXEC)

run: $(SOLVER)
	mkdir -p output
	./$(SOLVER)

compare: run
	@echo "Comparing C and Python outputs..."
	@echo "Running Python solver..."
	python GLE_solver.py
	@echo "Outputs saved in output/ directory"

clean:
	rm -rf $(BUILD_DIR) $(SOLVER) $(TEST_EXEC)
	rm -f output/GLE_*_c.csv

help:
	@echo "Available targets:"
	@echo "  make all      - Build the solver"
	@echo "  make test     - Build and run tests"
	@echo "  make run      - Run the solver"
	@echo "  make compare  - Run both C and Python solvers"
	@echo "  make clean    - Clean build artifacts"
	@echo "  make help     - Show this help message"