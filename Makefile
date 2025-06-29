# Makefile for the GLE solver implementation
# Date: 2025-05-31

# Configurable paths - can be overridden externally
# Example: make INCLUDE_PATH=/usr/local/include LIB_PATH=/usr/local/lib
INCLUDE_PATH ?= /Users/vatsal/anaconda3/include
LIB_PATH ?= /Users/vatsal/anaconda3/lib

CC = gcc
CFLAGS = -Wall -Wextra -O2 -g -std=c99 -Isrc-local -I$(INCLUDE_PATH)
LDFLAGS = -L$(LIB_PATH) -Wl,-rpath,$(LIB_PATH) -lgsl -lgslcblas -lopenblas -lm

# Check if GSL BVP is available
GSL_BVP_CHECK := $(shell echo "\#include <gsl/gsl_bvp.h>" | $(CC) -E -x c - >/dev/null 2>&1 && echo "yes" || echo "no")

ifeq ($(GSL_BVP_CHECK),yes)
    CFLAGS += -DHAVE_GSL_BVP_H
    $(info GSL BVP support detected)
else
    $(warning GSL BVP support not detected - solver will use fallback implementation)
endif

# Source files
SRC_DIR = .
SRC_LOCAL_DIR = src-local
TEST_DIR = test
BUILD_DIR = build

# Targets
SOLVER = gle_solver_gsl
TEST_EXEC = test_gle_solver_gsl

# No core object files needed - all functions are static inline in headers

# Main program object
SOLVER_OBJ = $(BUILD_DIR)/GLE_solver-GSL.o

# Test object
TEST_OBJ = $(BUILD_DIR)/test_GLE_solver-GSL.o

.PHONY: all clean test run

all: $(BUILD_DIR) $(SOLVER)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Build the solver executable
$(SOLVER): $(SOLVER_OBJ)
	$(CC) $^ -o $@ $(LDFLAGS)


# Header dependencies for rebuild when headers change
HEADERS = $(SRC_LOCAL_DIR)/GLE_solver-GSL.h \
          $(SRC_LOCAL_DIR)/gle_physics.h \
          $(SRC_LOCAL_DIR)/gle_ode_systems.h \
          $(SRC_LOCAL_DIR)/gle_optimization.h \
          $(SRC_LOCAL_DIR)/gle_shooting.h \
          $(SRC_LOCAL_DIR)/gle_io.h

# Update dependency for main object file
$(BUILD_DIR)/GLE_solver-GSL.o: $(SRC_DIR)/GLE_solver-GSL.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

# Build the test executable
$(TEST_EXEC): $(TEST_OBJ)
	$(CC) $^ -o $@ $(LDFLAGS)

# Build test object file
$(BUILD_DIR)/test_GLE_solver-GSL.o: $(TEST_DIR)/test_GLE_solver-GSL.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

test: $(TEST_EXEC)
	@echo ""
	@echo "\033[1;36m╔══════════════════════════════════════════╗\033[0m"
	@echo "\033[1;36m║        Running C Unit Tests              ║\033[0m"
	@echo "\033[1;36m╚══════════════════════════════════════════╝\033[0m"
	@echo ""
	@./$(TEST_EXEC) || (echo "\n\033[1;31m✗ Tests failed!\033[0m" && exit 1)
	@echo ""
	@echo "\033[1;32m✓ All C tests passed successfully!\033[0m"
	@echo ""

run: $(SOLVER)
	mkdir -p output
	./$(SOLVER)

compare: run
	@echo "Running Python solver..."
	python GLE_solver.py
	@echo "Running comparison between C and Python solvers..."
	python3 compare_results.py

clean:
	rm -rf $(BUILD_DIR) $(SOLVER) $(TEST_EXEC)
	rm -f output/data-c-gsl.csv output/GLE_h_profile_c.csv output/GLE_theta_profile_c.csv

help:
	@echo "Available targets:"
	@echo "  make all      - Build the solver"
	@echo "  make test     - Build and run tests"
	@echo "  make run      - Run the solver"
	@echo "  make compare  - Run both C and Python solvers"
	@echo "  make clean    - Clean build artifacts"
	@echo "  make help     - Show this help message"
