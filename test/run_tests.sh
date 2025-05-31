#!/bin/bash

# Run unit tests for Contact Line Subgrid Modeling
# This script runs all pytest tests in the current directory

echo "Running Contact Line Subgrid Modeling Tests..."
echo "============================================"

# Add parent directory to PYTHONPATH so tests can import the modules
PARENT_DIR="$(dirname "$(pwd)")"
PYTHONPATH="${PYTHONPATH}:${PARENT_DIR}"
export PYTHONPATH

# Run pytest with verbose output and coverage if available
if command -v pytest &> /dev/null; then
    pytest -v --tb=short
    PYTEST_EXIT_CODE=$?
else
    echo "Error: pytest is not installed. Please install it with:"
    echo "pip install -r ../requirements-python.txt"
    exit 1
fi

# Run C tests if available
echo ""
if [ -f "run_c_tests.sh" ]; then
    echo "Running C tests..."
    bash run_c_tests.sh
    C_TESTS_EXIT_CODE=$?
else
    echo "C tests not available (run_c_tests.sh not found)"
    C_TESTS_EXIT_CODE=0
fi

echo ""
echo "=== All Tests Completed ==="

if [ $PYTEST_EXIT_CODE -ne 0 ] || [ $C_TESTS_EXIT_CODE -ne 0 ]; then
    exit 1
else
    exit 0
fi
