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
else
    echo "Error: pytest is not installed. Please install it with:"
    echo "pip install -r ../requirements-python.txt"
    exit 1
fi

# Capture pytest exit code
PYTEST_EXIT_CODE=$?

echo ""
echo "Running C unit tests via Makefile..."
echo "=================================="
cd ..
make test
MAKE_EXIT_CODE=$?
cd test

# Return non-zero if either pytest or make failed
if [ $PYTEST_EXIT_CODE -ne 0 ] || [ $MAKE_EXIT_CODE -ne 0 ]; then
    echo "One or more test suites failed."
    exit 1
else
    echo "All test suites passed."
    exit 0
fi