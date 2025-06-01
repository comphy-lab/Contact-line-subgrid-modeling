#!/bin/bash

# Run unit tests for Contact Line Subgrid Modeling
# This script runs all pytest tests in the current directory

echo "Running Contact Line Subgrid Modeling Tests..."
echo "============================================"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the parent directory (project root)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Add parent directory to PYTHONPATH so tests can import the modules
PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"
export PYTHONPATH

# Run pytest with verbose output and coverage if available
if command -v pytest &> /dev/null; then
    # Run pytest from the script directory
    cd "$SCRIPT_DIR" || { echo "Error: Failed to change to test directory"; exit 1; }
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
# Change to project root to run make
cd "$PROJECT_ROOT" || { echo "Error: Failed to change to project root"; exit 1; }
make test
MAKE_EXIT_CODE=$?

# Return non-zero if either pytest or make failed
if [ $PYTEST_EXIT_CODE -ne 0 ] || [ $MAKE_EXIT_CODE -ne 0 ]; then
    echo "One or more test suites failed."
    exit 1
else
    echo "All test suites passed."
    exit 0
fi