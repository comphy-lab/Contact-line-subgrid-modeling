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

# Return exit code from pytest
exit $?