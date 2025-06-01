#!/bin/bash

# Run unit tests for Contact Line Subgrid Modeling
# This script runs all pytest tests in the current directory

# Color codes
RESET='\033[0m'
BOLD='\033[1m'
GREEN='\033[32m'
RED='\033[31m'
YELLOW='\033[33m'
BLUE='\033[34m'
CYAN='\033[36m'

echo -e "${BOLD}${CYAN}"
echo "╔══════════════════════════════════════════════════════╗"
echo "║    Contact Line Subgrid Modeling Test Suite          ║"
echo "╚══════════════════════════════════════════════════════╝"
echo -e "${RESET}"

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Get the parent directory (project root)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Add parent directory to PYTHONPATH so tests can import the modules
PYTHONPATH="${PYTHONPATH}:${PROJECT_ROOT}"
export PYTHONPATH

# Run pytest with verbose output and coverage if available
if command -v pytest &> /dev/null; then
    echo -e "\n${BOLD}${YELLOW}Running Python Unit Tests...${RESET}"
    echo -e "${BLUE}══════════════════════════════════════════════════════${RESET}\n"
    # Run pytest from the script directory
    cd "$SCRIPT_DIR" || { echo -e "${RED}Error: Failed to change to test directory${RESET}"; exit 1; }
    pytest -v --tb=short
else
    echo -e "${RED}Error: pytest is not installed. Please install it with:${RESET}"
    echo "pip install -r ../requirements-python.txt"
    exit 1
fi

# Capture pytest exit code
PYTEST_EXIT_CODE=$?

echo -e "\n${BOLD}${YELLOW}Running C Unit Tests...${RESET}"
echo -e "${BLUE}══════════════════════════════════════════════════════${RESET}"
# Change to project root to run make
cd "$PROJECT_ROOT" || { echo -e "${RED}Error: Failed to change to project root${RESET}"; exit 1; }
make test
MAKE_EXIT_CODE=$?

# Return non-zero if either pytest or make failed
if [ $PYTEST_EXIT_CODE -ne 0 ] || [ $MAKE_EXIT_CODE -ne 0 ]; then
    echo -e "\n${BOLD}${RED}✗ One or more test suites failed.${RESET}"
    exit 1
else
    echo -e "\n${BOLD}${GREEN}╔══════════════════════════════════════════════════════╗${RESET}"
    echo -e "${BOLD}${GREEN}║           ✓ All test suites passed!                  ║${RESET}"
    echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════╝${RESET}\n"
    exit 0
fi