#!/bin/bash


set -e  # Exit on any error

echo "=== Running C Tests for GLE Solver ==="

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

BUILD_DIR="$PROJECT_ROOT/build"
mkdir -p "$BUILD_DIR"

echo "Building C tests..."
cd "$BUILD_DIR"

if command -v cmake >/dev/null 2>&1; then
    cmake "$PROJECT_ROOT"
    
    if make; then
        echo "✅ C code compiled successfully"
        
        echo ""
        echo "Running mathematical function tests..."
        if [ -f "./test_gle_math_c" ]; then
            ./test_gle_math_c
        else
            echo "⚠️  Mathematical function tests not found (test_gle_math_c)"
        fi
        
        echo ""
        echo "Running solver tests..."
        if [ -f "./test_gle_solver_c" ]; then
            ./test_gle_solver_c
        else
            echo "⚠️  Solver tests not found (test_gle_solver_c)"
        fi
        
        echo ""
        echo "Testing main executable..."
        if [ -f "./gle_solver_main" ]; then
            echo "Running GLE solver with default parameters..."
            ./gle_solver_main --output-dir test_c_output --points 50
            
            if [ -f "test_c_output/gle_h_profile.csv" ] && [ -f "test_c_output/gle_theta_profile.csv" ]; then
                echo "✅ CSV output files created successfully"
                
                echo "Sample h profile data:"
                head -5 test_c_output/gle_h_profile.csv
                echo "Sample theta profile data:"
                head -5 test_c_output/gle_theta_profile.csv
            else
                echo "⚠️  CSV output files not found"
            fi
        else
            echo "⚠️  Main executable not found (gle_solver_main)"
        fi
        
    else
        echo "❌ C code compilation failed"
        echo "This may be due to missing SUNDIALS library"
        echo "To install SUNDIALS on Ubuntu: sudo apt-get install libsundials-dev"
        exit 1
    fi
else
    echo "❌ CMake not found. Please install CMake to build C tests."
    echo "On Ubuntu: sudo apt-get install cmake"
    exit 1
fi

echo ""
echo "=== C Tests Completed ==="
