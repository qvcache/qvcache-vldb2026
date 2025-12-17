#!/bin/bash

# Build script for QVCache
# This script builds both C++ binaries and Python bindings

set -e

echo "=========================================="
echo "Building QVCache"
echo "=========================================="

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring CMake..."
cmake ..

# Build
echo "Building..."
make -j$(nproc)

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo ""
echo "Binaries are located in: build/benchmarks/"
echo "Python module is located in: python/"
echo ""
echo "To use Python bindings, add the python directory to PYTHONPATH:"
echo "  export PYTHONPATH=\${PYTHONPATH}:$(pwd)/../python"
echo ""

