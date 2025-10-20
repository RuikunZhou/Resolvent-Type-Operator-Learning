#!/bin/bash

# Setup script for Koopman Resolvent Learning Framework

echo "=================================================="
echo "Koopman Resolvent Learning Framework - Setup"
echo "=================================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1)
echo "Found: $python_version"
echo ""

# Install dependencies
echo "Installing required packages..."
pip3 install -r requirements.txt
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p results
echo "✓ Created data/ directory"
echo "✓ Created results/ directory"
echo ""

# Verify installation
echo "Verifying installation..."
python3 -c "import numpy; import scipy; import matplotlib; print('✓ All dependencies successfully installed')"
echo ""

echo "=================================================="
echo "Setup complete!"
echo "=================================================="
echo ""
echo "To run an example:"
echo "  cd examples"
echo "  python3 example_van_der_pol.py"
echo ""
echo "To create a custom example:"
echo "  cp examples/example_template.py examples/my_example.py"
echo "  # Edit my_example.py with your system"
echo "  python3 my_example.py"
echo ""
