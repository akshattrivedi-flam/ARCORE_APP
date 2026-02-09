#!/bin/bash

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="$PROJECT_DIR/env"
PYTHON_CMD="python3" # Default to system python3. Can be changed to /usr/bin/python3.10 if needed.

echo "--- Setting up Development Environment ---"
echo "Project Directory: $PROJECT_DIR"
echo "Environment Directory: $ENV_DIR"

# 1. Check for Python
if ! command -v $PYTHON_CMD &> /dev/null; then
    echo "Error: $PYTHON_CMD could not be found."
    exit 1
fi

# 2. Create Virtual Environment
if [ -d "$ENV_DIR" ]; then
    echo "Environment already exists. Skipping creation."
else
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv "$ENV_DIR"
fi

# 3. Activate and Install Dependencies
source "$ENV_DIR/bin/activate"

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies from requirements.txt..."
if [ -f "$PROJECT_DIR/requirements.txt" ]; then
    # Using extra-index-url for PyTorch to ensure we get appropriate Linux binaries if needed, 
    # but standard pip install usually works. 
    # Validating opencv-python-headless for server/headless environments if needed, 
    # but user has desktop so opencv-python is fine.
    pip install -r "$PROJECT_DIR/requirements.txt"
else
    echo "Warning: requirements.txt not found!"
fi

echo "--- Setup Complete ---"
echo "To activate the environment manually, run:"
echo "source $ENV_DIR/bin/activate"
