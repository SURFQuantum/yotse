#!/bin/bash

# Function to check the exit status of the last command
check_status() {
  if [ $? -ne 0 ]; then
    echo "Error: A previous command failed"
    exit 1  # Exit with a non-zero error code
  fi
}

# Create venv and activate
python -m venv blueprint_venv
check_status  # Check the exit status after creating venv
source blueprint_venv/bin/activate
check_status  # Check the exit status after activating venv

# Clone the blueprint repository
git clone https://gitlab.com/d.maier/simulation-code-for-requirements-for-a-processing-node-quantum-repeater-on-a-real-world-fiber-grid.git
check_status  # Check the exit status after cloning the repository

# Change to the repository directory
cd simulation-code-for-requirements-for-a-processing-node-quantum-repeater-on-a-real-world-fiber-grid
check_status  # Check the exit status after changing the directory

# Install the dependencies
which python
python --version
poetry config http-basic.netsquid-pypi ${USERNAME} ${PASSWORD}
poetry install
check_status  # Check the exit status after installing dependencies
pip list

# Exit the virtual environment
cd ..
deactivate
check_status  # Check the exit status after deactivating venv

# If all commands succeed, exit with a zero error code (optional)
exit 0
