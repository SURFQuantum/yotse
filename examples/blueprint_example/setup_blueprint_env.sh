python -m venv blueprint_venv
source blueprint_venv/bin/activate

# Clone the repository
git https://gitlab.com/d.maier/simulation-code-for-requirements-for-a-processing-node-quantum-repeater-on-a-real-world-fiber-grid.git

# Change to the repository directory
cd simulation-code-for-requirements-for-a-processing-node-quantum-repeater-on-a-real-world-fiber-grid

# Install the dependencies
poetry install
cd ..

# Exit the virtual environment
deactivate
