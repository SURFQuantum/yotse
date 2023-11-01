# create venv and activate
python -m venv blueprint_venv
source blueprint_venv/bin/activate

# Clone the blueprint repository
git clone https://gitlab.com/d.maier/simulation-code-for-requirements-for-a-processing-node-quantum-repeater-on-a-real-world-fiber-grid.git

# Change to the repository directory
cd simulation-code-for-requirements-for-a-processing-node-quantum-repeater-on-a-real-world-fiber-grid

# Install the dependencies
poetry config netsquid-pypi ${USERNAME} ${PASSWORD}
poetry install
pip list
# Exit the virtual environment
cd ..
deactivate
