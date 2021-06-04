set -e

# Get and install miniforge
if [ ! -d "$HOME/miniforge3" ]; then
  curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge-$(uname)-$(uname -m).sh
  bash Miniforge-$(uname)-$(uname -m).sh -b
  rm Miniforge-$(uname)-$(uname -m).sh
fi

# Make an init file so we don't need to edit bashrc
# Note this will currently work for bash terminals
envsubst '$HOME' < conda/miniforge-init.sh > ~/.miniforge-init.sh
source ~/.miniforge-init.sh

# Create the bluemira conda environment
conda env create -f conda/environment.yml
conda activate bluemira

# Install bluemira
python -m pip install --no-cache-dir -e .
