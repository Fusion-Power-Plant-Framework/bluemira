set -e

if [ "$1" ]
  then
    PYTHON_VERSION="$1"
else
    PYTHON_VERSION="3.10"
fi

# Get and install mambaforge
if [ ! -d "$HOME/mambaforge" ]; then
  curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh
  bash Mambaforge-$(uname)-$(uname -m).sh -b -p "$HOME/mambaforge"
  rm Mambaforge-$(uname)-$(uname -m).sh
fi

# Make an init file so we don't need to edit bashrc
# Note this will currently work for bash terminals
envsubst '$HOME' < conda/mambaforge-init.sh > ~/.mambaforge-init.sh
source ~/.mambaforge-init.sh

# Create the bluemira conda environment
sed s/".*python.*"/"  - python="$PYTHON_VERSION/g ../conda/environment.yml > tmp_env.yml
mamba env create -f tmp_env.yml
conda activate bluemira

# Install bluemira
python -m pip install --no-cache-dir -e .
