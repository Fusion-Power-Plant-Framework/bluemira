set -e

PYTHON_VERSION="3.10"
ENVIRONMENT="bluemira"
while getopts "e:p:" flag
do
    case "${flag}" in
        e) ENVIRONMENT="${OPTARG}";;
        p) PYTHON_VERSION="${OPTARG}";;
    esac
done

echo $ENVIRONMENT
# Get and install miniforge
if [ ! -d "$HOME/miniforge" ]; then
  curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh
  bash Miniforge3-$(uname)-$(uname -m).sh -b -p "$HOME/miniforge"
  rm Miniforge3-$(uname)-$(uname -m).sh
fi

# Make an init file so we don't need to edit bashrc
# Note this will currently work for bash terminals
envsubst '$HOME' < conda/miniforge-init.sh > ~/.miniforge-init.sh
source ~/.miniforge-init.sh

# Create the bluemira conda environment
sed s/".*python.*"/"  - python="$PYTHON_VERSION/g ./conda/environment.yml > ./conda/tmp_env.yml
mamba env create -f ./conda/tmp_env.yml -n $ENVIRONMENT
conda activate $ENVIRONMENT

# Install bluemira
python -m pip install --no-cache-dir -e .
