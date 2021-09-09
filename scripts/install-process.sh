set -e

# Ensure we are working in the (bluemira) conda environment
source ~/.miniforge-init.sh && conda activate bluemira

if [[ $(basename $PWD) == *"bluemira"* ]]; then
  cd ..
fi

if [ ! -d process ]; then
  git clone git@git.ccfe.ac.uk:process/process.git
  cd process
  git checkout develop
  cd ..
fi

cd process

pip install -r requirements.txt --no-cache-dir
pip install --upgrade --no-cache-dir -e ../bluemira/'[process]'

# Make sure we always perform a fresh install of PROCESS
# Takes longer but ensures we avoid leaving old environment references in cmake
if [ -d build ]; then
  rm -r build
fi

# Do the PROCESS build
# This will also put PROCESS into our (bluemira) environment
cmake -S . -B build
cmake --build build
