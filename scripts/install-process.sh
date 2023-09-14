# NOTE: It appears that we currently get a segfault if we build PROCESS directly against
# the bluemira conda environment, so this script currently build PROCESS in a second
# environment and then installs the resulting package into the bluemira environment.
# This is clearly not idea, but works for now. Ideally there would be a manylinux wheel
# containing PROCESS and the required dependencies (see commented section at the end).

set -e

# Ensure we are working in the (bluemira) conda environment
source ~/.mambaforge-init.sh && conda activate bluemira

if [[ $(basename $PWD) == *"bluemira"* ]]; then
  cd ..
fi

if [ ! -d process ]; then
  git clone git@github.com:ukaea/process.git
fi

cd process
git checkout main
git pull

if [ "$1" ]
  then
    git checkout "$1"
else
    git checkout v3.0.0
fi


# Make sure we always perform a fresh install of PROCESS
# Takes longer but ensures we avoid leaving old environment references in cmake
if [ -d build ]; then
  rm -rf build
fi

# Come out of bluemira conda and make a new environment for our build
conda deactivate
conda env remove -n bluemira-process-build || true
conda create -y -n bluemira-process-build python=3.8 numpy=1.21.5
conda activate bluemira-process-build

# Install requirements into the build environment
pip install -r ../bluemira/requirements.txt
pip install -r requirements.txt --no-cache-dir
pip install --upgrade --no-cache-dir -e ../bluemira/'[process]'

# Do the PROCESS build
cmake -S . -B build
cmake --build build

# Deactivate build environment and reactivate bluemira
conda deactivate && conda activate bluemira

# Install PROCESS dependencies into bluemira environment
pip install -r requirements.txt --no-cache-dir
pip install --upgrade --no-cache-dir -e ../bluemira/'[process]'

# Install PROCESS into bluemira environment
pip install .

# Clean up our build environment
conda env remove -n bluemira-process-build || true

# The following suggests how to install PROCESS via a manylinux wheel, if you have docker
# installed.

# docker run -v $PWD:/src/process/ \
#            -v $PWD/../bluemira/:/src/bluemira/ \
#            -it quay.io/pypa/manylinux2014_x86_64:latest \
#            /bin/bash

# cd /src/process

# python3.8 -m venv .venv  # Create a venv otherwise cmake gets confused finding f90wrap
# source .venv/bin/activate

# python -m pip install -r ../bluemira/requirements.txt  # So that we have the right numpy version
# python -m pip install -r requirements.txt --no-cache-dir
# python -m pip install --upgrade --no-cache-dir -e ../bluemira/'[process]'

# cmake -S . -B build
# cmake --build build

# if [ -d wheelhouse ]; then
#   rm -rf wheelhouse
# fi
# python -m pip wheel . -w wheelhouse
# python -m pip install auditwheel
# auditwheel repair wheelhouse/process*whl -w wheelhouse

# exit

# You should now have a portable wheel in process/wheelhouse
