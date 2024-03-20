# NOTE: Ideally there would be a manylinux wheel
# containing PROCESS and the required dependencies (see commented section at the end).

set -e

REQUIRED_PKG="build-essential ninja-build"
PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $REQUIRED_PKG|grep -zoP "install ok installed\ninstall ok installed")
if [ "" = "$PKG_OK" ]; then
  echo "No $REQUIRED_PKG. Setting up $REQUIRED_PKG."
  sudo apt --yes install $REQUIRED_PKG
fi

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
    git checkout v3.0.2
fi

# Make sure we always perform a fresh install of PROCESS
# Takes longer but ensures we avoid leaving old environment references in cmake
if [ -d build ]; then
  rm -rf build
fi

# Do the PROCESS build
cmake -G Ninja -S . -B build -DRELEASE=TRUE
cmake --build build

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
