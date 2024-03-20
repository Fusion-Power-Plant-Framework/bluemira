set -e

apt-get update\
    && apt-get upgrade -y \
    && apt-get install -y \
     ninja-build build-essential libblas-dev liblapack-dev mpich libboost-timer-dev \
     libboost-filesystem-dev xtensor-dev pkg-config petsc-dev slepc-dev parmetis libpugixml-dev

# libboost-all-dev not necessary
# libboost-timer-dev libboost-filesystem-dev parmetis libhdf5-openmpi-dev ninja-build

source ../h5py/install-h5py.sh

# pip install -U "pip<23.1"
# pip install petsc
# pip install -U pip
# export PETSC_DIR=$(python -c 'import site; print(site.getsitepackages()[0])')/petsc

pip install petsc4py
