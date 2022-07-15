set -e

apt-get update\
    && apt-get upgrade -y \
    && apt-get install -y \
     ninja-build libblas-dev liblapack-dev mpich libboost-all-dev xtensor-dev \
     pkg-config petsc-dev slepc-dev libhdf5-openmpi-dev libparmetis-dev libpugixml-dev
