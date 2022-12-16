#!/usr/bin/env bash

set -e

apt update
apt install -y \
    git cmake pybind11-dev g++ gcc ninja-build pkg-config zlib1g-dev \
    libopenmpi-dev petsc64-dev libeigen3-dev wget libgmp-dev libmpfr-dev

python3 -m pip install fenics-ffc==2019.1.0.post0

# Dolfin won't compile with Boost > 1.72, so let's build boost
wget https://boostorg.jfrog.io/artifactory/main/release/1.72.0/source/boost_1_72_0.tar.gz
tar -xvf boost_1_72_0.tar.gz
(cd boost_1_72_0 && \
    bash ./bootstrap.sh \
        --with-libraries=filesystem,iostreams,program_options,timer,regex,system && \
    ./b2 && ./b2 install)
