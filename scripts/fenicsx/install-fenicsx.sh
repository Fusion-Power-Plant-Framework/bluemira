set -e

VERSION="v0.7.2"

export HDF5_ROOT=/usr/lib/x86_64-linux-gnu/hdf5/openmpi/

pip install fenics-ffcx fenics-basix fenics-ufl
git clone https://github.com/FEniCS/dolfinx.git
git checkout $VERSION
cd dolfinx/cpp
mkdir build
cd build
cmake -G Ninja ..
ninja
sudo ninja install
cd ../..
pip install --check-build-dependencies --no-build-isolation ./python/
