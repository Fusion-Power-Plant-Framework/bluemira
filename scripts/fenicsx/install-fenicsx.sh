set -e

pip install "git+https://github.com/FEniCS/basix.git#egg=fenics-basix[optional]"
pip install git+https://github.com/FEniCS/ufl.git
pip install git+https://github.com/FEniCS/ffcx.git
git clone https://github.com/FEniCS/dolfinx.git
cd dolfinx/cpp
mkdir build
cd build
cmake -G Ninja ..
ninja install
cd ../..
pip install python/
