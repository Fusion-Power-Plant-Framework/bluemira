sudo apt install freecad libgl1-mesa-glx xvfb libboost-timer-dev libboost-filesystem-dev parmetis libhdf5-openmpi-dev

pip install <ffcx ufc basix from git repo probably, depends on dolfinx version>

export PETSC_DIR=$(python -c 'import site; print(site.getsitepackages()[0])')/petsc
export HDF5_ROOT=/usr/lib/x86_64-linux-gnu/hdf5/openmpi/
cmake -G Ninja <dolfinx cpp dir>
 pip install --check-build-dependencies --no-build-isolation <dolfinx python dir>

HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/openmpi/ CC=mpicc HDF5_MPI="ON" pip install --no-binary=h5py h5py

TODO look into
pyvista import segfault

Notes
-----
petsc has to be installed with pip<23.1
petsc4py isnt affected
