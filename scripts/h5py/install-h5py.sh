#!/usr/bin/env bash

set -e

apt-get update
apt-get install -y libhdf5-openmpi-dev
HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/openmpi/ CC=mpicc HDF5_MPI="ON" pip install --no-binary=h5py h5py
