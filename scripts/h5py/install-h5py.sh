#!/usr/bin/env bash

set -e

apt-get update
apt-get install -y libhdf5-dev
HDF5_DIR=HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial/ python -m pip install h5py
