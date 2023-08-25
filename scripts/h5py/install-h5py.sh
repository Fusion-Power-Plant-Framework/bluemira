#!/usr/bin/env bash

set -e

apt-get update
apt-get install -y libhdf5-dev
HDF5_DIR=/usr/lib/x86_64-linux-gnu/hdf5/serial/ python3 -m pip install --no-binary h5py h5py
