#!/usr/bin/env bash

set -ex

git clone --branch=2019.1.0.post0 https://bitbucket.org/fenics-project/dolfin
cmake -B dolfin/build -S dolfin -G Ninja -DDOLFIN_ENABLE_TRILINOS=OFF
ninja -C dolfin/build
ninja -C dolfin/build install

git clone --branch=2019.1.0 https://bitbucket.org/fenics-project/mshr
cmake -B mshr/build -S mshr -G Ninja
ninja -C mshr/build
ninja -C mshr/build install

python3 -m pip install dolfin/python
python3 -m pip install mshr/python
