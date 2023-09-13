#!/bin/bash

read -p "Are you in the correct python environment? (y/n) " answer
case ${answer:0:1} in
    y|Y ) echo "Let's go!";;
    * )
        exit;;
esac

echo
echo

clean_up() {
  test -d "$tmp_dir" && rm -fr "$tmp_dir"
}

tmp_dir=$( mktemp -d -t install-openmc.XXX)
trap "clean_up $tmp_dir" EXIT

cd $tmp_dir

echo Installing libhdf5
sudo apt install libhdf5-serial-dev -y

echo
echo

echo Cloning and building opemc
git clone --recurse-submodules https://github.com/openmc-dev/openmc.git
cd openmc
mkdir build && cd build
echo "-- cmake"
cmake -DOPENMC_USE_MPI=ON ..
echo "-- make"
make
echo "-- sudo make install"
sudo make install

echo
echo

echo pip installing openmc
pip install .

echo Finished
