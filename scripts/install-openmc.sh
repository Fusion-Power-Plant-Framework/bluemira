#!/bin/bash

read -p "Are you in the correct python environment? (y/n) " answer
case ${answer:0:1} in
    y|Y )
        ;;
    * )
        exit;;
esac

echo
echo Installing...
echo

clean_up() {
  test -d "$tmp_dir" && rm -fr "$tmp_dir"
}

tmp_dir=$( mktemp -d -t install-openmc.XXX)
trap "clean_up $tmp_dir" EXIT

set -euxo pipefail

cd $tmp_dir

sudo apt install libhdf5-serial-dev -y

git clone --recurse-submodules https://github.com/openmc-dev/openmc.git
mkdir build && cd build
cmake -DOPENMC_USE_MPI=ON ..
make
sudo make install

pip install .

echo Finished
