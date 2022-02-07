set -e

if [[ $(basename $PWD) == *"bluemira"* ]]; then
  cd ..
fi

if [ ! -d pyside2-tools ]
    git clone https://github.com/pyside/pyside2-tools.git
    cd pyside2-tools
    cd ..
fi

if [ -d coin_build ]; then
  echo "Removing previous pyside2-tools build"
  rm -rf pyside2-tools/build
fi

mkdir build && cd build
cmake ..
make
