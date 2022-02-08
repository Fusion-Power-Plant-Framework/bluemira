set -e

if [[ $(basename $PWD) == *"bluemira"* ]]; then
  cd ..
fi

if [ ! -d pivy ]; then
  git clone https://github.com/coin3d/pivy.git
  cd pivy
  git checkout 0.6.6
  cd ..
fi

cd pivy

if [ -d build ]; then
  echo "Removing previous pivy build"
  rm -rf build
fi

mkdir pivy-build && cd pivy-build
cmake ../pivy
make -j$(nproc --ignore=2)
make install
