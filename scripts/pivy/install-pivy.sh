set -e

NJOBS=$(nproc --ignore=2)

while getopts j: option
do
  case "${option}"
  in
    j) NJOBS=${OPTARG};;
  esac
done

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

mkdir build && cd build
cmake ..
make -j$NJOBS
make install
