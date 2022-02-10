set -e

NJOBS=$(nproc --ignore=2)
FORCE="false"

while getopts j:f option
do
  case "${option}"
  in
    j) NJOBS=${OPTARG};;
    f) FORCE="true";;
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
  if ${FORCE}; then
    echo "Removing previous pivy build"
    rm -rf build
  else
    echo "Existing pivy build exists. Use the flag -f if you want to rebuild."
    exit 1
  fi
fi

mkdir build && cd build
cmake ..
make -j$NJOBS
make install
