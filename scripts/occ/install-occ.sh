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

if [ ! -d pythonocc ]; then
  git clone git://github.com/tpaviot/pythonocc-core.git pythonocc
  cd pythonocc
  git checkout 1f1ef205d878c9a5fbca6f97eb8fe7b4a141db12
  cd ..
fi

cd pythonocc

if [ -d build ]; then
  if ${FORCE}; then
    echo "Removing previous pythonocc build"
    rm -rf build
  else
    echo "Existing pythonocc build exists. Use the flag -f if you want to rebuild."
    exit 1
  fi
fi

mkdir build && cd build
cmake -DOCE_INCLUDE_PATH=/usr/include/opencascade -DOCE_LIB_PATH=/usr/lib/x86_64-linux-gnu ..
make -j$NJOBS
make install
