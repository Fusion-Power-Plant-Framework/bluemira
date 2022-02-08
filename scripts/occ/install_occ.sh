set -e

if [[ $(basename $PWD) == *"bluemira"* ]]; then
  cd ..
fi

if [ ! -d pythonocc ]; then
  git clone git://github.com/tpaviot/pythonocc-core.git pythonocc
  cd pythonocc
  git checkout 7.5.1
  cd ..
fi

cd pythonocc

if [ -d build ]; then
  echo "Removing previous pythonocc build"
  rm -rf build
fi

mkdir build && cd build
cmake -DOCE_INCLUDE_PATH=/usr/include/opencascade -DOCE_LIB_PATH=/usr/lib/x86_64-linux-gnu ..
make -j$(nproc --ignore=2)
