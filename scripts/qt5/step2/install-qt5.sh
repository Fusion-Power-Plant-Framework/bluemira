set -e

if [[ $(basename $PWD) == *"bluemira"* ]]; then
    cd ..
fi

if [ ! -d qt5-build ]; then
  echo "Could not find build for qt5" >> /dev/stderr
  exit 1
fi

cd qt5-build
make install
