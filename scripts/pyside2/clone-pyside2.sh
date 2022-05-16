set -e

if [[ $(basename $PWD) == *"bluemira"* ]]; then
  cd ..
fi

if [ ! -d pyside-setup ]; then
    git clone --recursive https://code.qt.io/pyside/pyside-setup
    cd pyside-setup
    git checkout 5.15.2
    cd ..
fi
