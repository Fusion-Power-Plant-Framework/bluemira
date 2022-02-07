set -e

if [[ $(basename $PWD) == *"bluemira"* ]]; then
    cd ..
fi

if [ ! -d qt5 ]; then
    git clone https://code.qt.io/qt/qt5.git
    cd qt5
    git checkout 5.14.2
    cd ..
fi

cd qt5
perl init-repository --module-subset=essential,deprecated
export LLVM_INSTALL_DIR=/usr/lib/llvm-6.0
cd ..

if [ -d qt5-build ]; then
    echo "Removing previous qt5 build"
    rm -rf qt5-build
fi

mkdir qt5-build && cd qt5-build
../qt5/configure -opensource -confirm-license -nomake examples -nomake tests
make -j$(nproc --ignore=2)
