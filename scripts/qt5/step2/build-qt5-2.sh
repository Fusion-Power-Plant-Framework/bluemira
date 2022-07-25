set -euxo pipefail

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

cd ..

export LLVM_INSTALL_DIR=/usr/lib/llvm-14
PATCH_DIR=/opt/bluemira/scripts/qt5/step2
cd qt5/qtbase
# patch -p 1 -f < "$PATCH_DIR/gcc11_patch1.patch"
# patch -p 1 -f < "$PATCH_DIR/gcc11_patch2.patch"

cd ../qtdeclarative
# patch -p 1 -f < "$PATCH_DIR/gcc11_patch3.patch"

cd ../../

if [ -d qt5-build ]; then
  if ${FORCE}; then
    echo "Removing previous qt5 build"
    rm -rf qt5-build
  else
    echo "Existing qt5 build exists. Use the flag -f if you want to rebuild."
    exit 1
  fi
fi

mkdir qt5-build && cd qt5-build
../qt5/configure -opensource -platform linux-g++ -confirm-license -nomake examples -nomake tests
make -j$NJOBS
