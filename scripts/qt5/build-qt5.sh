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

if [ ! -d qt5 ]; then
  git clone https://code.qt.io/qt/qt5.git
  cd qt5
  git checkout 5.14.2
  cd ..
fi

cd qt5
perl init-repository -f --module-subset=essential,qtxmlpatterns,qtsvg
export LLVM_INSTALL_DIR=/usr/lib/llvm-6.0
cd ..

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
../qt5/configure -opensource -confirm-license -nomake examples -nomake tests
make -j$NJOBS
