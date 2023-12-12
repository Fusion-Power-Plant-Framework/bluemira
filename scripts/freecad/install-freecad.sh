set -e

FORCE="false"
PYSIDE_VERSION="libpyside2.abi3.so.5.15"
QT_VERSION="Qt-5.15.5"

while getopts f option
do
  case "${option}"
  in
    f) FORCE="true"
  esac
done

if [[ $(basename $PWD) == *"bluemira"* ]]; then
  cd ..
fi

if [ -d freecad-build ]; then
  if ${FORCE}; then
    echo "Removing previous FreeCAD build."
    rm -rf freecad-build
  else
    echo "Existing freecad build exists. Use the flag -f if you want to rebuild."
    exit 1
  fi
fi

PYTHON_VERSION=`python -c 'import sys; version=sys.version_info[:2]; print("{0}.{1}".format(*version))'`
PYTHON_BIN=$(which python)
PYTHON_BIN_DIR=$(dirname $PYTHON_BIN)
PYTHON_PACKAGES_DIR=$(python -c 'import site; print(site.getsitepackages()[0])')
PYTHON_FREECAD_DIR=$PYTHON_PACKAGES_DIR/freecad

mkdir -p $PYTHON_FREECAD_DIR
mkdir freecad-build
cd freecad-build

cmake -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_QT5=TRUE \
      -DBUILD_GUI=TRUE \
      -DBUILD_FEM=FALSE \
      -DBUILD_SANDBOX=FALSE \
      -DBUILD_TEMPLATE=FALSE \
      -DBUILD_ADDONMGR=FALSE \
      -DBUILD_ARCH=FALSE \
      -DBUILD_ASSEMBLY=FALSE \
      -DBUILD_COMPLETE=FALSE \
      -DBUILD_DRAFT=FALSE \
      -DBUILD_DRAWING=FALSE \
      -DBUILD_IDF=FALSE \
      -DBUILD_IMAGE=FALSE \
      -DBUILD_IMPORT=FALSE \
      -DBUILD_INSPECTION=FALSE \
      -DBUILD_JTREADER=FALSE \
      -DBUILD_MATERIAL=FALSE \
      -DBUILD_MESH=FALSE \
      -DBUILD_MESH_PART=FALSE \
      -DBUILD_FLAT_MESH=FALSE \
      -DBUILD_OPENSCAD=FALSE \
      -DBUILD_PART=TRUE \
      -DBUILD_PART_DESIGN=FALSE \
      -DBUILD_PATH=FALSE \
      -DBUILD_PLOT=FALSE \
      -DBUILD_POINTS=FALSE \
      -DBUILD_RAYTRACING=FALSE \
      -DBUILD_REVERSEENGINEERING=FALSE \
      -DBUILD_ROBOT=FALSE \
      -DBUILD_SHIP=FALSE \
      -DBUILD_SHOW=TRUE \
      -DBUILD_SKETCHER=FALSE \
      -DBUILD_SPREADSHEET=FALSE \
      -DBUILD_START=FALSE \
      -DBUILD_TEST=FALSE \
      -DBUILD_TECHDRAW=FALSE \
      -DBUILD_TUX=FALSE \
      -DBUILD_WEB=FALSE \
      -DBUILD_SURFACE=FALSE \
      -DBUILD_VR=FALSE \
      -DBUILD_CLOUD=FALSE \
      -DFREECAD_USE_PYBIND11:BOOL=ON \
      -DPYTHON_EXECUTABLE=$PYTHON_BIN \
      -DPYSIDE_INCLUDE_DIR=$PYTHON_PACKAGES_DIR/PySide2/include \
      -DPYSIDE_LIBRARY=$PYTHON_PACKAGES_DIR/PySide2/$PYSIDE_VERSION \
      -DCMAKE_PREFIX_PATH=/usr/local/$QT_VERSION/lib/cmake \
      -DPYSIDE2UICBINARY=$PYTHON_PACKAGES_DIR/PySide2/uic \
      -DPYSIDE2RCCBINARY=$PYTHON_PACKAGES_DIR/PySide2/rcc \
      -DCMAKE_INSTALL_PREFIX=$PYTHON_FREECAD_DIR \
      -DFREECAD_USE_EXTERNAL_ZIPIOS=TRUE \
      -DINSTALL_TO_SITEPACKAGES=TRUE \
      ../freecad-source

ninja install

# Installing lib into different location so overwrite init
echo "import os
import sys

sys.path += __path__
sys.path += [os.path.join(__path__[0], 'lib')]
import FreeCAD as app" > $PYTHON_FREECAD_DIR/__init__.py
