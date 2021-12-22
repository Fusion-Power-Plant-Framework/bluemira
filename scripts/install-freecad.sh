set -e

if [[ $(basename $PWD) == *"bluemira"* ]]; then
  cd ..
fi

if [ ! -d freecad-source ]; then
  git clone https://github.com/FreeCAD/FreeCAD.git freecad-source
  cd freecad-source
  git checkout 0.19.3
  cd ..
fi

if [ ! -d pyside2-tools ]; then
  echo "Missing pyside2-tools repo." >> /dev/stderr
  exit 1
fi

if [ -d freecad-build ]; then
  echo "Removing previous FreeCAD build"
  rm -rf freecad-build
fi

PYTHON_VERSION=`python -c 'import sys; version=sys.version_info[:2]; print("{0}.{1}".format(*version))'`
PYTHON_BIN_DIR=$(dirname $(which python))
PYTHON_PACKAGES_DIR=$(readlink --canonicalize $PYTHON_BIN_DIR/../lib/python$PYTHON_VERSION/site-packages)
PYTHON_FREECAD_DIR=$PYTHON_PACKAGES_DIR/freecad

sLD_LIBRARY_PATH=/opt/qt514/lib/
LIBRARY_PATH=/opt/qt514/lib/
export CPLUS_INCLUDE_PATH=/opt/qt514/include/

mkdir freecad-build
cd freecad-build
cmake -DBUILD_QT5=TRUE \
      -DBUILD_GUI=TRUE \  # Linking error when building the GUI (needed for display)
      -DPYTHON_EXECUTABLE=$(which python) \
      -DPYSIDE_INCLUDE_DIR=$PYTHON_PACKAGES_DIR/PySide2/include \
      -DCMAKE_PREFIX_PATH=/opt/qt514/lib/cmake \
      -DPYSIDE2UICBINARY=$PYTHON_PACKAGES_DIR/PySide2/uic \
      -DPYSIDE2RCCBINARY=$PYTHON_PACKAGES_DIR/PySide2/rcc \
      ../freecad-source

# Crashes for me if I try to use more than one core (possibly OOM).
# Also shouldn't force a specific number of build threads.
make -j4

# FreeCAD doesn't give us much help when putting files into python, so mock up some
# infrastructure in out site-packages directory.

echo "Installing FreeCAD for Python version: $PYTHON_VERSION"
echo "Using installation directory: $PYTHON_FREECAD_DIR"

if [ -d $PYTHON_FREECAD_DIR ]; then
  echo "WARNING: FreeCAD install exists and will be overwritten"
  rm -rf $PYTHON_FREECAD_DIR
fi

mkdir -p $PYTHON_FREECAD_DIR/lib

echo "import os
import sys

sys.path += __path__
sys.path += [os.path.join(__path__[0], 'lib')]
import FreeCAD as app" >> $PYTHON_FREECAD_DIR/__init__.py

cp ../freecad-build/lib/*.so $PYTHON_FREECAD_DIR/lib
cp ../freecad-build/Mod/Part/Part.so $PYTHON_FREECAD_DIR/lib
cp -r ../freecad-build/Mod/Part/BOPTools $PYTHON_FREECAD_DIR
