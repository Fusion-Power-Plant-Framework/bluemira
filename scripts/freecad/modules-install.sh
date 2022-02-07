set -e

if [ ! -d ../freecad-source ]; then
    cd ..
    git clone https://github.com/FreeCAD/FreeCAD.git freecad-source
    cd freecad-source
    git checkout 0.19.3
fi

cd ..
if [ -d freecad-build ]; then
    rm -r freecad-build
fi
mkdir freecad-build
cd freecad-build
cmake -DBUILD_QT5=TRUE \
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
      -DPYTHON_EXECUTABLE=/usr/bin/python3 \
      -DCMAKE_BUILD_TYPE=Debug \
      ../freecad-source
make -j$(nproc --ignore=2)
