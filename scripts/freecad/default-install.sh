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
cmake -DBUILD_QT5=ON -DPYTHON_EXECUTABLE=/usr/bin/python3 -DCMAKE_BUILD_TYPE=Debug ../freecad-source
make -j$(nproc --ignore=2)
