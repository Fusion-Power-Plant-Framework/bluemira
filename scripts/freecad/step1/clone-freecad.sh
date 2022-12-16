set -e

if [[ $(basename $PWD) == *"bluemira"* ]]; then
  cd ..
fi

if [ ! -d freecad-source ]; then
  git clone https://github.com/FreeCAD/FreeCAD.git freecad-source
  cd freecad-source
  git checkout 0.20.1
  cd ..
fi
