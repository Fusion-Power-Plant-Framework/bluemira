set -e

NJOBS=$(nproc --ignore=2)

while getopts j: option
do
  case "${option}"
  in
    j) NJOBS=${OPTARG};;
  esac
done

if [[ $(basename $PWD) == *"bluemira"* ]]; then
  cd ..
fi

if [ ! -d pyside-setup ]; then
    git clone --recursive https://code.qt.io/pyside/pyside-setup
    cd pyside-setup
    git checkout 5.14.2
    cd ..
fi

cd pyside-setup
python3 setup.py install --qmake=/usr/local/Qt-5.14.2/bin/qmake --parallel=$NJOBS
