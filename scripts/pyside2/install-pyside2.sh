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

cd pyside-setup
# https://bugreports.qt.io/browse/PYSIDE-1873
python3 setup.py build --qmake=/usr/local/Qt-5.15.2/bin/qmake --parallel=$NJOBS --limited-api=yes
python3 setup.py install --qmake=/usr/local/Qt-5.15.2/bin/qmake --parallel=$NJOBS --limited-api=yes
