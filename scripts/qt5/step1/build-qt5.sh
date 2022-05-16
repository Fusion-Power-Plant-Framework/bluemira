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
  git checkout 5.15.2
  cd ..
fi

cd qt5
perl init-repository -f --module-subset=essential,qtxmlpatterns,qtsvg
pwd
