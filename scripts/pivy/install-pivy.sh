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

if [ ! -d pivy ]; then
  git clone https://github.com/coin3d/pivy.git
  cd pivy
  git checkout 0.6.7
  cd ..
fi

cd pivy

if [ -d build ]; then
  if ${FORCE}; then
    echo "Removing previous pivy build"
    rm -rf build
  else
    echo "Existing pivy build exists. Use the flag -f if you want to rebuild."
    exit 1
  fi
fi

# >0.6.6
mkdir build && cd build
cmake ..
make -j$NJOBS
make install
# 0.6.5
# python3 setup.py build
# python3 setup.py install
# File "/opt/venv/lib/python3.10/site-packages/Pivy-0.6.5-py3.10.egg/pivy/coin.py", line 21, in <module>
# ImportError: cannot import name '_coin' from partially initialized module 'pivy'
# (most likely due to a circular import) (/opt/venv/lib/python3.10/site-packages/Pivy-0.6.5-py3.10.egg/pivy/__init__.py)
