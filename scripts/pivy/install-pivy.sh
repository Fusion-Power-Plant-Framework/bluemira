set -e

if [[ $(basename $PWD) == *"bluemira"* ]]; then
  cd ..
fi

if [ ! -d pivy ]; then
  git clone git@github.com:coin3d/pivy.git
  cd pivy
  git checkout 0.6.6
  cd ..
fi

cd pivy

if [ -d build ]; then
  echo "Removing previous pivy build"
  rm -rf build
fi

pip install wheel
python setup.py bdist_wheel
pip install dist/*.whl
