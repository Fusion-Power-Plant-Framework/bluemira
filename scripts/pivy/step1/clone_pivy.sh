set -e

if [[ $(basename $PWD) == *"bluemira"* ]]; then
  cd ..
fi

if [ ! -d pivy ]; then
  git clone https://github.com/coin3d/pivy.git
  cd pivy
  git checkout 0.6.8
  cd ..
fi
