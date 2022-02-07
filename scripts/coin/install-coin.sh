set -e

if [[ $(basename $PWD) == *"bluemira"* ]]; then
  cd ..
fi

if [ ! -d coin/build ]; then
  echo "Could not find build for coin" >> /dev/stderr
  exit 1
fi

cmake --build coin_build --target install --config Release -- -j4
