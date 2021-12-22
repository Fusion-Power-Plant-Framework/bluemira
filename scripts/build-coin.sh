set -e

if [[ $(basename $PWD) == *"bluemira"* ]]; then
  cd ..
fi

if [ ! -d coin ]; then
  git clone --recurse-submodules https://github.com/coin3d/coin.git
  git checkout Coin-4.0.0
fi

if [ -d coin_build ]; then
  echo "Removing previous coin build"
  rm -rf coin_build
fi

cmake -Hcoin -Bcoin_build -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=/usr/local \
            -DCMAKE_BUILD_TYPE=Release -DCOIN_BUILD_DOCUMENTATION=OFF
cmake --build coin_build --target all --config Release -- -j4
