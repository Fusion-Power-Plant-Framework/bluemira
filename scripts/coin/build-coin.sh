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

if [ ! -d coin ]; then
  git clone --recurse-submodules https://github.com/coin3d/coin.git
  cd coin
  git checkout Coin-4.0.0
  cd ..
fi

if [ -d coin_build ]; then
  echo "Removing previous coin build"
  rm -rf coin_build
fi

cmake -Hcoin -Bcoin_build -G "Unix Makefiles" -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_BUILD_TYPE=Release -DCOIN_BUILD_DOCUMENTATION=OFF
cmake --build coin_build --target all --config Release -- -j$NJOBS
