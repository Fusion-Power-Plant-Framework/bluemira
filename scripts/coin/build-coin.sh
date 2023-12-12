set -e

FORCE="false"

while getopts f option
do
  case "${option}"
  in
    f) FORCE="true";;
  esac
done

if [[ $(basename $PWD) == *"bluemira"* ]]; then
  cd ..
fi

if [ ! -d coin ]; then
  git clone --recurse-submodules https://github.com/coin3d/coin.git
  cd coin
  git checkout v4.0.2
  cd ..
fi

if [ -d coin_build ]; then
  if ${FORCE}; then
    echo "Removing previous coin build"
    rm -rf coin_build
  else
    echo "Existing coin build exists. Use the flag -f if you want to rebuild."
    exit 1
  fi
fi

cmake -Hcoin -Bcoin_build -G Ninja -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DCMAKE_BUILD_TYPE=Release -DCOIN_BUILD_DOCUMENTATION=OFF
cmake --build coin_build --target all --config Release
