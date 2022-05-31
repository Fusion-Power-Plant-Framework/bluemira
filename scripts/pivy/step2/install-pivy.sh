set -euxo

cd /opt/pivy

NJOBS=$(nproc --ignore=2)

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
cmake -G Ninja ..
ninja install -j$NJOBS
