set -e

cd /opt/pivy

PATCH_DIR=/opt/bluemira/scripts/pivy/step2/
# pivy#91 changes some interface that produces
# "SystemError: <built-in function cast> returned NULL without setting an exception"
patch -R -p 1 -f < "$PATCH_DIR/pivy1.patch"

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
