set -e

cd /opt/pivy

FORCE="false"

while getopts f option
do
  case "${option}"
  in
    f) FORCE="true";;
  esac
done

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
ninja install
