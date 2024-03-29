set -e

apt-get update
apt-get install -y \
    cmake cmake-gui libboost-date-time-dev libboost-dev libboost-filesystem-dev \
    libboost-graph-dev libboost-iostreams-dev libboost-program-options-dev \
    libboost-python-dev libboost-regex-dev libboost-serialization-dev \
    libboost-thread-dev libeigen3-dev libgts-bin libgts-dev libkdtree++-dev \
    libmedc-dev libocct-data-exchange-dev libocct-ocaf-dev libocct-visualization-dev \
    libopencv-dev libproj-dev libspnav-dev libvtk9-dev libx11-dev libxerces-c-dev \
    libzipios++-dev occt-draw python3-dev libclang-dev llvm ninja-build
