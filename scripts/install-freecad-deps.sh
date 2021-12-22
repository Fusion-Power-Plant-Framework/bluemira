add-apt-repository -y ppa:freecad-maintainers/freecad-stable
add-apt-repository -y ppa:beineri/opt-qt-5.14.2-focal
apt-get update
apt-get install -y \
    cmake cmake-gui libboost-date-time-dev libboost-dev libboost-filesystem-dev \
    libboost-graph-dev libboost-iostreams-dev libboost-program-options-dev \
    libboost-python-dev libboost-regex-dev libboost-serialization-dev \
    libboost-thread-dev libcoin-dev libeigen3-dev libgts-bin libgts-dev libkdtree++-dev \
    libmedc-dev libocct-data-exchange-dev libocct-ocaf-dev libocct-visualization-dev \
    libopencv-dev libproj-dev \
    # libpyside2-dev \
    # libqt5opengl5-dev libqt5svg5-dev \
    # libqt5webkit5-dev libqt5x11extras5-dev libqt5xmlpatterns5-dev libqt5quick5s \
    libshiboken2-dev \
    libspnav-dev libvtk7-dev libx11-dev libxerces-c-dev libzipios++-dev occt-draw \
    # pyside2-tools \
    python3-dev \
    # python3-matplotlib python3-pivy \
    # python3-ply \
    # python3-pyside2.qtcore python3-pyside2.qtgui python3-pyside2.qtsvg \
    # python3-pyside2.qtwidgets python3-pyside2.qtnetwork \
    # python3-markdown python3-git \
    # qtbase5-dev qttools5-dev qtdeclarative5-dev \
apt-get install -y qt514-meta-full
