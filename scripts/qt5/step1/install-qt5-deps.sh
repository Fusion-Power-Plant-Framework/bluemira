set -e
apt-get update\
    && apt-get upgrade -y \
    && apt-get install -y \
    build-essential perl python3 git '^libxcb.*-dev' libx11-xcb-dev \
    libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev \
    libxkbcommon-x11-dev libclang-dev llvm-dev
