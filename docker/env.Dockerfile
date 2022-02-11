FROM ubuntu:latest AS base

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y python3-dev python3.8-venv build-essential cmake git

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH "${VIRTUAL_ENV}/bin:$PATH"

WORKDIR /opt/bluemira

COPY requirements.txt .

# Update and install dependencies available through pip
RUN python -m pip install --upgrade pip setuptools wheel pybind11 \
    && python -m pip install -r requirements.txt

COPY scripts/ ./scripts

# Build and install Qt5 (5.14.2)
RUN bash scripts/qt5/install-qt5-deps.sh \
    && bash scripts/qt5/build-qt5.sh \
    && bash scripts/qt5/install-qt5.sh

# Build and install PySide2 and shiboken (5.14.2)
RUN bash scripts/pyside2/install-pyside2.sh

# Build and install coin (4.0.0)
RUN bash scripts/coin/install-coin-deps.sh \
    && bash scripts/coin/build-coin.sh \
    && bash scripts/coin/install-coin.sh

# Build and install pivy (0.6.6)
RUN bash scripts/pivy/install-pivy-deps.sh \
    && bash scripts/pivy/install-pivy.sh

# Build and install freecad (0.19.3)
RUN bash scripts/freecad/install-freecad-deps.sh \
    && bash scripts/freecad/install-freecad.sh

# [Optional] Build and install pythonocc (approx 7.5.2)
RUN bash scripts/occ/install-occ-deps.sh \
    && bash scripts/occ/install-occ.sh

FROM ubuntu:latest as env

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y python3.8-venv

COPY --from=base /opt/venv /opt/venv
COPY --from=base /usr/local/Qt-5.14.2/lib /usr/local/Qt-5.14.2/lib
COPY --from=base /lib/x86_64-linux-gnu /lib/x86_64-linux-gnu

ENV PATH "/opt/venv/bin:$PATH"

RUN useradd -ms /bin/bash user
USER user
WORKDIR /home/user

FROM env AS dev

WORKDIR /opt/bluemira

COPY requirements-develop.txt .

RUN python -m pip install -r requirements-develop.txt

WORKDIR /home/user
