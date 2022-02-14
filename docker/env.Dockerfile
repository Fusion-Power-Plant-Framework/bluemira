FROM ubuntu:latest AS base

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
        python3-dev \
        python3.8-venv \
        build-essential \
        software-properties-common \
        cmake \
        git

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH "${VIRTUAL_ENV}/bin:$PATH"

WORKDIR /opt/bluemira

COPY requirements.txt .

# Update and install dependencies available through pip
RUN python -m pip install --upgrade pip setuptools wheel pybind11 \
    && python -m pip install -r requirements.txt

# Build and install Qt5 (5.14.2)
COPY scripts/qt5/ ./scripts/qt5/
RUN bash scripts/qt5/install-qt5-deps.sh \
    && bash scripts/qt5/build-qt5.sh \
    && bash scripts/qt5/install-qt5.sh

# Build and install PySide2 and shiboken (5.14.2)
COPY scripts/pyside2/ ./scripts/pyside2/
RUN bash scripts/pyside2/install-pyside2.sh

# Build and install coin (4.0.0)
COPY scripts/coin/ ./scripts/coin/
RUN bash scripts/coin/install-coin-deps.sh \
    && bash scripts/coin/build-coin.sh \
    && bash scripts/coin/install-coin.sh

# Build and install pivy (0.6.6)
COPY scripts/pivy/ ./scripts/pivy/
RUN bash scripts/pivy/install-pivy-deps.sh \
    && bash scripts/pivy/install-pivy.sh

# Build and install freecad (0.19.3)
COPY scripts/freecad/ ./scripts/freecad/
RUN bash scripts/freecad/install-freecad-deps.sh \
    && bash scripts/freecad/install-freecad.sh

# Build and install pythonocc (approx 7.5.2)
COPY scripts/occ/ ./scripts/occ/
RUN bash scripts/occ/install-occ-deps.sh \
    && bash scripts/occ/install-occ.sh

FROM base as dev-base

COPY requirements-develop.txt .

RUN pip install -r requirements-develop.txt

FROM base as env

RUN useradd -ms /bin/bash user

COPY --from=base --chown=user /opt/venv/ /opt/venv/

USER user
WORKDIR /home/user

FROM dev-base AS dev

RUN useradd -ms /bin/bash user

COPY --from=dev-base --chown=user /opt/venv/ /opt/venv/

USER user
WORKDIR /home/user
