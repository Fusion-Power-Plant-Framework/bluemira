FROM ubuntu:latest AS base

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
        python3-dev \
        python3-venv \
        build-essential \
        software-properties-common \
        cmake \
        git

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH "${VIRTUAL_ENV}/bin:$PATH"

WORKDIR /opt/bluemira

# Build and install Qt5 (5.15.5)
FROM base as build_deps
COPY scripts/qt5/step1 ./scripts/qt5/step1
RUN bash scripts/qt5/step1/install-qt5-deps.sh \
    && bash scripts/qt5/step1/build-qt5.sh

COPY scripts/qt5/step2 ./scripts/qt5/step2
RUN bash scripts/qt5/step2/build-qt5-2.sh \
    && bash scripts/qt5/step2/install-qt5.sh

# Build and install PySide2 and shiboken (5.15.2.1)
COPY scripts/pyside2/clone-pyside2.sh ./scripts/pyside2/clone-pyside2.sh
RUN bash scripts/pyside2/clone-pyside2.sh

COPY scripts/pyside2/install-pyside2.sh ./scripts/pyside2/install-pyside2.sh
RUN bash scripts/pyside2/install-pyside2.sh

# Build and install coin (4.0.0)
COPY scripts/coin/ ./scripts/coin/
RUN bash scripts/coin/install-coin-deps.sh \
    && bash scripts/coin/build-coin.sh \
    && bash scripts/coin/install-coin.sh

# Build and install pivy (0.6.7)
COPY scripts/pivy/step1 ./scripts/pivy/step1
RUN bash scripts/pivy/step1/clone_pivy.sh && \
 bash scripts/pivy/step1/install-pivy-deps.sh

# Build and install pivy (0.6.7)
COPY scripts/pivy/step2 ./scripts/pivy/step2
RUN bash scripts/pivy/step2/install-pivy.sh

# Build and install freecad (0.20.0)
COPY scripts/freecad/step1 ./scripts/freecad/step1
RUN bash scripts/freecad/step1/install-freecad-deps.sh \
    && bash scripts/freecad/step1/clone-freecad.sh

RUN python -m pip install --upgrade pip setuptools wheel pybind11-global

COPY scripts/freecad/install-freecad.sh ./scripts/freecad/install-freecad.sh
RUN bash scripts/freecad/install-freecad.sh

COPY requirements.txt .
# Update and install dependencies available through pip
RUN pip install -i https://test.pypi.org/simple/ 'CoolProp==6.4.2.dev0' \
    && python -m pip install -r requirements.txt

# Build and install fenics
COPY scripts/fenics ./scripts/fenics/
RUN bash scripts/fenics/install-fenics-deps.sh
RUN bash scripts/fenics/install-fenics.sh

# QT5 has some not standard lib locations which freecad install doesnt remember
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/Qt-5.15.5/lib
# Dolfin needs help finding Boost runtime libaries
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib

FROM base as release
COPY --from=build_deps /usr /usr
COPY --from=build_deps /etc /etc
RUN pip install git+https://github.com/Fusion-Power-Plant-Framework/bluemira.git@v0.1.0

RUN useradd -ms /bin/bash user
COPY --from=build_deps --chown=user /opt/venv/ /opt/venv/
RUN chown user:user /opt/venv
USER user
WORKDIR /home/user

FROM base as develop
COPY requirements-develop.txt .
RUN pip install --no-cache-dir -r requirements-develop.txt
RUN apt-get install git -y
