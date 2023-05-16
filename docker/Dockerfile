FROM ubuntu:22.04 AS build_base

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

RUN add-apt-repository ppa:freecad-maintainers/freecad-stable \
    && apt-get update \
    && apt-get -y install freecad

ENV VIRTUAL_ENV /opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH "${VIRTUAL_ENV}/bin:$PATH"

WORKDIR /opt/build

# COPY requirements.txt ./requirements.txt
# RUN python3 -m pip install --no-cache-dir -r requirements.txt

# COPY requirements-develop.txt ./requirements-develop.txt
# RUN python3 -m pip install --no-cache-dir -r requirements-develop.txt

# RUN python3 -m pip install --no-cache-dir pytest-xdist

# Build and install fenics
COPY scripts/fenics ./scripts/fenics/
RUN bash scripts/fenics/install-fenics-deps.sh
RUN bash scripts/fenics/install-fenics.sh

# h5py is a dependency of meshio. We need to manually install it or we run into
# a segfault when meshio attempts to import it, even though we can import h5py
# separately without issue.
# see https://github.com/Fusion-Power-Plant-Framework/bluemira/issues/1258 and
# https://github.com/h5py/h5py/issues/695
COPY scripts/h5py ./scripts/h5py
RUN bash scripts/h5py/install-h5py.sh

FROM ubuntu:22.04

# Dolfin needs help finding Boost runtime libaries
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
ENV BLUEMIRA_ENV "CONTAINER"

# COPY --from=build_base /opt/venv/ /opt/venv/
# RUN chown user:user /opt/venv
COPY --from=build_base --chown=user /opt/venv/ /opt/venv/
ENV PATH "${VIRTUAL_ENV}/bin:$PATH"

COPY --from=build_base /usr /usr
COPY --from=build_base /etc /etc

RUN useradd -ms /bin/bash user
WORKDIR /home/user/bluemira

COPY . .
COPY main.py .

RUN pip install --no-cache-dir -e .'[dev]'

USER user
ENTRYPOINT [ "tail","-f", "/dev/null" ]
