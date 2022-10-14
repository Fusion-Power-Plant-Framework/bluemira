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
        git \
        curl \
        gettext-base

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv ${VIRTUAL_ENV}
ENV PATH "${VIRTUAL_ENV}/bin:$PATH"

WORKDIR /opt/bluemira

COPY scripts/install-conda.sh ./scripts/
COPY conda conda
 
COPY requirements* ./
# Update and install dependencies available through pip
RUN bash ./scripts/install-conda.sh 

# Build and install fenicsx
COPY scripts/fenicsx ./scripts/fenicsx/
RUN bash scripts/fenicsx/install-fenicsx-deps.sh
RUN bash -c "source ~/.miniforge-init.sh && conda activate bluemira && bash scripts/fenicsx/install-fenicsx.sh"
ENV VIRTUAL_ENV=/root/miniforge3/envs/bluemira/bin/python
ENV PATH "${VIRTUAL_ENV}/bin:$PATH"
RUN bash -c "source ~/.miniforge-init.sh && conda activate bluemira && git clone https://github.com/FEniCS/dolfinx.git \
    && cd dolfinx/cpp \
    && mkdir build \
    && cd build \
    && cmake -G Ninja .. \
    && ninja install \
    &&  cd ../.. "

RUN bash -c "source ~/.miniforge-init.sh && conda activate bluemira && pip install dolfinx/python/"
RUN apt-get install libxcursor-dev libxft-dev libxinerama-dev -y
WORKDIR /root/
