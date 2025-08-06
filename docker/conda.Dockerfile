# Dockerfile containing Bluemira and PROCESS installed into a conda environment.
#
# Build arguments:
#   BLUEMIRA_VERSION:  The tag or branch of the installed bluemira.
#                      Note that this does not set the bluemira version, but
#                      informs the build of the version being built so that
#                      the image can be labelled accordingly.
#   BLUEMIRA_REVISION: The git revision of installed bluemira.
#                      Note that this does not set the bluemira revision, but
#                      informs the build of the revision being built so that
#                      the image can be labelled accordingly.
#   PROCESS_VERSION:   The version of PROCESS to install.
#   PYTHON_VERSION:    The version of Python to use in the conda environment.
#                      This supports wildcards using '*'.
#
# Bluemira is installed at '/opt/bluemira' and PROCESS is installed at
# '/opt/process'. The bluemira conda environment is activated by default. The
# user is left as 'root'.
FROM condaforge/miniforge3:25.3.1-0@sha256:d316fd5f637251f9294c734cac3040fd7a8ea012225b84875d5454dde62f8fd3

ARG BLUEMIRA_VERSION="unknown"
ARG BLUEMIRA_REVISION="unknown"
ARG PROCESS_VERSION="v3.1.0"
ARG PYTHON_VERSION="3.11.*"

# OCI specs: https://github.com/opencontainers/image-spec
LABEL org.opencontainers.image.title="bluemira"
LABEL org.opencontainers.image.description="An integrated inter-disciplinary design tool for future fusion \
    reactors, incorporating several modules, some of which rely on \
    other codes, to carry out a range of typical conceptual fusion \
    reactor design activities."
LABEL org.opencontainers.image.url="https://github.com/Fusion-Power-Plant-Framework/bluemira"
LABEL org.opencontainers.image.source="https://github.com/Fusion-Power-Plant-Framework/bluemira"
LABEL org.opencontainers.image.version="${BLUEMIRA_VERSION}"
LABEL org.opencontainers.image.revision="${BLUEMIRA_REVISION}"
LABEL org.opencontainers.image.licenses="LGPL-2.1-or-later"

ENV BLUEMIRA_ROOT=/opt/bluemira
ENV LANG=C
ENV LC_ALL=C
ENV PYTHONIOENCODING="utf-8"

COPY . "${BLUEMIRA_ROOT}"

RUN mamba env create python="${PYTHON_VERSION}" git git-lfs xorg-libxft \
        --name bluemira \
        --yes \
        --file "${BLUEMIRA_ROOT}/conda/environment.yml" \
    && conda clean --tarballs --index-cache --packages --yes \
    && find "${CONDA_DIR}" -follow -type f -name '*.a' -delete \
    && find "${CONDA_DIR}" -follow -type f -name '*.pyc' -delete \
    && conda clean --force-pkgs-dirs --all --yes

# Use 'conda run' for subsequent 'RUN' instructions.
SHELL ["/opt/conda/bin/conda", "run", "-n", "bluemira", "/bin/bash", "-c"]

RUN git lfs install

# Install PROCESS.
ARG PROCESS_VERSION=v3.1.0
ENV PROCESS_ROOT=/opt/process
RUN git clone \
        --branch ${PROCESS_VERSION} \
        --depth 1 \
        https://github.com/ukaea/process.git \
        "${PROCESS_ROOT}" \
    && apt-get update \
    && apt-get install -y build-essential gfortran --no-install-recommends \
    && pip install --no-cache-dir --upgrade 'setuptools<74' \
    && cmake \
        -S "${PROCESS_ROOT}" \
        -B "${PROCESS_ROOT}/build" \
        -DRELEASE=TRUE \
        -DCMAKE_BUILD_TYPE=Release \
    && cmake --build "${PROCESS_ROOT}/build" \
    && apt-get purge -y build-essential gfortran \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# The base image activates the base enviornment in .bashrc. Activate bluemira
# instead.
RUN sed -i -e 's/"conda activate base"/"conda activate bluemira"/g' ~/.bashrc \
    && sed -i -e 's/"conda activate base"/"conda activate bluemira"/g' /etc/skel/.bashrc
