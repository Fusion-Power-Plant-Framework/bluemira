Getting Started
===============

Installation
------------

Bluemira can be installed into a conda environment using miniforge by running the
following steps in a mac or Ubuntu command terminal:


.. code-block:: bash


    # Clone bluemira
    sudo apt-get install git
    git clone git@github.com:Fusion-Power-Plant-Framework/bluemira.git
    cd bluemira

    # Run the conda installation script
    # This installs miniforge, if not already present, and sets up a bluemira environment
    bash scripts/install-conda.sh

    # To activate your environment
    source ~/.miniforge-init.sh
    conda activate bluemira

    # If you are going to be developing BLUEPRINT
    python -m pip install --no-cache-dir -e .[dev]
    pre-commit install -f

    # To test the install
    cd examples/cad
    python make_some_cad.py

When you want to activate your bluemira environment after closing your terminal (or
after ``conda deactivate``) then you can initialise miniforge and activate your
BLUEPRINT environment by running:

.. code-block:: bash

    source ~/.miniforge-init.sh
    conda activate bluemira

PROCESS Installation
^^^^^^^^^^^^^^^^^^^^

Bluemira is able perform a PROCESS run as the initial step in the reactor design.

.. note::
    PROCESS requires gfortran-9, which is not the default version on Ubuntu 18.04. In
    order to make this more recent gfortran version available, if you are using Ubuntu
    18.04 you then must first run:

    ..code-block:: bash
        sudo apt-get update
        sudo apt-get install -y software-properties-common
        sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
        sudo apt-get update && apt-get install -y gfortran-9
        sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-9 30

In order install PROCESS in your bluemira environment, run the following:

.. code-block:: bash

    bash install-process.sh

.. note::
    You will need to have first obtained permissions to be able to clone the PROCESS
    source repository.
