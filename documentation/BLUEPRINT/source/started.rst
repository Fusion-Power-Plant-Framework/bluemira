Getting started
===============

Installation
------------

BLUEPRINT is intended to be installed on Ubuntu. It is tested on Ubuntu 18.04. BLUEPRINT
has several system, python and binary dependencies. The python dependencies are installed
into a virtual environment and not the system Python. If you are running on Windows, we
suggest you first `set up WSL <windows.html>`_.

These steps assume that you are in a terminal with the current directory at the level
below where you'd like to install BLUEPRINT e.g. if you'd like to install it in
~/code/BLUEPRINT then first create ~/code and change directory to that location. The
scripts create directories in the parent directory (..) for dependencies they build.

Installation steps:

.. code-block:: bash

    # If you have conda active first do: `conda deactivate`
    # If you have a virtual environment for another project active first do: `deactivate`
    sudo apt install git
    git clone https://gitlab.com/CoronelBuendia/BLUEPRINT.git
    cd BLUEPRINT
    sudo bash install-ubuntu-runtime.sh
    sudo bash install-ubuntu.sh
    python3 -m venv .env
    . .env/bin/activate
    python -m pip install --upgrade pip setuptools wheel
    sudo bash install-pythonocc.sh
    find  . -type d  -user root -printf "sudo chown -R $USER \"%p\"\n" | sh
    python -m pip install --no-cache-dir -e .

    # If you are going to be developing BLUEPRINT
    python -m pip install --no-cache-dir -e .[dev]
    pre-commit install -f

    # To test the install
    cd examples/cad
    python make_some_cad.py

Installation in conda (experimental)
************************************

We are currently working on an alternative installation approach, using ``conda``
(specifically `miniforge <https://github.com/conda-forge/miniforge>`_).
If you would like to install BLUEPRINT this way then run the following steps:

.. code-block:: bash


    # Clone BLUEPRINT
    sudo apt-get install git
    git clone git@gitlab.com:CoronelBuendia/BLUEPRINT.git
    cd BLUEPRINT

    # Install pyenv
    # This step is optional
    # pyenv is a convenient way of managing multiple python versions.
    # We will use it to install miniforge.
    # If you already have pyenv installed then skip this step
    sudo apt-get install curl
    curl https://pyenv.run | bash

    # Follow the instructions in the command prompt to activate pyenv

    # Install miniforge
    # This step is optional
    # If you already have a valid conda flavour installed the you can install the
    # blueprint environment within that
    pyenv install miniforge3-4.10
    envsubst '$HOME' < conda/miniforge-init.sh > ~/.miniforge-init.sh
    source ~/.miniforge-init.sh  # this will put you in a (base) conda environment

    # Install system dependencies
    sudo apt-get update
    sudo apt-get install -y build-essential qt5-default

    # Install environment and BLUEPRINT
    conda env create --file conda/environment.yml
    conda activate blueprint  # run this whenever you want your (blueprint) environment
    python -m pip install --no-cache-dir -e .

    # If you are going to be developing BLUEPRINT
    python -m pip install --no-cache-dir -e .[dev]
    pre-commit install -f

    # To test the install
    cd examples/cad
    python make_some_cad.py

When you want to activate your BLUEPRINT environment after closing your terminal (or
after ``conda deactivate``) then you can initialise miniforge and activate your
BLUEPRINT environment by running:

.. code-block:: bash

    source ~/.miniforge-init.sh
    conda activate blueprint

Running BLUEPRINT
-----------------
To test your installation, we suggest you carry out a full run for a typical large single
null machine:

.. code-block:: bash

	cd examples

	# This will build a full reactor object: R in an interactive python console
	python -i EUDEMO.py


Then, in the Python console, you can manipulate the reactor

.. code-block:: python

	# You can list its parameters
	R.params

	# Plot the reactor
	R.plot_xz()

	# And build and show the CAD model
	R.show_CAD()

If this all works, you are ready to build your own reactors.

Running examples in Jupyter
---------------------------

BLUEPRINT ships with a set of examples that demonstrate how to use various features.
These are mostly available as `Jupyter notebooks <https://jupyter.org/>`_. If you would
like to run the examples in a local Jupyter server then you can start that server in the
base directory of the BLUEPRINT repo by running the below command after installing
BLUEPRINT:

.. code-block:: bash

    jupyter notebook

This should open your web browser, in which you can navigate to the examples directory
and explore the various examples that are available, and run them by clicking on the
desired .ipynb file.

Installing PROCESS
------------------

BLUEPRINT is able perform a PROCESS run as the initial step in the reactor design.

.. note::
    PROCESS requires gfortran-9, which is not the default version on Ubuntu 18.04. In
    order to make this more recent gfortran version available, you must first run:

    ..code-block:: bash
        sudo apt-get update
        sudo apt-get install -y software-properties-common
        sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
        sudo apt-get update && apt-get install -y gfortran-9
        sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-9 30

    **If you have not already run the above then the follow will automatically change
    your default gfortran version.**

In order to give BLUEPRINT visibility of an installed version of PROCESS in a virtual
environment then run the following:

.. code-block:: bash

    sudo bash install-process-ubuntu.sh
    bash install-process.sh

Or if you are using ``conda`` then run:

.. code-block:: bash

    sudo bash install-process-ubuntu.sh
    bash install-process-conda.sh

.. note::
    You will need to have first obtained permissions to be able to clone the PROCESS
    source repository.
