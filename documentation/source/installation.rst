Installation
============

Cloning the repository
----------------------

You can clone with SSH if you have :ref:`configured your SSH keys <ssh-keys>`:

.. code-block:: bash
    # Clone bluemira
    sudo apt-get install git
    git clone git@github.com:Fusion-Power-Plant-Framework/bluemira.git
    cd bluemira

Otherwise, you can clone with HTTPS:

.. code-block:: bash

    # Clone bluemira
    sudo apt-get install git
    git clone https://github.com/Fusion-Power-Plant-Framework/bluemira.git
    cd bluemira

.. note::

  If you are going to be developing ``bluemira``, it is best to set yourself up via SSH.

Setting up the environment and installing
-----------------------------------------

``Bluemira`` can be installed into a conda environment using mambaforge by running the
following steps in a mac or Ubuntu command terminal.

.. code-block:: bash

    # Install curl if its not available (eg vanilla Ubuntu 22.04)
     sudo apt install -y curl gettext-base
    # Run the conda installation script
    # This installs mambaforge, if not already present, and sets up a bluemira environment
    bash scripts/install-conda.sh
    # To activate your environment
    source ~/.mambaforge-init.sh
    conda activate bluemira

    # If you are going to be developing bluemira
    python -m pip install --no-cache-dir -e .'[dev]'
    pre-commit install -f

When you want to activate your ``bluemira`` environment after closing your terminal (or
after ``conda deactivate``) then you can initialise mambaforge and activate your
``bluemira`` environment by running:

.. code-block:: bash

    source ~/.mambaforge-init.sh
    conda activate bluemira

Please refer to our :ref:`FAQ <faq>` if you run into issues.

Installing PROCESS
------------------

``PROCESS`` is a 0D-1D fusion systems code. More information on ``PROCESS`` and how to
inquire about access can be found `here <https://ccfe.ukaea.uk/resources/process/>`_.
``Bluemira`` is able perform a ``PROCESS`` run as the initial step in the reactor design.

.. note::

    The ``PROCESS`` build requires gfortran-9, gcc and make.
    gfortran-9 is not the default version on Ubuntu 18.04.
    In order to install the dependencies and make this more recent gfortran version available,
    if you are using Ubuntu 18.04, you must first run:

    .. code-block:: bash

        sudo apt-get update
        sudo apt-get install -y software-properties-common
        sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
        sudo apt-get update && sudo apt-get install -y gfortran-9 gcc make
        sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-9 30

In order install ``PROCESS`` in your ``bluemira`` environment, run the following:

.. code-block:: bash

    bash scripts/install-process.sh

.. note::

    You will need to have first obtained permissions to be able to clone the ``PROCESS``
    source repository and have set up an ssh key within UKAEA's GitLab.
