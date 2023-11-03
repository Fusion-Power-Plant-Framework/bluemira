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
    # To activate conda's base environment
    source ~/.mambaforge-init.sh
    # To activate your bluemira environment
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

This may result in a(n otherwise harmless) warning about ``overwriting variable {'LANG'}``.
To disable this warning and for any other issues, please refer to our :ref:`FAQ <faq>` for solutions.

Installing PROCESS
------------------

``PROCESS`` is a 0D-1D fusion systems code. More information on ``PROCESS`` can be found in
thier documentation `here <https://ukaea.github.io/PROCESS/>`_ and
`git repository <https://github.com/ukaea/PROCESS>`_.
``Bluemira`` is able perform a ``PROCESS`` run as the initial step in the reactor design.

In order install ``PROCESS`` in your ``bluemira`` environment, run the following:

.. code-block:: bash

    bash scripts/install-process.sh

you can optionally specify which version of ``PROCESS`` is installed by specifying the version tag
or commit hash as an argument.
