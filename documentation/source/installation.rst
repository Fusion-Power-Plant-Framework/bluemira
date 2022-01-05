Installation
============

Configuring SSH keys
--------------------

First, generate an ssh key (in a Linux terminal). In the directory `~/.ssh/` which you
can get to by running `cd ~/.ssh/`::

  ssh-keygen -t ed25519 -o -a 100 -f <name_of_key>

for example for a github key (the name is up to you)::

  ssh-keygen -t ed25519 -o -a 100 -f id_github

The command will ask you to enter a password, it is not strictly needed but personal 
preference (just hit enter if you dont want one). Two files will have been created `id_github` 
and `id_github.pub` you can see with the command `ls`.

Next uploading the key. For github go to this link https://github.com/settings/keys and 
select add new ssh key and paste the contents of `id_github.pub` into the larger box. To 
see the contents of the file you can use::

    more id_github.pub

Finally to configure to use the key for github locally we need to create a file in `~/.ssh` 
called `config`. To do so you can use vim but nano may have a more familiar interface::

    nano ~/.ssh/config

paste this into the file::

  Host github.com
  User git
  IdentityFile ~/.ssh/id_github
  PasswordAuthentication no

and ctrl x to save and exit (follow the instructions at the bottom). You can test to see 
if everything has been done correctly with::

  ssh -T git@github.com

It will give a message like this if everything is configured properly::

  Hi <user>! You've successfully authenticated, but GitHub does not provide shell access.

For other git websites (eg gitlab) it is exactly the same except you change the name of 
your key everywhere and the Host in the config file. The location to upload the key will 
be somewhere in your profile.

Cloning, setting up the environment, and installing
---------------------------------------------------

Bluemira can be installed into a conda environment using miniforge by running the 
following steps in a mac or Ubuntu command terminal.

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

    # If you are going to be developing bluemira
    python -m pip install --no-cache-dir -e .[dev]
    pre-commit install -f

When you want to activate your ``bluemira`` environment after closing your terminal (or
after ``conda deactivate``) then you can initialise miniforge and activate your
``bluemira`` environment by running:

.. code-block:: bash

    source ~/.miniforge-init.sh
    conda activate bluemira

Installing PROCESS
------------------

``PROCESS`` is a 0D-1D fusion systems code. More information on ``PROCESS`` and how to 
inquire about access can be found `here <https://ccfe.ukaea.uk/resources/process/>`_.
``Bluemira`` is able perform a ``PROCESS`` run as the initial step in the reactor design.

.. note::

    ``PROCESS`` requires gfortran-9, which is not the default version on Ubuntu 18.04. In
    order to make this more recent gfortran version available, if you are using Ubuntu
    18.04 you then must first run:

    .. code-block:: bash

        sudo apt-get update
        sudo apt-get install -y software-properties-common
        sudo add-apt-repository -y ppa:ubuntu-toolchain-r/test
        sudo apt-get update && sudo apt-get install -y gfortran-9
        sudo update-alternatives --install /usr/bin/gfortran gfortran /usr/bin/gfortran-9 30

In order install ``PROCESS`` in your ``bluemira`` environment, run the following:

.. code-block:: bash

    bash scripts/install-process.sh

.. note::

    You will need to have first obtained permissions to be able to clone the ``PROCESS``
    source repository and have set up an ssh key within UKAEA's GitLab.
