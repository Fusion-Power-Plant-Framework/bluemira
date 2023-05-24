.. _faq:

Frequently Asked Questions
==========================

Conda warns of ``WARNING: overwriting environment variables set in the machine; overwriting variable {'LANG'}``
    The warning is printed every time bluemira is activated, and may stop VSCode's pytest test discoverer from working properly.

    This can be fixed by::

        # This makes sure you're already in the bluemira environment
        conda activate bluemira
        # Run this to stop your bluemira environment overwriting the 'LANG' variable.
        conda env config vars unset LANG

My github ssh connection doesn't connect
    Firstly make sure you have followed our :ref:`SSH key guide <ssh-keys>`.
    A simple fix could be to set a new variable in your `~/.ssh/config`::

        # below Host github.com
        HostName ssh.github.com
        Port 443

    For further ssh issues please see the github `troubleshooting guide
    <https://docs.github.com/en/authentication/troubleshooting-ssh>`_.

The ``pip install -U -e .[dev]`` command has a cryptic error "zsh: no matches found: .[dev]"
    On zsh the square brackets need to be escaped this can either be achieved with
    backslashes ``\[dev\]`` or quotes ``"[dev]"``

How do I update my dependencies?
    For dependencies installed with pip (which are the majority of the dependencies) use the following script::

        bash scripts/update_dependencies.sh

    This script is a wrapper around the below command with some safety checks::

        pip install -r requirements.txt && pip install -r requirements-develop.txt

    Both commands will install the pip dependencies as run in the CI system.
    To update the conda environment completely (only relevant when we update the conda environment
    which is less frequently)::

        conda env update --file conda/environment.yml

On MacOS the ``envsubst`` command cannot be found
    MacOS is not currently a supported OS (it is not part of our CI system)
    however we do have some MacOS users. If you want to use MacOS,
    installing ``gettext`` with homebrew should install ``envsubst``::

        brew install gettext
