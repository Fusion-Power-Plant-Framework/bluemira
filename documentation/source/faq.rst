.. _faq:

Frequently Asked Questions
==========================

My github ssh connection doesnt connect
    Firstly make sure you have followed our :ref:`SSH key guide <ssh-keys>`.
    A simple fix could be to set a new variable in you `~/.ssh/config`::

        # below Host github.com
        HostName ssh.github.com
        Port 443

    For further ssh issues please see the github `troubleshooting guide
    <https://docs.github.com/en/authentication/troubleshooting-ssh>`_.

The `pip install -U -e .[dev]` command has a cryptic error "zsh: no matches found: .[dev]"
    On zsh the square brackets need to be escaped this can either be achieved with
    backslashes `\[dev\]` or quotes "[dev]"

On MacOS the `envsubst` command cannot be found
    MacOS is not currently a supported OS (it is not part of our CI system)
    however we do have some MacOS users. If you want to use MacOS,
    installing `gettext` with homebrew should install `envsubst`::

        brew install gettext
