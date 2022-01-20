Frequently Asked Questions
==========================

My github ssh connection doesnt connect. What is wrong?
    Firstly make sure you have followed our :ref:`SSH key guide <ssh-keys>`.
    A simple fix could be to set a new variable in you `~/.ssh/config`::

        # below Host github.com
        HostName ssh.github.com
        Port 443

    For further ssh issues please see the github `troubleshooting guide
    <https://docs.github.com/en/authentication/troubleshooting-ssh>`_.
