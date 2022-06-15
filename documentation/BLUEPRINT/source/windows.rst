Installation on Windows
=======================

BLUEPRINT is supported on Ubuntu 18.04, so if you are running on Windows you will need
to be able to access an Ubuntu environment.

Setting up WSL
--------------

To run an Ubunutu environment in Windows, we recommend using
`Windows Subsystem for Linux <https://docs.microsoft.com/en-us/windows/wsl/>`_ (WSL).

WSL can be enabled and installed by following the
`instructions provided by Microsoft <https://docs.microsoft.com/en-us/windows/wsl/install-win10>`_.

If you already have WSL installed on your computer then skip to
`installing Ubuntu 18.04 in WSL`_.

.. note::

   If you are working on an organisational computer then you may need to request your
   administrator to enable and perform the WSL installation.

When you have rebooted your computer following the WSL installation, open a new command
prompt (hold the Windows key and press R on your keyboard then type cmd in the dialog box
and hit enter). In the command prompt type `wsl -l` - if you see a message as below then
WSL is enabled on your environment.

    Windows Subsystem for Linux has no installed distributions.
    Distributions can be installed by visiting the Microsoft Store:
    https://aka.ms/wslstore

You'll now need to get an `Ubuntu 18.04 distribution installed <#installing-ubuntu-18-04-in-wsl>`_
in your WSL environment.

Installing Ubuntu 18.04 in WSL
------------------------------

Once you have WSL enabled you can download and install Ubuntu 18.04 to run in that
environment.

In your Windows environment, open a web browser (e.g. Chrome or Firefox) and navigate to
the `Ubuntu 18.04 distribution in Microsoft Store <https://www.microsoft.com/en-gb/p/ubuntu-1804-lts/9n9tngvndl3q>`_.

You should see a Windows Store page with header Ubuntu 18.04 LTS. We will be using Ubuntu
18.04 as the BLUEPRINT setup scripts are currently supported on that version of Ubuntu.

Click the blue Get button on the right hand side of the page. This will likely produce a
pop up in your browser asking if you'd like to "Open Microsoft Store?". You will need to
click the "Open Microsoft Store" button to proceed.

This should open a new Microsoft Store window showing the Ubuntu 18.04 LTS application.
Click Get in that new window and you should see that the application begins to download.
Once the download has completed, the Get button will have changed to a Launch button.
Click the Launch button from within the Windows Store window.

You will now see a command prompt that says "Installing, this may take a few minutes...".
When the installation has completed, the prompt will then say "Installation successful!"
and ask you to create a new UNIX user account. You will need to enter a user name and
password (the password will need to be entered twice), which do not need to match your
Windows credentials.

You will need to remember this user name and password to be able to access your Ubuntu
environment, and you will also need to remember the password in order to run commands
that require elevated privileges as "root".

When you have created your new credentials you will be placed in your Ubuntu environment
this will be a command prompt with an active command line like
[user name]@[machine name]:[current working directory]. You should then update your
Ubuntu distribution to make sure you have all the latest versions of the default
packages:

.. code-block:: bash

    sudo apt-get update
    sudo apt-get upgrade

Installing and configuring VcXsrv
---------------------------------

As WSL runs in command line-only mode by default we will need to set up a mechanism for
images to be displayed on our Windows environment. We recommend using
`VcXsrv <https://sourceforge.net/projects/vcxsrv/>`_ to do this.

.. note::

   If you are working on an organisational computer then you may need to request your
   administrator to perform the VcXsrv installation.

When you have completed the VcXsrv installation, open XLaunch from the Windows start
menu. Select your display settings (recommend multiple windows) and leave the display
number as -1 and click Next.

Select "Start no client" and click Next.

Ensure that the checkboxes are selected for "Clipboard" and "Primary Selection" and
deselected for "Native opengl" (note this is selected by default) and selected for
"Disable access control" and click Next. If convenient, save your configuration so that
it can be reused - if you save the configuration then you can start XLaunch in the future
by double clicking the saved configuration file.

Click finish.

In your WSL session, open your ~/.bashrc file e.g. with `nano ~/.bashrc` and add the
following to the end of the file:

.. code-block:: bash

    export DISPLAY=$(awk '/nameserver / {print $2; exit}' /etc/resolv.conf 2>/dev/null):0
    export LIBGL_ALWAYS_INDIRECT=1

This will allow WSL to connect the display to your Windows X-server.

Close and re-open your WSL session at this point so that the change to your ~/.bashrc
takes effect.

You can now continue with the `BLUEPRINT installation <started.html>`_ in your WSL
environment.
