.. _ssh-keys:

Configuring SSH keys
--------------------

Firstly generating an ssh key (in a Linux terminal). In the directory `~/.ssh/` which you
can get to by running `cd ~/.ssh/`::

  ssh-keygen -t ed25519 -o -a 100 -f <name_of_key>

for example for a github key (the name is up to you)::

  ssh-keygen -t ed25519 -o -a 100 -f id_github

The command will ask you to enter a password, it is not strictly needed but personal
preference (just hit enter if you don't want one). Two files will have been created `id_github`
and `id_github.pub` you can see with the command `ls`.

Next uploading the key. For github go to this link https://github.com/settings/keys and
select add new ssh key and paste the contents of `id_github.pub` into the larger box. To
see the contents of the file you can use::

    more id_github.pub

Finally to configure to use the key for github locally we need to create a file in
`~/.ssh` called `config`. To do so you can use vim but nano may have a more familiar
interface::

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

If you have issues getting to this stage please refer to github's `ssh troubleshooting
guide <https://docs.github.com/en/authentication/troubleshooting-ssh>`_.
For other git websites (eg gitlab) it is exactly the same except you change the name of
your key everywhere and the Host in the config file. The location to upload the key will
be somewhere in your profile.
