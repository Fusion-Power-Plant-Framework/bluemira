Generating ParameterFrames
--------------------------

You can generate a py:class:ParameterFrame or set of py:class:ParameterFrames
from a reactor design.

The py:mod:gen_params module can be used on any python module to pull out the
py:class:ParameterFrames and create a global py:class:ParameterFrame or set of frames
and associated json files for those frames. It is used as follows:

.. code-bock::bash

    python -m bluemira.gen_params <module file location>

There is also a built in help for the function which explains further options:

.. code-bock::bash

    python -m bluemira.gen_params -h
