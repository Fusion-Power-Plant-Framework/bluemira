Generating ParameterFrames and Parameter Files
----------------------------------------------

You can generate a py:class:ParameterFrame or set of :py:class:`ParameterFrames`
from a reactor design.

The :py:mod:`bluemira.gen_params` module can be used on any python module to pull out the
:py:class:`ParameterFrames` and create a global :py:class:`ParameterFrame` or set of frames
and associated json files for those frames. It is used as follows:

.. code-block:: shell

    python -m bluemira.gen_params <module file location>

There is also a built in help for the function which explains further options:

.. code-block:: shell

    python -m bluemira.gen_params -h
