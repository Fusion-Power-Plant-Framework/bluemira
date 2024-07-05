codes
=====

The :py:mod:`.codes` module defines and contains interfaces to
`(pluggable) dependencies <#pluggable-dependencies>`_
and `external codes <#external-code-interfaces>`_.

Pluggable Dependencies
----------------------

Each pluggable dependency should have an API that
isolates bluemira users from any changes to that dependency.
The main APIs that we have created so far are for the CAD engine
and the optimiser library.

For the CAD engine all the functions are wrapped by equivalent functions
in the geometry module.
The geometry module should be the interface most users access CAD through.
The CAD engine is still accessible for advanced users with the hope that
any new functionality that is created can be ported to the geometry module.
If there is a future CAD engine change we aim to make this invisible to the
end user.

The optimiser library is wrapped in a similar way
but as optimisers are used in various areas of bluemira
the main user interface is through the optimiser module in utilities.

Any additional pluggable dependencies should have APIs created in a similar way.

External Code Interfaces
------------------------

External codes are defined as programs that are run outside of `bluemira`.
Most often this will be a program that is run on the command line
whose outputs can be parsed and incorporated into `bluemira`.

Codes Interface
^^^^^^^^^^^^^^^

A generic interface for all programs that are run externally is defined in
the :py:mod:`.codes.interface` module.

Solver
""""""

The key class defining the interface to an external program is
:py:class:`.CodesSolver`.

A :py:class:`.CodesSolver` takes, as input, a :py:class:`.MappedParameterFrame`,
defining the input/output parameters of the solver,
and a ``build_config``, defining the solver's run options.
When executed, the solver sequentially calls `tasks <#tasks>`_ that
set up and run the external program, using a given '`run mode <#run-mode>`_'
and a shared '`MappedParameterFrame <#mappedparameterframes>`_'.

Below is an example of how one would use a :py:class:`.CodesSolver`.

.. code-block::python  # doctest: +SKIP

    import bluemira.codes.my_code as my_code

    params: ParameterFrame
    build_config: Dict

    solver = my_code.Solver(params, build_config)
    solver.execute("run")

Tasks
"""""

A task can be thought of as a stage of a solver,
a :py:class:`.CodesSolver` requires implementations of classes for three tasks:

1. Setup

    This class will generally be responsible for taking parameters from
    ``bluemira`` and generating an input to the external code.
    This input will most often be a file.

2. Run

    This class is responsible for running the external code.
    It will often execute a shell command to do so.
    The :py:meth:`._run_subprocess` method on :py:class:`CodesTask`
    is provided to run shell commands and incorporate standard output with
    ``bluemira``'s logging system.

3. Teardown

    This class is responsible for parsing the external program's outputs,
    an incorporating them into ``bluemira``.
    This incorporating usually involves updating the solver's parameter
    values with the outputs of the program.

Run Mode
""""""""

Within each :py:class:`~bluemira.codes.interface.CodesTask`,
there is the concept of a :py:class:`~bluemira.codes.interface.RunMode`.
A :py:class:`~bluemira.codes.interface.RunMode` enum enumerates the ways in which
an external program can be run.
It must, at least, contain a ``RUN`` option,
but will often also have ``READ`` and ``MOCK`` options.
This way, if an external code is not installable (e.g., for licensing reasons),
the solver can instead be called in ``MOCK`` mode
to output some pre-defined result,
or in ``READ`` mode to parse results from a previous run of the program.

MappedParameterFrames
^^^^^^^^^^^^^^^^^^^^^

:py:class:`.MappedParameterFrame` extends :py:class:`.ParameterFrame`
to allow mapping to external codes' variables through `ParameterMapping`_.
Default values for external codes' parameters are provided for instances
where there are many unmapped variables,
which are usually only known by users experienced with the external code.
These unmapped parameters can be set using the :py:attr:`problem_settings` of a
:py:class:`.CodesSolver` instance.

ParameterMapping
""""""""""""""""

:py:class:`.ParameterMapping` is used to create a connection
between ``bluemira`` parameters and parameters of any external program.
At its most basic level, it is a key-value mapping between two variable names.
On top of the mapping, how the parameter value flows
between ``bluemira`` and the external program
is modified by the :py:attr:`send` and :py:attr:`recv` attributes.

:py:attr:`send`
    ``True`` - set bluemira parameter value as input to external code

    ``False`` - use default value as input to external code

:py:attr:`recv`
    ``True`` - set external code result to the new value of the bluemira parameter

    ``False`` - keep the original bluemira parameter value ignoring the external value


Example
"""""""

The simplest interface definition would look something like the below:

.. code-block:: python

    from enum import auto

    from bluemira.codes.interface import BaseRunMode
    from bluemira.codes.interface import CodesSolver, CodesTeardown, CodesSetup, CodesTask


    class RunMode(BaseRunMode):
        RUN = auto()


    class Setup(CodesSetup):

        def run(self):
            # Write input file using input parameter values
            pass


    class Run(CodesTask):

        def run(self, inputs):
            # eg self._run_subprocess(self.binary)
            pass

    class Teardown(CodesTeardown):

        def run(self, inputs):
            # read results from the output file
            pass


    class Solver(CodesSolver):
        name = "external_program"
        setup_cls = Setup
        run_cls = Run
        teardown_cls = Teardown
        run_mode_cls = RunMode

        def __init__(
            self,
            params,
            build_config,
        ):
            super().__init__(params)

            self.build_config = build_config
            self.binary = build_config.get("binary", None),
            # problem settings are parameters passed directly to the external program
            self.problem_settings = build_config.get("problem_settings", None)

    Solver(params=None, build_config={}).execute('run')

APIs
^^^^

An interface for programs that have an API to a Python library
should follow the same pattern as above.
For now, we do not have an example integration.
The first expected integration will be the PROCESS,
once its python interface has been completed.

If you have an existing code that you would like to integrate into ``bluemira``,
please contact the maintainers so we can discuss the best way forward.
