codes
=====

The codes subpackage has all interfaces to codes that are either made to be easily switched or replaced (APIs) as well as any external program that runs using input and output files (a File-Program interface, FPI). There are two types of program interfaces in this submodule, the first is for pluggable dependenices and the second is for external codes (eg. PROCESS or PLASMOD).

Pluggable Dependencies
----------------------

Each pluggable dependency should have an API that isolates bluemira users from any changes to that dependency. The main APIs that we have created so far are for the CAD engine and the optimiser library.

For the CAD engine all the functions are wrapped by equivalent functions in the geometry module. The geometry module should be the interface most users access CAD through. The CAD engine is still accessible for advanced users with the hope that any new functionality that is created can be ported to the geometry module. If there is a future CAD engine change we aim to make this invisible to the end user.

The optimiser library is wrapped in a similar way but as optimisers are used in various areas of bluemira the main user interface is through the optimiser module in utilities.

Any additional pluggable dependencies should have APIs created in a similar way.

External Code Interfaces
------------------------

FPIs
^^^^

A generic interface for all programs that use files for I/O is in the ``codes.interface`` module.
There are 5 main classes that need to be inherited from to create a file interface for a program. These are ``FileProgramInterface`` and ``RunMode`` and the Tasks: ``Setup``, ``Run`` and ``Teardown``.

The simplest interface would look like:

.. code-block:: python

    from enum import auto

    import bluemira.codes.interface as interface


    class RunMode(interface.RunMode):
        RUN = auto()


    class Setup(interface.Setup):

        def _run(self):
            # Write input file
            pass


    class Run(interface.Run):

        def _run(self):
            self._run_subprocess(self._binary)


    class Teardown(interface.Teardown):

        def _run(self):
            # read from the output file
            pass


    class Solver(interface.FileProgramInterface):
        _runmode = RunMode

        def __init__(
            self,
            params,
            build_config=None,
            run_dir: Optional[str] = None,
        ):
            super().__init__(
                "MYPROG",
                params,
                build_config.get("mode", "run"),
                binary=build_config.get("binary", None),
                run_dir=run_dir,
                mappings=[],
                problem_settings=build_config.get("problem_settings", None),
            )


FileProgramInterface
""""""""""""""""""""

The ``FileProgramInterface`` class collects all the tasks together providing a single point to interface between bluemira and the external program.
A child of FileProgramInterface is the only class that needs to be imported to run a specific solver as seen below.

.. code-block:: python

    import bluemira.codes.mycode as mycode

    params: ParameterFrame
    build_config: Dict

    solver = mycode.Solver(params, build_config)
    solver.run()

All mappings for a code are passed to the ``ParameterFrame`` from the child class. The ``RunMode``, ``Setup``, ``Run`` and ``Teardown`` classes are forced to inherit from their respective baseclasses, and a few properties for ease of access are defined. The runmode and the directory in which the code is run are set in the class initialisation.

Variables can be passed through the run method to each task. Using keyword arguments is recommended so that the variables are used correctly.

The only class that technically needs to be defined is ``RunMode``. The rest can be set to None (although they will produce a warning) or just not defined in the inherited class attributes.

RunMode
"""""""

Each run mode of the code should be defined as a class attribute inherited from this class. The name of the run mode corresponds to the task method that is called when the solver is run. The method that is called is prefixed with '_' and the lower case version of the run mode eg. ``RunMode.RUN`` would call ``_run``. Tasks do not need to have any run methods. The methods will only be called if they exists.

Tasks
"""""

The basic task that the three task types inherit from (``Setup``, ``Run``, ``Teardown``)
The ``_run_subprocess`` method is defined here as some tasks other than ``Run`` may want to run an external program. All stdout/err outputs of any external code are captured here so we can control what is output to the screen. ``stdout`` is sent to the INFO logger and ``stderr`` is sent to the ERROR logger.
The parent attribute of tasks is an instance of a ``FileProgramInterface`` child class which allows communication between tasks.

All base tasks have a ``__init__`` method therefore any child task need to call ``super().__init__(**kwargs)`` to ensure the task is initialised completely.
The tasks are defined as follows:

Setup
    The ``Setup`` task is designed to create and write any input files from a ParameterFrame and any extra problem_settings.

Run
    The ``Run`` task is usually smallest task. Essentially should only call the program as seen above. The binary name is stored here.

Teardown
    The ``Teardown`` task deals with reading back in and processing the output data. By default it does very little as this is usuall bespoke.


Pattern for external codes
""""""""""""""""""""""""""

Each external code should contain:
 - A default input file either in json form or directly in the input file format
 - A constants file where the default binary name and the program name is defined
 - A mappings file where the mappings between bluemira variable names and the external variable names are defined.

APIs
^^^^

An interface for programs that have an API to python should follow the same pattern as FPIs. For now we do not have an example integration. The first possible integration will be the PROCESS integration as its python interface is currently being fleshed out.

If you have an existing code that you would like to integrate into bluemira through this method please contact the maintainers so we can discuss the best way forward.
