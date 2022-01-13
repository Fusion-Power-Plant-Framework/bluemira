Command Line Interface
======================

The BLUEPRINT Command Line Interface (CLI) can be used to initiate a run of BLUEPRINT
from the terminal.
The run requires four input files, which it uses to build a configurable reactor class
object. Desired output files are then saved to a specified directory.
The command for the CLI is ``blueprint`` followed by the required inputs and any options
for your run.

::

    $ blueprint [INPUTS] [OPTIONS]

Inputs
------

The BLUEPRINT CLI accepts four JSON file inputs as positional arguments.
A path can be provided for each input or the following defaults can be used:

1. ``template.json``
2. ``config.json``
3. ``build_config.json``
4. ``build_tweaks.json``

.. note::

    The intended way to use the BLUEPRINT CLI is to leave these positional arguments
    blank, instead using  the `--indir` and `--reactornamein` options to specify the
    directory and reactor name used by input files.

The `Configurable_Reactor` class imports parameters and values from `template`, then
overrides parameters specified in `config` with specified values. The class also takes
build parameters from `build_config` and `build_tweaks` to be used by the BLUEPRINT run.

Outputs
-------

The BLUEPRINT CLI generates up to six outputs depending on the output mode. These are:

-  ``REACTORNAME_output.txt``: a text dump of the stdout for the run
-  ``REACTORNAME_errors.txt``: a text dump of any errors for the run
-  ``REACTORNAME_params.json``: a data file containing the optimise reactor parameters
-  ``REACTORNAME_XZ.png``: a 2D reactor plot in the XZ plane
-  ``REACTORNAME_XY.png``: a 2D reactor plot in the XY plane
-  ``REACTORNAME_CAD_MODEL.stp``: a 3D reactor CAD model

Options
-------

The following options are available when running the BLUEPRINT CLI:

-  ``-i, --indir PATH``
-  ``-ri, --reactornamein TEXT``
-  ``-o, --outdir PATH``
-  ``-ro, --reactornameout TEXT``
-  ``-v, --verbose``
-  ``-f, --force_rerun``
-  ``-t, --tarball``
-  ``--log / --no_log``
-  ``--data / --no_data``
-  ``--plots / --no_plots``
-  ``--cad / --no_cad``
-  ``--help``

-i, --indir PATH
        Prepends ``PATH`` to each input. Default = ``""``

        This option can be used to more easily specify a common directory for inputs.
        For example, if each input is stored in ``input_directory``,
        the following commands are equivalent:

        ::

                $ blueprint \
                input_directory/template.json \
                input_directory/config.json \
                input_directory/build_config.json \
                input_directory/build_tweaks.json

        ::

                $ blueprint -i ./input_directory \
                        template.json \
                        config.json \
                        build_config.json \
                        build_tweaks.json

        .. note::
                If your inputs are stored in separate directories, this option can still
                be used to enter the common components of their paths. For example:

                ::

                        $ blueprint -i ~/code/BLUEPRINT/ \
                                path/to/template.json \
                                path/to/config.json \
                                path/to/build_config.json \
                                path/to/build_tweaks.json

-ri, --reactornamein TEXT
        Specifies a reactor name used as the filename prefix to each input.

        This can be used as a similar quality of life feature to ``--indir`` if each
        input uses the same reactorname such that
        ::

                $ blueprint -i ./input_directory -ri reactor_name

        is equivalent to
        ::

            $ blueprint input_directory/reactor_name_template.json \
                        input_directory/reactor_name_config.json \
                        input_directory/reactor_name_build_config.json \
                        input_directory/reactor_name_build_tweaks.json

-o, --outdir PATH
        Specifies ``PATH`` as the generated data root for outputs. Default = ``None``

        .. note::

                The keyword ``!BM_ROOT!`` may be used, to be replaced with the path to
                the local BLUEPRINT directory (e.g. ``~/code/BLUEPRINT``).

-ro, --reactornameout TEXT
        Specifies the reactor name to use for the run and makes a copy of reference data
        in a new subdirectory using this reactor name. Also sets the name of the output
        subdirectory and the filename prefix for each output file.

        .. note::

                BLUEPRINT build will not run if a subdirectory already exists using the
                new reactor name in either reference or generated data root directories.
                This can be bypassed using the ``--force_rerun`` flag if required.

-f, --force_rerun
        Force rerun flag. When on, existing data directories will be ignored and any data
        overwritten. This applies to both the output data directory as well as the copy
        of reference data made when --reactornameout is used. By default, BLUEPRINT CLI
        will raise an error before writing any files if the relevant directory exists.

-v, --verbose
        Flag to switch verbose mode on.

        When on, data output to ``REACTORNAME_params.json`` will also contain metadata.

-t, --tarball
        Flag to enable creation of a tarball of the output directory.

        This can be used to simplify storing output files locally after running BLUEPRINT
        remotely.

        following switches to override the ``--outmode`` setting for a given output:

.. note::
        By default, the CLI produces each output type except the CAD model, which
        increases the BLUEPRINT runtime and the output storage use. The following
        switches can be used to turn each output type on or off as desired.

--log, --no_log
        Enables / disables output of the ``stdout`` and ``stderr`` text dumps for the run, saved as ``REACTORNAME_output.txt`` and ``REACTORNAME_errors.txt``.
        Default: On

--data, --no_data
        Enables / disables output of the optimised reactor parameters data file, saved as ``REACTORNAME_params.json``.
        Default: On

--plots, --no_plots
        Enables / disables output of the 2D reactor images in the xz and xy planes, saved as ``REACTORNAME_XZ.png`` and ``REACTORNAME_XY.png``.
        Default: On

--cad, --no_cad
        Enables / disables output of the 3D reactor CAD model, saved as ``REACTORNAME_CAD_MODEL.stp``.
        Default: Off

Example Usage
-------------

See ``examples/cli/README.md`` for example usage.
