Command Line Interface
======================

The BLUEPRINT Command Line Interface (CLI) can be used to initiate a run of BLUEPRINT
from the terminal. The run requires four input files, which it uses to build a
configurable reactor class object. Desired output files are then saved to a specified
directory. The command for the CLI is `blueprint` followed by the required inputs and any
options for your run.

::

    $ blueprint [INPUTS] [OPTIONS]

Inputs
------

The BLUEPRINT CLI accepts four JSON file inputs as positional arguments. A path can be
provided for each input or the following defaults can be used:

1. ``template.json``
2. ``config.json``
3. ``build_config.json``
4. ``build_tweaks.json``

Note that the `--indir` and `--reactornamein` options can be used to prepend a common
directory and reactor name prefix to all four inputs, meaning that in most cases the
positional arguments may be left as default.

The configurable reactor class imports parameters and values from ``template``, then
overides parameters specified in ``config`` with specified values. The class also
contains build parameters from ``build_config`` and ``build_tweaks`` to be used by the
BLUEPRINT run.

Outputs
-------

The BLUEPRINT CLI generates up to five outputs depending on the output mode. These are:

-  ``output.txt``: a text dump of the stdout for the run
-  ``errors.txt``: a text dump of any errors for the run
-  ``REACTORNAME_params.json``: a data file containing the optimise reactor parameters
-  ``REACTORNAME_XZ.png``: a 2D reactor plot in the XZ plane
-  ``REACTORNAME_XY.png``: a 2D reactor plot in the XY plane
-  ``REACTORNAME_CAD.stp``: a 3D reactor CAD model

Options
-------

The following options are available when running the BLUEPRINT CLI:

-  ``-i, --indir PATH``
-  ``-ri, --reactornamein TEXT``
-  ``-o, --outdir PATH``
-  ``-ro, --reactornameout TEXT``
-  ``-v, --verbose``
-  ``-t, --tarball``
-  ``-m, --outmode TEXT``
-  ``--output_log / --no_output_log``
-  ``--output_data / --no_output_data``
-  ``--output_xz / --no_output_xz``
-  ``--output_xy / --no_output_xy``
-  ``--output_cad / --no_output_cad``
-  ``--help``

-i, --indir PATH
        Prepends ``PATH`` to each input. Default = ``""``

        This option can be used to more easily specify a common directory for
        inputs. For example, if each input is stored in ``./input_directory``,
        the following commands are equivalent::

          $ blueprint ./input_directory/template.json  \
                      ./input_directory/config.json \
                      ./input_directory/build_config.json \
                      ./input_directory/build_tweaks.json

          $ blueprint -i ./input_directory \
                     template.json \
                     config.json \
                     build_config.json \
                     build_tweaks.json

        Which, since the default filenames are used, is also equivalent to::

          $ blueprint -i ./input_directory

        Note that if your inputs are stored
        in seperate directories, this option can still be used to enter the part
        of their path that they do have in common. e.g.::

          $ blueprint -i ~/code/BLUEPRINT/ \
                      path/to/template.json \
                      path/to/config.json \
                      path/to/build_config.json \
                      path/to/build_tweaks.json

-ri, --reactornamein TEXT
        Specifies a reactor name used as the filename prefix to each input.

        This can be used as a similar quality of life feature to ``--indir`` if each input uses
        the same reactorname such that
        ``$ blueprint -i ./input_directory -ri reactor_name``
        is equivalent to
        ::

            $ blueprint input_directory/reactor_name_template.json \
                        input_directory/reactor_name_config.json \
                        input_directory/reactor_name_build_config.json \
                        input_directory/reactor_name_build_tweaks.json

-ro, --reactornameout TEXT
        Specifies an overide name to use

-o, --outdir PATH
        Specifies ``PATH`` as the output directory. Default = ``"f"{get_BP_root()}/generated_data/reactors"``

-v, --verbose
        Flag to switch verbose mode on.

        When on, data output to ``REACTORNAME_params.json`` will also contain metadata.

-t, --tarball
        Flag to enable creation of a tarball of the output directory.

        This can be used to simplify storing output files locally after running BLUEPRINT remotely.

-m, --outmode TEXT
        Sets the output mode, specifying which outputs to generate.

        Options are ``default``, ``full``, ``lite``, ``cad``, and ``none``.

        The following table shows which outputs are provided in each output
        mode.::

            default   : log, data, plot_xz, plot_xy, ___
            full      : log, data, plot_xz, plot_xy, cad
            lite      : log, data, _______, _______, ___
            cad       : log, ____, _______, _______, cad
            none      : ___, ____, _______, _______, ___

        If your desired outputs are not covered by any of the above, you can also use the
        following switches to override the ``--outmode`` setting for a given output:

        --output_log, --no_output_log
                Enables / disables output of the ``stdout`` and ``stderr`` text dumps for the run, saved as ``output.txt`` and ``errors.txt``.
        --output_data, --no_output_data
                Enables / disables output of the optimised reactor parameters data file, saved as ``REACTORNAME_params.json``.
        --output_xz, --no_output_xz
                Enables / disables output of the 2D reactor plot in the XZ-plane, saved as ``REACTORNAME_XZ.json``.

        --output_xy, --no_output_xy
                Enables / disables output of the 2D reactor plot in the XY-plane, saved as ``REACTORNAME_XY.json``.

        --output_cad, --no_output_cad
                Enables / disables output of the 3D reactor CAD model, saved as ``REACTORNAME_CAD.json``.

Example Usage
-------------

See examples/cli/README.md example usage.