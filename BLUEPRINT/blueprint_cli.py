# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
BLUEPRINT Command Line Interface
"""

import click
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
import os
from pathlib import Path
import shutil
import sys
import tarfile
from typing import Optional

from BLUEPRINT.base.file import KEYWORD
from bluemira.base.file import get_bluemira_root
from bluemira.base.logs import set_log_level
from bluemira.utilities.tools import get_module, CommentJSONDecoder

try:
    from functools import cached_property
except ImportError:
    from cached_property import cached_property


@dataclass
class InputManager:
    """
    A class to manage the inputs to the CLI.

    Parameters
    ----------
    template: str
        Name of the file containing the template parameter configuration.
    config: str
        Name of the file containing the specific parameter configuration for the run.
    build_config: str
        Name of the file containing the build configuration.
    build_tweaks: str
        Name of the file containing the tweaking parameters configuration.
    indir: str
        Path to the directory containing the input configuration files.
    reactornamein: str
        The name of the input reactor name providing any template reference data.
    outdir: str
        Path to the output directory.
    reactornameout: str
        The name of the output reactor for the run.
    """

    template: str
    config: str
    build_config: str
    build_tweaks: str
    indir: str
    reactornamein: str
    datadir: str
    outdir: str
    reactornameout: str

    def _build_path_in(self, file_name: str) -> str:
        if self.reactornamein is not None:
            file_name = f"{self.reactornamein}_{file_name}"
        return os.path.join(self.indir, file_name)

    @property
    def template_config_path_in(self) -> str:
        """
        The input template configuration file path.
        """
        return self._build_path_in(self.template)

    @property
    def config_path_in(self) -> str:
        """
        The input run configuration file path.
        """
        return self._build_path_in(self.config)

    @property
    def build_config_path_in(self) -> str:
        """
        The input build configuration file path.
        """
        return self._build_path_in(self.build_config)

    @property
    def build_tweaks_path_in(self) -> str:
        """
        The input tweaking parameters configuration file path.
        """
        return self._build_path_in(self.build_tweaks)

    @property
    def reactor_name(self) -> str:
        """
        The derived name of the reactor for this run.

        Notes
        -----
        This will provide the reactornameout, if provided, else the reactornamein if
        provided, else it will read the reactor name from the specified run configuration
        file.
        """
        if self.reactornameout is not None:
            return self.reactornameout
        elif self.reactornamein is not None:
            return self.reactornamein
        elif isinstance(self.config_dict["Name"], dict):
            return self.config_dict["Name"]["value"]
        else:
            return self.config_dict["Name"]

    @cached_property
    def build_config_dict(self) -> str:
        """
        The dictionary representation of the build configuration.
        """
        return self.read_json(self.build_config_path_in)

    @cached_property
    def config_dict(self) -> str:
        """
        The dictionary representation of the run configuration.
        """
        return self.read_json(self.config_path_in)

    @cached_property
    def output_root_path(self) -> str:
        """
        The root output path, excluding the reactor subdirectory for the run.
        """
        return self._try_get_path_from_config(
            "generated_data_root", "generated_data/BLUEPRINT", dir=self.outdir
        )

    @cached_property
    def output_path(self) -> str:
        """
        The full output path, including the reactor and reactor_name subdirectories for
        the run.
        """
        return os.path.join(self.output_root_path, "reactors", self.reactor_name)

    @cached_property
    def reference_root_path(self) -> str:
        """
        The root reference data path, excluding the reactor subdirectory for the run
        """
        return self._try_get_path_from_config(
            "reference_data_root", "data/BLUEPRINT", dir=self.datadir
        )

    @cached_property
    def reference_path(self) -> str:
        """
        The full reference data path, including the reactor and reactor_name
        subdirectories for the run
        """
        return os.path.join(self.reference_root_path, "reactors", self.reactor_name)

    def _try_get_path_from_config(
        self, key: str, default_value: str, dir: Optional[str] = None
    ) -> str:
        path = dir
        if path is None:
            if key in self.build_config_dict:
                path = self.build_config_dict[key]
            else:
                path = os.path.join(get_bluemira_root(), default_value)
                click.echo(
                    "Warning: outdir not specified in command line and no "
                    f"{key} found in {self.build_config}. Reverting to "
                    f"default path {path}."
                )

        if KEYWORD in path:
            path = path.replace(KEYWORD, get_bluemira_root())

        return path

    def read_json(self, file):
        """
        Reads a json file and returns a dict object of its contents.
        """
        with open(file, "r") as fh:
            file_dict = json.load(fh, cls=CommentJSONDecoder)
        return file_dict


class OutputManager:
    """
    A class to manage the output paths for the run.

    Parameters
    ----------
    template: str
        Path of the output file containing the template parameter configuration.
    config: str
        Path of the output file containing the specific parameter configuration for the
        run.
    build_config: str
        Path of the output file containing the build configuration.
    build_tweaks: str
        Path of the output file containing the tweaking parameters configuration.
    output: str
        Path of the output file containing the stdout text dump.
    errors: str
        Path of the output file containing the stderr text dump.
    params: str
        Path of the output file containing the optimised parameter configuration produced
        by the run.
    plot_xz: str
        Path of the output file containing the 2D plot in the xz plane.
    plot_xy: str
        Path of the output file containing the 2D plot in the xy plane.
    cad: str
        Path of the output file containing the 3D CAD model.
    tar: str
        Path of the compressed tarball of the output directory.
    """

    template: str
    config: str
    build_config: str
    build_tweaks: str
    output: str
    errors: str
    params: str
    plot_xz: str
    plot_xy: str
    cad: str
    tar: str

    def _make_path(self, inputs: InputManager, key: str, ext: str, subdir=""):
        path = os.path.join(
            inputs.output_path, subdir, f"{inputs.reactor_name}_{key}.{ext}"
        )
        if "_." in path:
            path = path.replace("_.", ".")
        return path

    def __init__(self, inputs: InputManager):
        self.template = self._make_path(inputs, "template", "json")
        self.config = self._make_path(inputs, "config", "json")
        self.build_config = self._make_path(inputs, "build_config", "json")
        self.build_tweaks = self._make_path(inputs, "build_tweaks", "json")
        self.output = self._make_path(inputs, "output", "txt")
        self.errors = self._make_path(inputs, "errors", "txt")
        self.params = self._make_path(inputs, "params", "json")
        self.plot_xz = self._make_path(inputs, "XZ", "png", subdir="plots")
        self.plot_xy = self._make_path(inputs, "XY", "png", subdir="plots")
        self.cad = self._make_path(inputs, "CAD_MODEL", "stp", subdir="CAD")
        self.tar = self._make_path(inputs, "", "tar.gz")

    def copy_files(self, inputs: InputManager):
        """
        Copy the input paths to the output paths

        Parameters
        ----------
        inputs: InputManager
            The Input instance containing the template, config, build_config, and
            build_tweaks paths to copy the files from
        """
        variables = {
            "template_config_path": self.template,
            "config_path": self.config,
            "build_config_path": self.build_config,
            "build_tweaks_path": self.build_tweaks,
        }
        self.reactor_kwargs = {}
        for name, var in variables.items():
            try:
                shutil.copy(getattr(inputs, f"{name}_in"), var)
                self.reactor_kwargs[name] = var
            except FileNotFoundError:
                click.echo(f"No {name} file")


def dump_json(dict_object: dict, output_path: str):
    """
    Saves a dict object as a json file.
    """
    with open(output_path, "w") as fh:
        json.dump(dict_object, fh)


def _check_path(name, path: str, force: bool = False, make: bool = True):
    if os.path.exists(path):
        if force:
            click.echo(
                f"Warning: Force rerun flag detected. Overwriting {name} directory "
                f"{path} for this reactor."
            )
            shutil.rmtree(path)
        else:
            raise FileExistsError(
                f"{name.capitalize()} directory {path} already exists for this reactor. "
                "Select a new outdir, change the reactor name, or remove the existing "
                "directory and run again."
            )
    if make:
        Path(path).mkdir(parents=True)


def get_reactor_class(reactor_string):
    """
    Dynamically import reactor class

    Parameters
    ----------
    reactor_string: str
        string to import reactor from

    Returns
    -------
    reactor class

    Notes
    -----
    reactor string examples:

    "Reactor" - default reactor from BLUEPRINT
    "path/to/file.py::Reactor" - import reactor from file
    "path.to.module.Reactor" - import reactor from known module

    """
    if "::" in reactor_string:
        module_string, reactor = reactor_string.split("::")
    elif "." in reactor_string:
        module_string, reactor = reactor_string.rsplit(".", 1)
    else:
        module_string = "BLUEPRINT.reactor"
        reactor = reactor_string

    try:
        return getattr(get_module(module_string), reactor)
    except AttributeError:
        raise ImportError(f"Class '{reactor_string}' not found")


@click.command()
@click.argument("template", type=click.Path(), default="template.json", required=False)
@click.argument("config", type=click.Path(), default="config.json", required=False)
@click.argument(
    "build_config", type=click.Path(), default="build_config.json", required=False
)
@click.argument(
    "build_tweaks", type=click.Path(), default="build_tweaks.json", required=False
)
@click.option(
    "-i",
    "--indir",
    type=click.Path(),
    default="",
    help="Specifies a directory to preprend to each input path.",
)
@click.option(
    "-ri",
    "--reactornamein",
    default=None,
    help="Specifies a reactor name used as a prefix to each input filename.",
)
@click.option(
    "-d",
    "--datadir",
    type=click.Path(writable=True),
    default=None,
    help="Specifies the directory in which any input reference data is stored. Note \
    that these inputs must be stored in a subdirectory within the directory provided, \
    corresponding to the specified reactor name at reactors/{reactor name}.",
)
@click.option(
    "-o",
    "--outdir",
    type=click.Path(writable=True),
    default=None,
    help="Specifies the directory in which to store output files. Note that outputs \
    will be stored in a subdirectory within the directory provided, corresponding to \
    the specified reactor name. The directory must not exist before running BLUEPRINT \
    to avoid unintentional overwrites.",
)
@click.option(
    "-ro",
    "--reactornameout",
    default=None,
    help="Specifies a reactor name, overriding the value in the input config file. \
    Also used as the output subdirectory name and as a prefix to each output file.",
)
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase logging severity level.",
)
@click.option(
    "-q",
    "--quiet",
    count=True,
    help="Decrease logging severity level.",
)
@click.option(
    "-f",
    "--force_rerun",
    is_flag=True,
    help="Forces a rerun of BLUEPRINT when existing data is detected. When on, existing \
    data will be overwritten.",
)
@click.option(
    "-t",
    "--tarball",
    is_flag=True,
    help="Enables creation of a tarball of the output directory.",
)
@click.option(
    "--log/--no_log",
    default=True,
    help="Enables/disables output of BLUEPRINT text dump.",
)
@click.option(
    "--data/--no_data",
    default=True,
    help="Enables/disables output of the .json data file.",
)
@click.option(
    "--plots/--no_plots",
    default=True,
    help="Enables/disables output of the 2D reactor images in the xz and xy planes.",
)
@click.option(
    "--cad/--no_cad",
    default=False,
    help="Enables/disables output of the 3D cad model.",
)
@click.option(
    "-r",
    "--reactor_class",
    default="ConfigurableReactor",
    help="specify reactor class (file or package path)",
)
@click.version_option(package_name="bluemira", prog_name="bluemira")
@click.pass_context
def cli(
    ctx,
    template,
    config,
    build_config,
    build_tweaks,
    indir,
    reactornamein,
    datadir,
    outdir,
    reactornameout,
    verbose,
    quiet,
    force_rerun,
    tarball,
    log,
    data,
    plots,
    cad,
    reactor_class,
):
    """
    Run BLUEPRINT build for a configurable reactor.
    Accepts four .json inputs as optional positional arguments:

    1.  template  [default = template.json]
            config file providing template reactor parameters and values.
    2.  config  [default = config.json]
            config file to override specified reactor parameters with specified values.
    3.  build_config  [default = build_config.json]
            file containing build parameters used by BLUEPRINT.
    4.  build_tweaks  [default = build_tweaks.json]
            file containing additional build parameters.
    """
    set_log_level(min(max(0, 2 + quiet - verbose), 5))

    inputs = InputManager(
        template=template,
        config=config,
        build_config=build_config,
        build_tweaks=build_tweaks,
        indir=indir,
        reactornamein=reactornamein,
        datadir=datadir,
        outdir=outdir,
        reactornameout=reactornameout,
    )

    outputs = OutputManager(inputs)
    click.echo(f"Output directory set to {inputs.output_path}")
    _check_path("output", inputs.output_path, force=force_rerun)

    # Copy inputs to output directory.
    click.echo(f"Copying inputs from {inputs.indir} to {inputs.output_path}")
    outputs.copy_files(inputs)

    # Update generated_data_root to value given in CLI options.
    inputs.build_config_dict["generated_data_root"] = str(inputs.output_root_path)

    # Update generated_data_root to value given in CLI options.
    inputs.build_config_dict["reference_data_root"] = str(inputs.reference_root_path)

    dump_json(inputs.build_config_dict, outputs.build_config)

    # Update reactor name and make a copy of reference data to a subdirectory using the
    # new reactor name.
    if inputs.reactornameout is not None:
        old_reactorname = inputs.config_dict["Name"]
        if isinstance(old_reactorname, dict):
            old_reactorname = old_reactorname["value"]
            inputs.config_dict["Name"]["value"] = inputs.reactor_name
            if inputs.config_dict["Name"]["source"] is None:
                inputs.config_dict["Name"]["source"] = "Input"
        else:
            inputs.config_dict["Name"] = inputs.reactor_name

        _check_path("input", inputs.reference_path, force=force_rerun, make=False)
        reference_source = os.path.join(
            inputs.reference_root_path, "reactors", old_reactorname
        )
        shutil.copytree(reference_source, inputs.reference_path)
        dump_json(inputs.config_dict, outputs.config)

    # Instantiate BLUEPRINT reactor class.
    reactor = get_reactor_class(reactor_class).from_json(**outputs.reactor_kwargs)

    # Return output log.
    if log:
        click.echo(
            f"Redirecting stdout to {outputs.output} and stderr to {outputs.errors}."
        )
        click.echo(
            f"In a new terminal window, use the following command to view outputs\
         while BLUEPRINT runs: tail -f {outputs.output}"
        )

        sys.stdout = open(outputs.output, "w")
        click.echo(f"Saving output log as {outputs.output}")
        sys.stderr = open(outputs.errors, "w")
        click.echo(f"Saving errors log as {outputs.errors}")

    # Run BLUEPRINT build.
    click.echo("Running BLUEPRINT build.")
    reactor.build()
    click.echo("BLUEPRINT build complete.")

    if (
        isinstance(ctx.obj, dict)
        and "standalone_mode" in ctx.obj
        and not ctx.obj["standalone_mode"]
    ):
        return reactor

    # Return specified outputs.
    click.echo(f"Saving outputs to {inputs.output_path}")

    if data:
        click.echo(f"Saving output data as {outputs.params}")
        reactor.params.to_json(
            output_path=outputs.params,
            verbose=verbose,
        )

    if plots:
        click.echo(f"Saving output xz image as {outputs.plot_xz}")
        reactor.plot_xz()
        plt.savefig(outputs.plot_xz)
        click.echo(f"Saving output xy image as {outputs.plot_xy}")
        reactor.plot_xy()
        plt.savefig(outputs.plot_xy)

    if cad:
        click.echo(f"Saving output CAD model as {outputs.cad}")
        click.echo("Generating CAD model.")
        reactor.save_CAD_model(pattern="full")
        click.echo("CAD generation complete.")

    click.echo("BLUEPRINT run complete.")
    click.echo(
        f"All requested outputs have been generated, available in: {inputs.output_path}"
    )

    # Create tarball of output directory.
    if tarball:
        click.echo(f"Creating tarball of output directory as {outputs.tar}")
        with tarfile.open(outputs.tar, "w:gz") as tar:
            tar.add(os.path.join(inputs.output_path), arcname=os.path.sep)
