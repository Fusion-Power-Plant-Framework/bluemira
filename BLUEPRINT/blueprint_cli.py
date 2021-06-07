# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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
import copy
from dataclasses import dataclass
import glob
import json
import matplotlib.pyplot as plt
import os
from pathlib import Path
import shutil
import sys
import tarfile

from BLUEPRINT.base.file import get_BP_root
from BLUEPRINT.reactor import ConfigurableReactor


@dataclass
class Output:
    """
    Class to determine what outputs should be produced from a BLUEPRINT run.
    """

    name: str
    log: bool = False
    data: bool = False
    plot_xz: bool = False
    plot_xy: bool = False
    cad: bool = False

    name_offset = 10

    def __str__(self):
        """
        Format the string representation of the Output.
        """
        name_pad = self.name_offset - len(self.name)

        settings = []
        for name, value in self.__dict__.items():
            if name != "name":
                settings.append(name if value else "_" * len(name))
        settings = ", ".join(settings)
        return f"{self.name}{' ' * name_pad}: {settings}"

    def __setattr__(self, name, value):
        """
        Avoid changing the value of an attribute to None.
        """
        if value is not None:
            self.__dict__[name] = value

    def clone(self):
        """
        Create a copy of the Output.
        """
        return copy.copy(self)


DEFAULT = Output("default", log=True, data=True, plot_xz=True, plot_xy=True)
FULL = Output("full", log=True, data=True, plot_xz=True, plot_xy=True, cad=True)
LITE = Output("lite", log=True, data=True)
CAD = Output("cad", log=True, cad=True)
NONE = Output("none")


def set_output_mode(mode):
    """
    Set the Output as specified by the mode string.
    """
    result = None
    if mode == "default" or mode is None:
        result = DEFAULT
    elif mode == "full":
        result = FULL
    elif mode == "lite":
        result = LITE
    elif mode == "cad":
        result = CAD
    elif mode == "none":
        result = NONE

    if result is not None:
        return result.clone()


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
    "-o",
    "--outdir",
    type=click.Path(writable=True),
    default=f"{get_BP_root()}/generated_data/reactors",
    help="Specifies the directory in which to store output files. Note that outputs \
    will be stored in a subdirectory within the directory provided, corresponding to \
    the specified reactor name. The directory must not exist before running BLUEPRINT \
    to avoid unintentional overwrites.",
)
@click.option(
    "-ro",
    "--reactornameout",
    default=None,
    help="Specifies a reactor name, overiding the value in the input config file. \
    Also used as the output subdirectory name and as a prefix to each output file.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    help="Enables verbose mode. When on, output data will include metadata.",
)
@click.option(
    "-t",
    "--tarball",
    is_flag=True,
    help="Enables creation of a tarball of the output directory.",
)
@click.option(
    "-m",
    "--outmode",
    default="default",
    help="""
Sets the output mode, specifying which outputs to save.\n\
Can be:

\b
"""
    + "\n".join([str(DEFAULT), str(FULL), str(LITE), str(CAD), str(NONE)])
    + """

To override the outmode setting for a specific output, use the options below.
""",
)
@click.option(
    "--log/--no_log",
    default=None,
    help="Enables/disables output of BLUEPRINT text dump.",
)
@click.option(
    "--data/--no_data",
    default=None,
    help="Enables/disables output of the .json data file.",
)
@click.option(
    "--plot_xz/--no_plot_xz",
    default=None,
    help="Enables/disables output of the 2D reactor image in the xz plane.",
)
@click.option(
    "--plot_xy/--no_plot_xy",
    default=None,
    help="Enables/disables output of the 2D reactor image in the xy plane.",
)
@click.option(
    "--cad/--no_cad",
    default=None,
    help="Enables/disables output of the 3D cad model.",
)
def cli(
    template,
    config,
    build_config,
    build_tweaks,
    indir,
    reactornamein,
    outdir,
    reactornameout,
    verbose,
    tarball,
    outmode,
    log,
    data,
    plot_xz,
    plot_xy,
    cad,
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
    # Set output flags according to outmode.
    output = set_output_mode(outmode)

    if output is None:
        click.echo("Invalid outmode. See blueprint --help for options.")
        return

    # Set any explicit output flags.
    # Note that the Output class skips assignment if a value is None.
    output.log = log
    output.data = data
    output.plot_xz = plot_xz
    output.plot_xy = plot_xy
    output.cad = cad

    # Set input paths.
    if reactornamein is not None:
        template = f"{reactornamein}_{template}"
        config = f"{reactornamein}_{config}"
        build_config = f"{reactornamein}_{build_config}"
        build_tweaks = f"{reactornamein}_{build_tweaks}"

    template_path = os.path.join(indir, template)
    config_path = os.path.join(indir, config)
    build_config_path = os.path.join(indir, build_config)
    build_tweaks_path = os.path.join(indir, build_tweaks)

    # Get reactor name, using override if provided.
    if reactornameout is not None:
        reactorname = reactornameout
    elif reactornamein is not None:
        reactorname = reactornamein
    else:
        with open(config_path, "r") as fh:
            config_dict = json.load(fh)
        reactorname = config_dict["Name"]

    # Set output directory path.
    output_path = os.path.join(outdir, reactorname)
    if os.path.exists(output_path):
        raise FileExistsError(
            f"Output directory {output_path} already exists for this reactor. "
            "Select a new outdir, change the reactor name, or remove the existing "
            "directory and run again."
        )
    Path(output_path).mkdir(parents=True, exist_ok=True)
    click.echo(f"Saving outputs in {output_path}/")

    # Copy inputs to output directory.
    template_path = shutil.copy(
        template_path,
        os.path.join(output_path, f"{reactorname}_template.json"),
    )
    config_path = shutil.copy(
        config_path,
        os.path.join(output_path, f"{reactorname}_config.json"),
    )
    build_config_path = shutil.copy(
        build_config_path,
        os.path.join(output_path, f"{reactorname}_build_config.json"),
    )
    build_tweaks_path = shutil.copy(
        build_tweaks_path,
        os.path.join(output_path, f"{reactorname}_build_tweaks.json"),
    )

    # Override reactor name if specified.
    if reactornameout is not None:
        config_dict = {}
        with open(config_path, "r") as fh:
            config_dict = json.load(fh)
        config_dict["Name"] = reactorname
        with open(config_path, "w") as fh:
            json.dump(config_dict, fh)

    # Instantiate BLUEPRINT reactor class.
    CR = ConfigurableReactor.from_json(
        template_path, config_path, build_config_path, build_tweaks_path
    )

    # Return output log.
    if output.log:
        filename_out = "output.txt"
        filename_err = "errors.txt"
        path_to_out = os.path.join(output_path, filename_out)
        path_to_err = os.path.join(output_path, filename_err)
        click.echo(f"Use tail -f {path_to_out} to view output while BLUEPRINT runs.")
        click.echo(f"See {path_to_err} to view error messages.")

        sys.stdout = open(os.path.join(output_path, filename_out), "w")
        sys.stderr = open(os.path.join(output_path, filename_err), "w")
        click.echo(f"Saving output log as {filename_out}")
        click.echo(f"Saving errors log as {filename_err}")

    # Return user inputs.
    CR.config_to_json(output_path)

    # Run BLUEPRINT build.
    click.echo("Running BLUEPRINT build.")
    CR.build()
    click.echo("BLUEPRINT build complete.")

    # Return specified outputs.
    generated_data_directory = CR.file_manager.generated_data_dirs["root"]

    if output.data:
        filename = f"{reactorname}_params.json"
        click.echo(f"Saving output data as {filename}")
        CR.params.to_json(
            output_path=os.path.join(generated_data_directory, filename),
            verbose=verbose,
        )

    if output.plot_xz:
        filename = f"{reactorname}_XZ.png"
        click.echo(f"Saving output xz image as {filename}")
        CR.plot_xz()
        plt.savefig(os.path.join(generated_data_directory, "plots", filename))

    if output.plot_xy:
        filename = f"{reactorname}_XY.png"
        click.echo(f"Saving output xy image as {filename}")
        CR.plot_xy()
        plt.savefig(os.path.join(generated_data_directory, "plots", filename))

    if output.cad:
        filename = f"{reactorname}_CAD_MODEL.stp"
        click.echo(f"Saving output CAD model as {filename}")
        click.echo("Generating CAD model.")
        CR.save_CAD_model(pattern="full")
        click.echo("CAD generation complete.")

    # Move all generated data to output directory.
    files = glob.glob(os.path.join(generated_data_directory, "*"))
    for f in files:
        shutil.move(f, output_path)
    click.echo("BLUEPRINT run complete.")
    click.echo(f"All requested outputs have been generated, available at: {output_path}")

    # Create tarball of output directory.
    if tarball:
        filename = f"{reactorname}.tar"
        click.echo(f"Creating tarball of output directory as {filename}")
        with tarfile.open(os.path.join(output_path, filename), "w:gz") as tar:
            tar.add(os.path.join(output_path), arcname=os.path.sep)
