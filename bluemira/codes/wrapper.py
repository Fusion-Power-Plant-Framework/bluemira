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
Bluemira External Codes Wrapper
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from bluemira.codes.interface import FileProgramInterface
from bluemira.codes.utilities import get_code_interface

if TYPE_CHECKING:
    from bluemira.base.builder import BuildConfig
    from bluemira.base.parameter import ParameterFrame


def systems_code_solver(
    params: ParameterFrame,
    build_config: BuildConfig,
    run_dir: Optional[str] = None,
    read_dir: Optional[str] = None,
    template_indat=None,
    module: Optional[str] = "PROCESS",
) -> FileProgramInterface:
    """
    Runs, reads or mocks systems code according to the build configuration dictionary.

    Parameters
    ----------
    params: ParameterFrame
        ParameterFrame for code
    build_config: Dict
        build configuration dictionary
    run_dir: str
        Path to the run directory, where the main executable is located
        and the input/output files will be written.
    read_dir: str
        Path to the read directory, where the output files from a run are
        read in
    template_indat: str
        Path to the template file to be used for the run.

    Returns
    -------
    Solver object: FileProgramInterface
        The solver that has been run.

    Raises
    ------
    CodesError
        If PROCESS is not being mocked and is not installed.

    """
    # Remove me, temp compatibility layer
    build_config["mode"] = build_config.get("mode", build_config["process_mode"])
    # #####################################
    syscode = get_code_interface(module)

    return syscode.Solver(params, build_config, run_dir, read_dir, template_indat)


def plot_radial_build(
    filename: str, width: float = 1.0, show: bool = True, module="PROCESS"
):
    """
    Systems code radial build

    Parameters
    ----------
    filename: str
        The directory containing the system code run results.
    width: float
        The relative width of the plot.
    show: bool
        If True then immediately display the plot, else delay displaying the plot until
        the user shows it, by default True.

    """
    syscode = get_code_interface(module)

    return syscode.plot_radial_build(filename, width, show)


def transport_code_solver(
    params: ParameterFrame,
    build_config: BuildConfig,
    run_dir: Optional[str] = None,
    read_dir: Optional[str] = None,
    module: Optional[str] = "PLASMOD",
) -> FileProgramInterface:
    """
    Transport solver

    Parameters
    ----------
    params: ParameterFrame
        ParameterFrame for plasmod
    build_config: Dict
        build configuration dictionary
    run_dir: str
        Plasmod run directory
    read_dir: str
        Directory to read in previous run

    Returns
    -------
    Solver object: FileProgramInterface

    """
    transp = get_code_interface(module)

    return transp.Solver(params, build_config, run_dir, read_dir)
