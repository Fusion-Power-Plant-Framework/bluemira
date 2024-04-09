# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Bluemira External Codes Wrapper
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bluemira.codes.utilities import get_code_interface

if TYPE_CHECKING:
    from bluemira.base.builder import BuildConfig
    from bluemira.base.parameter_frame import Parameter as ParameterFrame
    from bluemira.codes.interface import CodesSolver


def systems_code_solver(
    params: ParameterFrame,
    build_config: BuildConfig,
    module: str = "PROCESS",
) -> CodesSolver:
    """
    Runs, reads or mocks systems code according to the build configuration dictionary.

    Parameters
    ----------
    params:
        ParameterFrame for code
    build_config:
        build configuration dictionary
    module:
        Module to use

    Returns
    -------
    The solver that has been run.

    Raises
    ------
    CodesError
        If the system code is not being mocked and is not installed, or
        there is a problem running the system code.
    """
    syscode = get_code_interface(module)
    return syscode.Solver(params, build_config)


def plot_radial_build(
    filename: str, width: float = 1.0, *, show: bool = True, module: str = "PROCESS"
):
    """
    Systems code radial build

    Parameters
    ----------
    filename:
        The directory containing the system code run results.
    width:
        The relative width of the plot.
    show:
        If True then immediately display the plot, else delay displaying the plot until
        the user shows it, by default True.
    module:
        Module to use
    """
    syscode = get_code_interface(module)

    return syscode.plot_radial_build(filename, width, show=show)


def transport_code_solver(
    params: ParameterFrame,
    build_config: BuildConfig,
    module: str = "PLASMOD",
) -> CodesSolver:
    """
    Transport solver

    Parameters
    ----------
    params:
        ParameterFrame for plasmod
    build_config:
        build configuration dictionary
    module:
        Module to use

    Returns
    -------
    The solver object to be run
    """
    transp = get_code_interface(module)
    return transp.Solver(params, build_config)
