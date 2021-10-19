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
BLUEPRINT External Codes Wrapper
"""

from bluemira.codes.error import CodesError
from bluemira.codes import process


def run_systems_code(reactor, params_to_update=None):
    """
    Runs, reads or mocks PROCESS according to the build configuration dictionary.

    Parameters
    ----------
    reactor: class
        The instantiated reactor class for the run. Note the run mode is set by
        reactor.build_config.process_mode (or the build_config.json input file).

    params_to_update: list
        A list of parameter names compatible with the ParameterFrame class.
        If provided, parameters included in this list will be set up use BLUEPRINT
        values. All other parameters with a PROCESS mapping will be set to use
        PROCESS values. If None, the boolean which determines which value to use will
        be left unmodified for each parameter.

    Notes
    -----
    - "run": Run PROCESS creating a PROCESS input file (IN.DAT) from the
        BLUEPRINT inputs and template IN.DAT
    - "run input": Run PROCESS from an un-modified IN.DAT
    - "read": Read part of a PROCESS output file (MFILE.DAT)
    - "read all": Read all PROCESS mapped variable
    - "mock": Use a EU-DEMO default inputs without using PROCESS. Should not
        be used if PROCESS is installed

    Raises
    ------
    CodesError
        If PROCESS is being "run" but is not installed
    """
    process_mode = reactor.build_config["process_mode"]
    if (not process.PROCESS_ENABLED) and (process_mode.lower() != "mock"):
        raise CodesError("PROCESS not (properly) installed")

    process.Run(reactor, params_to_update)


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
