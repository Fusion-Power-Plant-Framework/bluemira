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
        If provided, parameters included in this list will be modified to write their
        values to PROCESS inputs, while all others will be modified to not be written to
        the PROCESS inputs. By default, None.

    Notes
    -----
    - "run": Run PROCESS within a bluemira run to generate an radial build.
        Creates a new input file from a template IN.DAT modified with updated parameters
        from the bluemira run mapped with write=True. If params_to_update are provided
        then these will be modified to have write=True and all other will be modified to
        have write=False.
    - "runinput": Run PROCESS from an unmodified input file (IN.DAT), generating the
        radial build to use as the input to the bluemira run. Overrides the write
        mapping of all parameters to be False.
    - "read": Load the radial build from a previous PROCESS run (MFILE.DAT). Loads
        only the parameters mapped with read=True.
    - "readall": Load the radial build from a previous PROCESS run (MFILE.DAT). Loads
        all values with a bluemira mapping regardless of the mapping.read bool.
        Overrides the read mapping of all parameters to be True.
    - "mock": Run bluemira without running PROCESS, using the default radial build based
        on EU-DEMO. This option should not be used if PROCESS is installed, except for
        testing purposes.

    Raises
    ------
    CodesError
        If PROCESS is not being mocked and is not installed.
    """
    process_mode = reactor.build_config["process_mode"]
    if (not process.PROCESS_ENABLED) and (process_mode.lower() != "mock"):
        raise CodesError("PROCESS not (properly) installed")

    process.Run(reactor, params_to_update)
