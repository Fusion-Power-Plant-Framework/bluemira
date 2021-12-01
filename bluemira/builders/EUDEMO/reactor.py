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
Perform the EU-DEMO design.
"""

from bluemira.base.parameter import ParameterFrame
from bluemira.base.design import Reactor
from bluemira.base.look_and_feel import bluemira_print
from bluemira.codes import run_systems_code
from bluemira.codes.process import NAME as PROCESS


class EUDEMO(Reactor):
    """
    The EU-DEMO Reactor object encapsulates the logic for performing an EU-DEMO tokamak
    design.
    """

    def run(self):
        """
        Run the EU-DEMO reactor build process. Performs the following tasks:

        - Run the (PROCESS) systems code
        """
        super().run()

        self.run_systems_code()

    def run_systems_code(self):
        """
        Run the systems code module in the requested run mode.
        """
        bluemira_print(f"Running: {PROCESS}")

        output: ParameterFrame = run_systems_code(
            self._params,
            self._build_config,
            self._file_manager.generated_data_dirs["systems_code"],
            self._file_manager.reference_data_dirs["systems_code"],
        )
        self._params.update_kw_parameters(output.to_dict())
