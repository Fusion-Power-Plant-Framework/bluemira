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
Test for transport solver
"""
import bluemira.codes.plasmod as plasmod
from bluemira.base.config import Configuration
from bluemira.base.logs import set_log_level
from bluemira.equilibria.fem_fixed_boundary.transport_solver import (
    PlasmodTransportSolver,
)

set_log_level("DEBUG")

PLASMOD_PATH = "/home/ivan/Desktop/bluemira_project/plasmod-master/bin/"
binary = f"{PLASMOD_PATH}plasmod"

new_params = {
    "A": 3.3,
    "R_0": 9.002,
    "I_p": 19.75,
    "B_0": 5.855,
    "V_p": -2500,
    "v_burn": -1.0e6,
    "kappa_95": 1.652,
    "delta_95": 0.333,
    "delta": 0.38491934960310104,
    "kappa": 1.6969830041844367,
}

params = Configuration(new_params)

for param_name in params.keys():
    if param_name in new_params:
        param = params.get_param(param_name)
        param.source = "Plasmod Example"

build_config = {
    "mode": "run",
    "binary": binary,
}

plasmod_solver = PlasmodTransportSolver(
    params=params,
    build_config=build_config,
)

