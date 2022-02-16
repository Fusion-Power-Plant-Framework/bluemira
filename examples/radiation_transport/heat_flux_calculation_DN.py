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
Example single null first wall particle heat flux
"""

import os

from bluemira.base.file import get_bluemira_path
from bluemira.base.parameter import ParameterFrame
from bluemira.equilibria import Equilibrium
from bluemira.geometry._deprecated_loop import Loop
from bluemira.radiation_transport.advective_transport import ChargedParticleSolver

read_path = get_bluemira_path("equilibria", subfolder="data")
eq_name = "DN-DEMO_eqref.json"
eq_name = os.sep.join([read_path, eq_name])
eq = Equilibrium.from_eqdsk(eq_name, load_large_file=True)

read_path = get_bluemira_path(
    "bluemira/radiation_transport/test_data", subfolder="tests"
)
fw_name = "DN_fw_shape.json"
fw_name = os.sep.join([read_path, fw_name])
fw_shape = Loop.from_file(fw_name)

params = ParameterFrame(
    # fmt: off
    [
        ["P_sep_particle", "Separatrix power", 150, "MW", None, "Input"],
        ["f_p_sol_near", "near scrape-off layer power rate", 0.65, "dimensionless", None, "Input"],
        ["fw_lambda_q_near_omp", "Lambda q near SOL at the outboard", 0.003, "m", None, "Input"],
        ["fw_lambda_q_far_omp", "Lambda q far SOL at the outboard", 0.1, "m", None, "Input"],
        ["fw_lambda_q_near_imp", "Lambda q near SOL at the inboard", 0.003, "m", None, "Input"],
        ["fw_lambda_q_far_imp", "Lambda q far SOL at the inboard", 0.1, "m", None, "Input"],
        ["f_lfs_lower_target", "Fraction of SOL power deposited on the LFS lower target", 0.9 * 0.5, "dimensionless", None, "Input"],
        ["f_hfs_lower_target", "Fraction of SOL power deposited on the HFS lower target", 0.1 * 0.5, "dimensionless", None, "Input"],
        ["f_lfs_upper_target", "Fraction of SOL power deposited on the LFS upper target (DN only)", 0.9 * 0.5, "dimensionless", None, "Input"],
        ["f_hfs_upper_target", "Fraction of SOL power deposited on the HFS upper target (DN only)", 0.1 * 0.5, "dimensionless", None, "Input"],
    ]
    # fmt: on
)


solver = ChargedParticleSolver(params, eq, dx_mp=0.001)
x, z, hf = solver.analyse(first_wall=fw_shape)

solver.plot()
