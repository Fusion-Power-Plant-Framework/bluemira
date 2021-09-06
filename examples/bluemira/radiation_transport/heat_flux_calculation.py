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
import matplotlib.pyplot as plt
from bluemira.base.file import get_bluemira_path
from BLUEPRINT.base.parameter import ParameterFrame
from BLUEPRINT.equilibria.equilibrium import Equilibrium
from bluemira.geometry._deprecated_loop import Loop
from bluemira.radiation_transport.advective_transport import ChargedParticleSolver

read_path = get_bluemira_path("BLUEPRINT/equilibria", subfolder="data")
eq_name = "EU-DEMO_EOF.json"
eq_name = os.sep.join([read_path, eq_name])
eq = Equilibrium.from_eqdsk(eq_name, load_large_file=True)

read_path = get_bluemira_path("bluemira/radiation_transport", subfolder="examples")
fw_name = "first_wall.json"
fw_name = os.sep.join([read_path, fw_name])
fw_shape = Loop.from_file(fw_name)


params = ParameterFrame(
    [
        ["plasma_type", "Type of plasma", "SN", "N/A", None, "Input"],
        ["fw_p_sol_near", "near scrape-off layer power", 50, "MW", None, "Input"],
        ["fw_p_sol_far", "far scrape-off layer power", 50, "MW", None, "Input"],
        ["fw_lambda_q_near", "Lambda q near SOL", 0.05, "m", None, "Input"],
        ["fw_lambda_q_far", "Lambda q far SOL", 0.05, "m", None, "Input"],
        ["f_outer_target", "Power fraction", 0.75, "N/A", None, "Input"],
        ["f_inner_target", "Power fraction", 0.25, "N/A", None, "Input"],
    ]
)

solver = ChargedParticleSolver(params, eq)
xx, zz, hh = solver.analyse(first_wall=fw_shape)

# Plot the analysis

f, ax = plt.subplots()
fw_shape.plot(ax, fill=False)
eq.get_separatrix().plot(ax)

for flux_surface in solver.flux_surfaces:
    flux_surface.plot(ax)

cm = ax.scatter(xx, zz, c=hh, cmap="plasma", zorder=40)
f.colorbar(cm, label="MW/m^2")
plt.show()
