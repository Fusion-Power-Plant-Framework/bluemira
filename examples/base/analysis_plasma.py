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
A basic tutorial for configuring and running a design with a parameterised plasma.
"""

import matplotlib.pyplot as plt

import bluemira.base as bm_base
import bluemira.geometry as geo


build_config = {
    "Plasma": {
        "class": "MakeParameterisedPlasma",
        "param_class": "bluemira.equilibria.shapes::JohnerLCFS",
        "variables_map": {
            "r_0": "R_0",
            "a": "A",
        },
        "target": "Plasma/LCFS",
        "segment_angle": 270.0,
    },
}
params = {
    "R_0": (9.0, "Input"),
    "A": (3.5, "Input"),
}
design = bm_base.Design(params, build_config)
design.run()

for dims in ["xy", "xz"]:
    component: bm_base.PhysicalComponent = design.component_manager.get_by_path(
        "/".join([dims, build_config["Plasma"]["target"]])
    )

    _, ax = plt.subplots()
    for wire in component.shape.boundary:
        shape = wire.discretize()
        ax.plot(*shape.T[0::2])
        ax.set_aspect("equal")
    plt.show()

component: bm_base.PhysicalComponent = design.component_manager.get_by_path(
    "/".join(["xyz", build_config["Plasma"]["target"]])
)
geo.tools.save_as_STEP(component.shape, "plasma")
