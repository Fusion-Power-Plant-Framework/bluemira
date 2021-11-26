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

from bluemira.builders.plasma import MakeParameterisedPlasma

from bluemira.base.design import Design


build_config = {
    "Plasma": {
        "class": "MakeParameterisedPlasma",
        "param_class": "bluemira.equilibria.shapes::JohnerLCFS",
        "variables_map": {
            "r_0": "R_0",
            "a": "A",
        },
    },
}
params = {
    "Name": "A Plasma Design",
    "R_0": (9.0, "Input"),
    "A": (3.5, "Input"),
}
design = Design(params, build_config)
result = design.run()

color = (0.80078431, 0.54, 0.80078431)
for dims in ["xz", "xy"]:
    lcfs = result.get_component("Plasma").get_component(dims).get_component("LCFS")
    lcfs.plot_options.face_options["color"] = color
    lcfs.plot_2d()

lcfs = result.get_component("Plasma").get_component("xyz").get_component("LCFS")
lcfs.display_cad_options.color = color
lcfs.display_cad_options.transparency = 0.2
lcfs.show_cad()

plasma_builder: MakeParameterisedPlasma = design.get_builder("Plasma")
lcfs = plasma_builder.build_xyz(segment_angle=270.0)
lcfs.display_cad_options.color = color
lcfs.show_cad()
