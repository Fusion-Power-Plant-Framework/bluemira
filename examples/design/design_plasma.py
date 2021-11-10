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

from bluemira.base.design import Design


build_config = {
    "Plasma": {
        "class": "MakeParameterisedPlasma",
        "param_class": "bluemira.equilibria.shapes::JohnerLCFS",
        "variables_map": {
            "r_0": "R_0",
            "a": "A",
        },
        "targets": {
            "Plasma/xz/LCFS": "build_xz",
            "Plasma/xy/LCFS": "build_xy",
            "Plasma/xyz/LCFS": "build_xyz",
        },
        "segment_angle": 270.0,
    },
}
params = {
    "R_0": (9.0, "Input"),
    "A": (3.5, "Input"),
}
design = Design(params, build_config)
design.run()

color = (0.80078431, 0.54, 0.80078431)
for dims in ["xz", "xy"]:
    component = design.component_manager.get_by_path(f"Plasma/{dims}/LCFS")
    component.plot_options.face_options["color"] = color
    component.plot_2d()

component = design.component_manager.get_by_path("Plasma/xyz/LCFS")
component.display_cad_options.color = color
component.display_cad_options.transparency = 0.2
component.show_cad()
