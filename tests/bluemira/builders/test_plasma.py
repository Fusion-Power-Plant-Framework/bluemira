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
Tests for plasma builders
"""

from bluemira.builders.plasma import MakeParameterisedPlasma

import tests


class TestMakeParameterisedPlasma:
    def test_builder(self):
        params = {
            "R_0": (9.0, "Input"),
            "A": (3.5, "Input"),
        }
        build_config = {
            "name": "Plasma",
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
        }
        builder = MakeParameterisedPlasma(params, build_config)
        outputs = builder(params)
        assert outputs is not None
        assert isinstance(outputs, list)
        assert len(outputs) == 3

        for idx, target in enumerate(build_config["targets"]):
            target_split = target.split("/")
            target_path = "/".join(target_split)
            component_name = target_split[-1]

            assert outputs[idx][0] == target_path

            component = outputs[idx][1]
            assert component.name == component_name

            if tests.PLOTTING:
                color = (0.80078431, 0.54, 0.80078431)
                if "xyz" in target:
                    component.display_cad_options.color = color
                    component.display_cad_options.transparency = 0.2
                    component.show_cad()
                else:
                    component.plot_options.face_options["color"] = color
                    component.plot_2d()
