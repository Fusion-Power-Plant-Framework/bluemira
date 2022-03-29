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

import sys
from unittest import mock

import tests
from bluemira.base.components import Component, PhysicalComponent
from bluemira.builders.plasma import MakeParameterisedPlasma


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
        }
        builder = MakeParameterisedPlasma(params, build_config)
        component = builder()
        assert component is not None
        assert isinstance(component, Component)
        assert len(component.children) == 3

        dims = ["xz", "xy", "xyz"]
        child: Component
        for child, dim in zip(component.children, dims):
            assert child.name == dim

            assert len(child.children) == 1

            lcfs: PhysicalComponent = child.get_component("LCFS")
            assert lcfs is not None

            if tests.PLOTTING:
                color = (0.80078431, 0.54, 0.80078431)
                if dim == "xyz":
                    lcfs.display_cad_options.color = color
                    lcfs.display_cad_options.transparency = 0.2
                    lcfs.show_cad()
                else:
                    lcfs.plot_options.face_options["color"] = color
                    lcfs.plot_2d()

    def test_builder_with_import_isolation(self):
        """Check that the build works with a clean set of imports."""
        with mock.patch.dict(sys.modules):
            for mod in list(sys.modules.keys()):
                if mod.startswith("bluemira"):
                    sys.modules.pop(mod)

            from bluemira.builders.plasma import MakeParameterisedPlasma

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
            }
            builder = MakeParameterisedPlasma(params, build_config)
            plasma = builder().get_component("xz").get_component("LCFS")

            assert plasma is not None
