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
Test the complete first wall builder, including divertor.
"""

import os

import numpy as np

from bluemira.base.file import get_bluemira_path
from bluemira.builders.EUDEMO.first_wall import FirstWallBuilder
from bluemira.equilibria import Equilibrium
from bluemira.equilibria.find import find_OX_points

DATA = get_bluemira_path("bluemira/equilibria/test_data", subfolder="tests")


class TestFirstWallBuilder:

    _default_variables_map = {
        "x1": {  # ib radius
            "value": "r_fw_ib_in",
        },
        "x2": {  # ob radius
            "value": "r_fw_ob_in",
        },
    }

    _default_config = {
        "param_class": "bluemira.builders.EUDEMO.first_wall::FirstWallPolySpline",
        "variables_map": _default_variables_map,
        "runmode": "mock",
        "name": "First Wall",
    }

    _params = {
        "Name": "First Wall Example",
        "plasma_type": "SN",
        # Wall shale opts
        "R_0": (9.0, "Input"),
        "kappa_95": (1.6 * (14 / 9), "Input"),
        "r_fw_ib_in": (5.8, "Input"),
        "r_fw_ob_in": (12.1, "Input"),
        "A": (3.1, "Input"),
        # Divertor opts
        "div_L2D_ib": (1.1, "Input"),
        "div_L2D_ob": (1.45, "Input"),
        "div_Ltarg": (0.5, "Input"),
        "div_open": (False, "Input"),
    }

    @classmethod
    def setup_class(cls):
        cls.eq = Equilibrium.from_eqdsk(os.path.join(DATA, "eqref_OOB.json"))
        _, cls.x_points = find_OX_points(cls.eq.x, cls.eq.z, cls.eq.psi())

    def test_wall_part_is_cut_below_x_point_in_z_axis(self):
        wall = FirstWallBuilder(
            self._params, build_config=self._default_config, equilibrium=self.eq
        )

        shape = wall.wall_part.get_component("first_wall").shape
        assert not shape.is_closed()
        # significant delta in assertion as the wire is discrete, so cut is not exact
        np.testing.assert_almost_equal(
            shape.bounding_box.z_min, self.x_points[0][1], decimal=1
        )
