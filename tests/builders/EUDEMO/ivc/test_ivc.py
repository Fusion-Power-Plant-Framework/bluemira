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

import copy
import os

import numpy as np
import pytest

from bluemira.base.error import BuilderError
from bluemira.base.file import get_bluemira_path
from bluemira.builders.EUDEMO.ivc import InVesselComponentBuilder
from bluemira.builders.EUDEMO.ivc.wall import WallBuilder
from bluemira.equilibria import Equilibrium
from bluemira.equilibria.find import find_OX_points

DATA = get_bluemira_path("equilibria/test_data", subfolder="tests")
OPTIMISER_MODULE_REF = "bluemira.geometry.optimisation"
WALL_MODULE_REF = "bluemira.builders.EUDEMO.ivc.wall"


class TestInVesselComponentBuilder:

    _default_variables_map = {
        "x1": {  # ib radius
            "value": "r_fw_ib_in",
        },
        "x2": {  # ob radius
            "value": "r_fw_ob_in",
        },
    }

    _default_config = {
        "name": "In Vessel Components",
        "param_class": f"{WALL_MODULE_REF}::WallPrincetonD",
        "problem_class": f"{OPTIMISER_MODULE_REF}::MinimiseLengthGOP",
        "runmode": "mock",
        "variables_map": _default_variables_map,
    }

    _params = {
        "Name": "IVC Example",
        "plasma_type": "SN",
        # Wall shape opts
        "R_0": (9.0, "Input"),
        "kappa_95": (2.4, "Input"),
        "r_fw_ib_in": (5, "Input"),
        "r_fw_ob_in": (13, "Input"),
        "fw_psi_n": (1.07, "Input"),
        "tk_sol_ib": (0.225, "Input"),
        "A": (3.1, "Input"),
        # Divertor opts
        "div_L2D_ib": (1.1, "Input"),
        "div_L2D_ob": (1.45, "Input"),
        "div_Ltarg": (0.5, "Input"),
        "div_open": (False, "Input"),
        # Blanket opts
        "tk_bb_ib": (0.8, "Input"),
        "tk_bb_ob": (1.1, "Input"),
    }

    @classmethod
    def setup_class(cls):
        cls.eq = Equilibrium.from_eqdsk(os.path.join(DATA, "eqref_OOB.json"))
        _, cls.x_points = find_OX_points(cls.eq.x, cls.eq.z, cls.eq.psi())

    def test_wall_boundary_is_cut_below_x_point_in_z_axis(self):
        builder = InVesselComponentBuilder(
            self._params, build_config=self._default_config, equilibrium=self.eq
        )

        first_wall = builder()

        shape = first_wall.get_component(WallBuilder.COMPONENT_WALL_BOUNDARY).shape
        assert not shape.is_closed()
        # significant delta in assertion as the wire is discrete, so cut is not exact
        np.testing.assert_almost_equal(
            shape.bounding_box.z_min, self.x_points[0][1], decimal=1
        )

    def test_contains_one_divertor_component_given_SN_plasma(self):
        builder = InVesselComponentBuilder(
            self._params, build_config=self._default_config, equilibrium=self.eq
        )

        wall = builder()

        divertors = wall.get_component(
            InVesselComponentBuilder.COMPONENT_DIVERTOR, first=False
        )
        assert len(divertors) == 1

    def test_contains_one_wall_component(self):
        builder = InVesselComponentBuilder(
            self._params, build_config=self._default_config, equilibrium=self.eq
        )

        wall = builder()

        walls = wall.get_component(InVesselComponentBuilder.COMPONENT_WALL, first=False)
        assert len(walls) == 1

    def test_component_tree_structure(self):
        builder = InVesselComponentBuilder(
            self._params, build_config=self._default_config, equilibrium=self.eq
        )

        first_wall = builder()

        assert first_wall.is_root
        xz = first_wall.get_component("xz")
        assert xz is not None
        assert xz.depth == 1
        divertor_xz = xz.get_component(InVesselComponentBuilder.COMPONENT_DIVERTOR)
        assert divertor_xz.depth == 2
        wall_xz = xz.get_component(InVesselComponentBuilder.COMPONENT_WALL)
        assert wall_xz is not None
        assert wall_xz.depth == 2

    def test_BuilderError_if_wall_does_not_enclose_x_point(self):
        params = copy.deepcopy(self._params)
        params.update({"r_fw_ib_in": 7, "r_fw_ob_in": 9})
        builder = InVesselComponentBuilder(
            params, build_config=self._default_config, equilibrium=self.eq
        )

        with pytest.raises(BuilderError):
            builder()
