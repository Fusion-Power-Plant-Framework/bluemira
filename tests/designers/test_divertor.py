# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Test divertor designer.
"""

import copy
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from bluemira.base.constants import EPS
from bluemira.base.file import get_bluemira_path
from bluemira.builders.divertor import DivertorDesigner, LegPosition
from bluemira.equilibria import Equilibrium
from bluemira.equilibria.find import find_OX_points
from bluemira.geometry.tools import make_polygon, signed_distance

DATA = get_bluemira_path("equilibria/test_data", subfolder="tests")


def get_turning_point_idxs(z: np.ndarray):
    diff = np.diff(z)
    return np.argwhere(diff[1:] * diff[:-1] < 0)


class TestDivertorDesigner:
    _default_params: ClassVar = {
        "div_L2D_ib": {"value": 1.1, "unit": "m"},
        "div_L2D_ob": {"value": 1.45, "unit": "m"},
        "div_Ltarg_ib": {"value": 0.5, "unit": "m"},
        "div_Ltarg_ob": {"value": 0.5, "unit": "m"},
        "strike_loc_ib": {"value": 0.5, "unit": ""},
        "strike_loc_ob": {"value": 0.5, "unit": ""},
        "div_targ_angle_ib": {"value": 42, "unit": "degrees"},
        "div_targ_angle_ob": {"value": -25, "unit": "degrees"},
        "div_targ_type_ib": {"value": "verticle", "unit": ""},
        "div_targ_type_ob": {"value": "verticle", "unit": ""},
        "div_baffle_type_ib": {"value": "circle_baffle", "unit": ""},
        "div_baffle_type_ob": {"value": "circle_baffle", "unit": ""},
    }

    @classmethod
    def setup_class(cls):
        cls.eq = Equilibrium.from_eqdsk(Path(DATA, "eqref_OOB.json"), from_cocos=7)
        cls.separatrix = make_polygon(cls.eq.get_separatrix().xyz.T)
        _, cls.x_points = find_OX_points(cls.eq.x, cls.eq.z, cls.eq.psi())
        cls.params = copy.deepcopy(cls._default_params)
        cls.x_limits = (5, 11)
        cls.z_limits = (cls.x_points[0][1], cls.x_points[0][1])

    @pytest.mark.parametrize("axis", ["x", "z"])
    @pytest.mark.parametrize("i", [0, 1])
    def test_wire_ends(self, axis, i):
        designer = DivertorDesigner(self.params, self.eq, self.x_limits, self.z_limits)
        inner_target = designer._make_target(LegPosition.INNER, "inner_target")
        largest = designer._get_wire_end_with_largest(inner_target, axis)
        smallest = designer._get_wire_end_with_smallest(inner_target, axis)
        assert largest[i] > smallest[i]

    def test_wire_ends_by_psi(self):
        designer = DivertorDesigner(self.params, self.eq, self.x_limits, self.z_limits)
        inner_target = designer._make_target(LegPosition.INNER, "inner_target")
        outer_target = designer._make_target(LegPosition.OUTER, "outer_target")
        inner_target_start, inner_target_end = designer._get_wire_ends_by_psi(
            inner_target
        )
        outer_target_end, outer_target_start = designer._get_wire_ends_by_psi(
            outer_target
        )
        assert inner_target_start[0] > inner_target_end[0]
        assert outer_target_start[0] > outer_target_end[0]

    def test_new_sets_leg_lengths(self):
        # Note:moved from  original location in
        # test_divertor_silhouette in eudemo_tests.
        params = copy.deepcopy(self.params)
        params["div_L2D_ib"]["value"] = 5
        params["div_L2D_ob"]["value"] = 10

        designer = DivertorDesigner(params, self.eq, self.x_limits, self.z_limits)

        assert designer.leg_length[LegPosition.INNER].value == 5
        assert designer.leg_length[LegPosition.OUTER].value == 10

    def test_targets_intersect_separatrix(self):
        designer = DivertorDesigner(self.params, self.eq, self.x_limits, self.z_limits)
        inner_target = designer._make_target(LegPosition.INNER, "inner_target")
        outer_target = designer._make_target(LegPosition.OUTER, "outer_target")
        for target in [inner_target, outer_target]:
            assert signed_distance(target, self.separatrix) == pytest.approx(0)

    @pytest.mark.parametrize(("div_ltarg", "div_targ_angle"), [(1.5, 52)])
    def test_target_length_set_by_parameter(self, div_ltarg, div_targ_angle):
        # Note:moved from  original location in
        # test_divertor_silhouette in eudemo_tests.
        params = copy.deepcopy(self.params)
        params["div_Ltarg_ib"]["value"] = div_ltarg / 2
        params["div_Ltarg_ob"]["value"] = div_ltarg / 2
        params["div_targ_angle_ib"]["value"] = div_targ_angle

        designer = DivertorDesigner(params, self.eq, self.x_limits, self.z_limits)
        inner_target = designer._make_target(LegPosition.INNER, "inner_target")
        outer_target = designer._make_target(LegPosition.OUTER, "outer_target")

        for target in [inner_target, outer_target]:
            assert target.length == pytest.approx(div_ltarg / 2, rel=EPS)

    def test_make_dome(self):
        # Note: these tests have been moved from their original location in
        # test_divertor_silhouette in eudemo_tests.
        designer = DivertorDesigner(self.params, self.eq, self.x_limits, self.z_limits)
        inner_target = designer._make_target(LegPosition.INNER, "inner_target")
        outer_target = designer._make_target(LegPosition.OUTER, "outer_target")
        _, inner_target_end = designer._get_wire_ends_by_psi(inner_target)
        _, outer_target_start = designer._get_wire_ends_by_psi(outer_target)

        dome = designer.make_dome(inner_target_end, outer_target_start, label="dome")

        # test dome intersects targets
        assert signed_distance(dome, inner_target) == pytest.approx(0)
        assert signed_distance(dome, outer_target) == pytest.approx(0)

        # test dome does not intersect separatrix
        assert signed_distance(dome, self.separatrix) < 0

        # test SN lower dome has turning point below x-point
        dome_coords = dome.discretise()
        turning_points = get_turning_point_idxs(dome_coords[2, :])
        assert len(turning_points) == 1
        assert dome_coords[2, turning_points[0]] < self.x_points[0].z

    @pytest.mark.parametrize(
        "baffle_type", ["circle_baffle", "straight_baffle", "fluxline_baffle"]
    )
    def test_make_baffle(self, baffle_type):

        params = copy.deepcopy(self.params)
        params["div_baffle_type_ib"]["value"] = baffle_type
        params["div_baffle_type_ob"]["value"] = baffle_type

        if baffle_type == "fluxline_baffle":
            params["div_L2D_ib"]["value"] = 2
            params["div_L2D_ob"]["value"] = 2

        designer = DivertorDesigner(params, self.eq, self.x_limits, self.z_limits)

        inner_target = designer._make_target(LegPosition.INNER, "inner_target")
        outer_target = designer._make_target(LegPosition.OUTER, "outer_target")
        inner_target_start, inner_target_end = designer._get_wire_ends_by_psi(
            inner_target
        )
        outer_target_end, outer_target_start = designer._get_wire_ends_by_psi(
            outer_target
        )

        # Build the baffles
        inner_baffle = designer.make_baffle(
            "inner_baffle",
            target_baffle_join_point=inner_target_start,
            target_dome_join_point=inner_target_end,
        )
        outer_baffle = designer.make_baffle(
            "outer_baffle",
            target_baffle_join_point=outer_target_end,
            target_dome_join_point=outer_target_start,
        )

        # test baffles intersect targets
        assert signed_distance(inner_baffle, inner_target) == pytest.approx(0)
        assert signed_distance(outer_baffle, outer_target) == pytest.approx(0)

        # # test baffles do not intersect separatrix
        assert signed_distance(inner_baffle, self.separatrix) < 0
        assert signed_distance(outer_baffle, self.separatrix) < 0
