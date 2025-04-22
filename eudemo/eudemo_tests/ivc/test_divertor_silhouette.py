# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Test divertor silhouette designer.
"""

import copy
from pathlib import Path
from typing import ClassVar
from unittest import mock

import numpy as np
import pytest

from bluemira.base.constants import EPS
from bluemira.base.file import get_bluemira_path
from bluemira.equilibria import Equilibrium
from bluemira.equilibria.find import find_OX_points
from bluemira.geometry.tools import make_polygon, signed_distance
from eudemo.ivc import DivertorSilhouetteDesigner
from eudemo.ivc.divertor_silhouette import LegPosition

DATA = get_bluemira_path("equilibria/test_data", subfolder="tests")


def get_turning_point_idxs(z: np.ndarray):
    diff = np.diff(z)
    return np.argwhere(diff[1:] * diff[:-1] < 0)


class TestDivertorSilhouetteDesigner:
    _default_params: ClassVar = {
        "div_type": {"value": "SN", "unit": ""},
        "div_L2D_ib": {"value": 1.1, "unit": "m"},
        "div_L2D_ob": {"value": 1.45, "unit": "m"},
        "div_Ltarg_ib": {"value": 0.5, "unit": "m"},
        "div_Ltarg_ob": {"value": 0.5, "unit": "m"},
        "div_targ_angle_ib": {"value": 42, "unit": "degrees"},
        "div_targ_angle_ob": {"value": -25, "unit": "degrees"},
        "div_open": {"value": False, "unit": ""},
    }

    @classmethod
    def setup_class(cls):
        cls.eq = Equilibrium.from_eqdsk(Path(DATA, "eqref_OOB.json"), from_cocos=7)
        cls.separatrix = make_polygon(cls.eq.get_separatrix().xyz.T)
        _, cls.x_points = find_OX_points(cls.eq.x, cls.eq.z, cls.eq.psi())

    def setup_method(self):
        self.params = copy.deepcopy(self._default_params)
        self.wall = mock.MagicMock()
        self.wall.start_point().x = [5]
        self.wall.start_point().z = [self.x_points[0][1]]
        self.wall.end_point().x = [11]
        self.wall.end_point().z = [self.x_points[0][1]]

    def test_new_builder_sets_leg_lengths(self):
        self.params["div_L2D_ib"]["value"] = 5
        self.params["div_L2D_ob"]["value"] = 10

        designer = DivertorSilhouetteDesigner(self.params, self.eq, self.wall)

        assert designer.leg_length[LegPosition.INNER].value == 5
        assert designer.leg_length[LegPosition.OUTER].value == 10

    def test_targets_intersect_separatrix(self):
        designer = DivertorSilhouetteDesigner(self.params, self.eq, self.wall)

        divertor = designer.execute()

        for target in [divertor[1], divertor[3]]:
            assert signed_distance(target, self.separatrix) == pytest.approx(0)

    @pytest.mark.parametrize(("div_ltarg", "div_targ_angle"), [(1.5, 52)])
    def test_target_length_set_by_parameter(self, div_ltarg, div_targ_angle):
        params = self.params.copy()
        params["div_Ltarg_ib"]["value"] = div_ltarg / 2
        params["div_Ltarg_ob"]["value"] = div_ltarg / 2
        params["div_targ_angle_ib"]["value"] = div_targ_angle

        designer = DivertorSilhouetteDesigner(params, self.eq, self.wall)

        divertor = designer.execute()

        for target in [divertor[1], divertor[3]]:
            assert target.length == pytest.approx(div_ltarg / 2, rel=EPS)

    def test_dome_added_to_divertor(self):
        designer = DivertorSilhouetteDesigner(self.params, self.eq, self.wall)

        _, _, dome, _, _ = designer.execute()

        assert dome is not None

    def test_dome_intersects_targets(self):
        designer = DivertorSilhouetteDesigner(self.params, self.eq, self.wall)

        _, inner_target, dome, outer_target, _ = designer.execute()

        assert signed_distance(dome, inner_target) == 0
        assert signed_distance(dome, outer_target) == 0

    def test_dome_does_not_intersect_separatrix(self):
        designer = DivertorSilhouetteDesigner(self.params, self.eq, self.wall)

        _, _, dome, _, _ = designer.execute()

        assert signed_distance(dome, self.separatrix) < 0

    def test_SN_lower_dome_has_turning_point_below_x_point(self):
        designer = DivertorSilhouetteDesigner(self.params, self.eq, self.wall)
        x_points, _ = self.eq.get_OX_points()

        _, _, dome, _, _ = designer.execute()

        dome_coords = dome.discretise()
        turning_points = get_turning_point_idxs(dome_coords[2, :])
        assert len(turning_points) == 1
        assert dome_coords[2, turning_points[0]] < x_points[0].z

    def test_baffle_start_and_end_points_and_target_intersects(self):
        designer = DivertorSilhouetteDesigner(self.params, self.eq, self.wall)

        inner_baffle, inner_target, _, outer_target, outer_baffle = designer.execute()

        assert inner_baffle is not None
        assert inner_baffle.end_point()[0][0] == pytest.approx(min(designer.x_limits))
        assert outer_baffle is not None
        assert outer_baffle.start_point()[0][0] == pytest.approx(max(designer.x_limits))

        for target, baffle in [
            [inner_target, inner_baffle],
            [outer_target, outer_baffle],
        ]:
            assert signed_distance(target, baffle) == pytest.approx(0)
