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
Tests for divertor builder classes
"""
import copy
import os

import numpy as np
import pytest

from bluemira.base.error import BuilderError
from bluemira.base.file import get_bluemira_path
from bluemira.builders.EUDEMO.ivc import DivertorSilhouetteBuilder
from bluemira.builders.EUDEMO.ivc.divertor import LegPosition
from bluemira.equilibria import Equilibrium
from bluemira.equilibria.find import find_OX_points
from bluemira.geometry.tools import make_polygon, signed_distance

DATA = get_bluemira_path("equilibria/test_data", subfolder="tests")


def get_turning_point_idxs(z: np.ndarray):
    diff = np.diff(z)
    return np.argwhere(diff[1:] * diff[:-1] < 0)


class TestDivertorSilhouetteBuilder:

    _default_params = {
        "div_L2D_ib": (1.1, "Input"),
        "div_L2D_ob": (1.45, "Input"),
        "div_Ltarg": (0.5, "Input"),
        "div_open": (False, "Input"),
    }
    targets = [
        DivertorSilhouetteBuilder.COMPONENT_INNER_TARGET,
        DivertorSilhouetteBuilder.COMPONENT_OUTER_TARGET,
    ]

    @classmethod
    def setup_class(cls):
        cls.eq = Equilibrium.from_eqdsk(os.path.join(DATA, "eqref_OOB.json"))
        cls.separatrix = make_polygon(cls.eq.get_separatrix().xyz.T)
        _, cls.x_points = find_OX_points(cls.eq.x, cls.eq.z, cls.eq.psi())

    def setup_method(self):
        self.params = copy.deepcopy(self._default_params)
        self.x_lims = [5, 11]
        self.z_lims = [self.x_points[0][1], self.x_points[0][1]]

    def test_no_BuilderError_on_init_given_valid_params(self):
        try:
            DivertorSilhouetteBuilder(
                self.params,
                {"name": "some_name"},
                self.eq,
                self.x_lims,
                self.z_lims,
            )
        except BuilderError:
            pytest.fail(str(BuilderError))

    @pytest.mark.parametrize(
        "required_param", DivertorSilhouetteBuilder._required_params
    )
    def test_BuilderError_given_required_param_missing(self, required_param):
        self.params.pop(required_param)

        with pytest.raises(BuilderError):
            DivertorSilhouetteBuilder(
                self.params, {"name": "some_name"}, self.eq, self.x_lims, self.z_lims
            )

    def test_new_builder_sets_leg_lengths(self):
        self.params.update({"div_L2D_ib": 5, "div_L2D_ob": 10})

        builder = DivertorSilhouetteBuilder(
            self.params, {"name": "some_name"}, self.eq, self.x_lims, self.z_lims
        )

        assert builder.leg_length[LegPosition.INNER] == 5
        assert builder.leg_length[LegPosition.OUTER] == 10

    def test_targets_intersect_separatrix(self):
        builder = DivertorSilhouetteBuilder(
            self.params, {"name": "some_name"}, self.eq, self.x_lims, self.z_lims
        )

        divertor = builder()

        for leg in self.targets:
            target = divertor.get_component(leg)
            assert signed_distance(target.shape, self.separatrix) == 0

    def test_target_length_set_by_parameter(self):
        self.params.update({"div_Ltarg": 1.5})
        builder = DivertorSilhouetteBuilder(
            self.params, {"name": "some_name"}, self.eq, self.x_lims, self.z_lims
        )

        divertor = builder()

        for leg in self.targets:
            target = divertor.get_component(leg)
            assert target.shape.length == 1.5

    def test_dome_added_to_divertor(self):
        builder = DivertorSilhouetteBuilder(
            self.params, {"name": "some_name"}, self.eq, self.x_lims, self.z_lims
        )

        divertor = builder()

        assert (
            divertor.get_component(DivertorSilhouetteBuilder.COMPONENT_DOME) is not None
        )

    def test_dome_intersects_targets(self):
        builder = DivertorSilhouetteBuilder(
            self.params, {"name": "some_name"}, self.eq, self.x_lims, self.z_lims
        )

        divertor = builder()

        dome = divertor.get_component(DivertorSilhouetteBuilder.COMPONENT_DOME)
        targets = [divertor.get_component(leg) for leg in self.targets]
        assert signed_distance(dome.shape, targets[0].shape) == 0
        assert signed_distance(dome.shape, targets[1].shape) == 0

    def test_dome_does_not_intersect_separatrix(self):
        builder = DivertorSilhouetteBuilder(
            self.params, {"name": "some_name"}, self.eq, self.x_lims, self.z_lims
        )

        divertor = builder()

        dome = divertor.get_component(DivertorSilhouetteBuilder.COMPONENT_DOME)
        assert signed_distance(dome.shape, self.separatrix) < 0

    def test_SN_lower_dome_has_turning_point_below_x_point(self):
        builder = DivertorSilhouetteBuilder(
            self.params, {"name": "some_name"}, self.eq, self.x_lims, self.z_lims
        )
        x_points, _ = self.eq.get_OX_points()

        divertor = builder()

        dome_coords = divertor.get_component(
            DivertorSilhouetteBuilder.COMPONENT_DOME
        ).shape.discretize()
        turning_points = get_turning_point_idxs(dome_coords[2, :])
        assert len(turning_points) == 1
        assert dome_coords[2, turning_points[0]] < x_points[0].z

    def test_inner_baffle_has_end_at_lower_x_limit(self):
        builder = DivertorSilhouetteBuilder(
            self.params, {"name": "some_name"}, self.eq, self.x_lims, self.z_lims
        )

        divertor = builder()

        inner_baffle = divertor.get_component(
            DivertorSilhouetteBuilder.COMPONENT_INNER_BAFFLE
        )
        assert inner_baffle is not None
        assert inner_baffle.shape.start_point()[0] == min(self.x_lims)

    def test_outer_baffle_has_end_at_upper_x_limit(self):
        builder = DivertorSilhouetteBuilder(
            self.params, {"name": "some_name"}, self.eq, self.x_lims, self.z_lims
        )

        divertor = builder()

        outer_baffle = divertor.get_component(
            DivertorSilhouetteBuilder.COMPONENT_OUTER_BAFFLE
        )
        assert outer_baffle is not None
        assert outer_baffle.shape.end_point()[0] == max(self.x_lims)

    @pytest.mark.parametrize("side", ("INNER", "OUTER"))
    def test_baffle_and_target_intersect(self, side):
        builder = DivertorSilhouetteBuilder(
            self.params, {"name": "some_name"}, self.eq, self.x_lims, self.z_lims
        )

        divertor = builder()

        target = divertor.get_component(
            getattr(DivertorSilhouetteBuilder, f"COMPONENT_{side}_TARGET")
        )
        baffle = divertor.get_component(
            getattr(DivertorSilhouetteBuilder, f"COMPONENT_{side}_BAFFLE")
        )
        assert signed_distance(target.shape, baffle.shape) == 0

    def test_setting_xz_limits_after_init_sets_start_and_end_points(self):
        builder = DivertorSilhouetteBuilder(
            self.params, {"name": "some_name"}, self.eq, [], []
        )

        builder.x_limits = self.x_lims
        builder.z_limits = self.z_lims
        divertor = builder()

        inner_baffle = divertor.get_component(
            DivertorSilhouetteBuilder.COMPONENT_INNER_BAFFLE
        )
        assert inner_baffle is not None
        assert inner_baffle.shape.start_point()[0] == min(self.x_lims)
        outer_baffle = divertor.get_component(
            DivertorSilhouetteBuilder.COMPONENT_OUTER_BAFFLE
        )
        assert outer_baffle is not None
        assert outer_baffle.shape.end_point()[0] == max(self.x_lims)

    @pytest.mark.parametrize("x_lims", [[], None, ()])
    def test_BuilderError_on_call_given_x_limits_empty(self, x_lims):
        builder = DivertorSilhouetteBuilder(
            self.params, {"name": "some_name"}, self.eq, x_lims, self.z_lims
        )

        with pytest.raises(BuilderError):
            builder()
