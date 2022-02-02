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

import pytest

from bluemira.base.error import BuilderError
from bluemira.base.file import get_bluemira_path
from bluemira.builders.EUDEMO.first_wall import DivertorBuilder
from bluemira.builders.EUDEMO.first_wall.divertor import Leg
from bluemira.equilibria import Equilibrium
from bluemira.geometry.tools import make_polygon, signed_distance

DATA = get_bluemira_path("bluemira/equilibria/test_data", subfolder="tests")


class TestDivertorBuilder:

    _default_params = {
        "div_L2D_ib": (1.1, "Input"),
        "div_L2D_ob": (1.45, "Input"),
        "div_Ltarg": (0.5, "Input"),
    }

    @classmethod
    def setup_class(cls):
        cls.eq = Equilibrium.from_eqdsk(os.path.join(DATA, "eqref_OOB.json"))

    def setup_method(self):
        self.params = copy.deepcopy(self._default_params)

    def test_no_BuilderError_on_init_given_valid_params(self):
        try:
            DivertorBuilder(self.params, {"name": "some_name"}, self.eq)
        except BuilderError:
            pytest.fail(str(BuilderError))

    @pytest.mark.parametrize("required_param", DivertorBuilder._required_params)
    def test_BuilderError_given_required_param_missing(self, required_param):
        self.params.pop(required_param)

        with pytest.raises(BuilderError):
            DivertorBuilder(self.params, {"name": "some_name"}, self.eq)

    def test_new_builder_sets_leg_lengths(self):
        self.params.update({"div_L2D_ib": 5, "div_L2D_ob": 10})

        builder = DivertorBuilder(self.params, {"name": "some_name"}, self.eq)

        assert builder.leg_length[Leg.INNER] == 5
        assert builder.leg_length[Leg.OUTER] == 10

    def test_targets_intersect_separatrix(self):
        builder = DivertorBuilder(self.params, {"name": "some_name"}, self.eq)
        separatrix = make_polygon(self.eq.get_separatrix().xyz.T)

        divertor = builder(self._default_params)

        for leg in [Leg.INNER, Leg.OUTER]:
            target = divertor.get_component(f"target {leg}")
            assert signed_distance(target.shape, separatrix) == 0

    def test_div_Ltarg_sets_target_length(self):
        self.params.update({"div_Ltarg": 1.5})
        builder = DivertorBuilder(self.params, {"name": "some_name"}, self.eq)

        divertor = builder(self.params)

        for leg in [Leg.INNER, Leg.OUTER]:
            target = divertor.get_component(f"target {leg}")
            assert target.shape.length == 1.5
