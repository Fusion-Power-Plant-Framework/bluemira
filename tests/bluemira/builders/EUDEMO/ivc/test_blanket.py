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
import pytest

from bluemira.base.error import BuilderError
from bluemira.builders.EUDEMO.ivc.blanket import BlanketThicknessBuilder
from bluemira.geometry.parameterisations import PictureFrame
from bluemira.geometry.tools import make_circle, signed_distance


class TestBlanketThicknessBuilder:

    picture_frame = PictureFrame({"ro": {"value": 6}, "ri": {"value": 3}}).create_shape()
    build_config = {
        "name": "Blanket",
        "runmode": "run",
    }
    params = {
        "tk_bb_ib": (0.8, "Input"),
        "tk_bb_ob": (1.1, "Input"),
    }

    def test_BuilderError_given_wall_shape_not_closed(self):
        wall_shape = make_circle(end_angle=180)

        with pytest.raises(BuilderError):
            BlanketThicknessBuilder({}, {}, wall_shape, 0.0)

    def test_build_returns_component_containing_boundary_in_xz(self):
        wall_shape = make_circle(axis=(0, 1, 0))
        builder = BlanketThicknessBuilder(self.params, self.build_config, wall_shape, 0)

        component = builder.build()

        assert component.is_root
        xz = component.get_component("xz")
        assert xz is not None
        assert xz.depth == 1
        boundary_xz = xz.get_component("blanket_boundary")
        assert boundary_xz is not None
        assert boundary_xz.depth == 2

    def test_build_returns_boundary_that_does_not_intersect_wire(self):
        builder = BlanketThicknessBuilder(
            self.params, self.build_config, self.picture_frame, -4
        )

        component = builder.build()

        shape = component.get_component("blanket_boundary").shape
        assert signed_distance(shape, self.picture_frame) < 0
