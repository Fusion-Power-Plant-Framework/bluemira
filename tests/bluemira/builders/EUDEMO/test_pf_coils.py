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

import numpy as np
import pytest

from bluemira.builders.EUDEMO.pf_coils import make_coil_mapper
from bluemira.equilibria.coils import Coil
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import PrincetonD, TripleArc
from bluemira.geometry.tools import boolean_cut, make_polygon


class TestMakeCoilMapper:
    tracks = [
        PrincetonD(
            {"x1": {"value": 4}, "x2": {"value": 14}, "dz": {"value": 0}}
        ).create_shape(label="PrincetonD"),
        # Waiting on #747
        # TaperedPictureFrame(
        #     {
        #         "x1": {"value": 4, "upper_bound": 5},
        #         "x2": {"value": 5, "upper_bound": 6},
        #         "x3": {"value": 11.5, "upper_bound": 12},
        #         "ri": {"value": 0},
        #         "ro": {"value": 1},
        #         "z1": {"value": 8},
        #         "z2": {"value": 9, "upper_bound": 10},
        #     }
        # ).create_shape(label="TPFrame"),
        TripleArc().create_shape(label="TripleArc"),
    ]

    @classmethod
    def setup_class(cls):

        exclusion1 = BluemiraFace(
            make_polygon([[6, 9, 9, 6], [0, 0, 0, 0], [0, 0, 20, 20]], closed=True)
        )
        exclusion2 = BluemiraFace(
            make_polygon([[9, 20, 20, 9], [0, 0, 0, 0], [-1, -1, 1, 1]], closed=True)
        )
        cls.exclusions = [exclusion1, exclusion2]

        cls.coils = [
            Coil(4, 9, current=1e6, j_max=1),
            Coil(9, -9, current=1e6, j_max=1),
            Coil(12, 0.1, current=1e6, j_max=1),
            Coil(6, -10, current=1e6, j_max=1),
        ]

    @pytest.mark.parametrize("track", tracks)
    def test_cuts(self, track):
        segments = boolean_cut(track, self.exclusions)
        actual_length = sum([seg.length for seg in segments])
        mapper = make_coil_mapper(track, self.exclusions, self.coils)
        interp_length = sum([tool.geometry.length for tool in mapper.interpolators])
        assert np.isclose(actual_length, interp_length, rtol=1e-2)

    @pytest.mark.parametrize("track", tracks)
    def test_simple(self, track):
        mapper = make_coil_mapper(track, self.exclusions, self.coils)
        assert len(mapper.interpolators) == len(self.coils)
