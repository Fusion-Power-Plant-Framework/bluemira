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

from bluemira.builders.EUDEMO.pf_coils import make_coil_mapper
from bluemira.equilibria.coils import Coil
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import (
    PrincetonD,
    TaperedPictureFrame,
    TripleArc,
)
from bluemira.geometry.tools import make_polygon


class TestMakeCoilMapper:
    @classmethod
    def setup_class(cls):
        track = PrincetonD().create_shape()
        exclusion = BluemiraFace(
            make_polygon([[6, 9, 9, 6], [0, 0, 0, 0], [0, 0, 20, 20]], closed=True)
        )
        coil1 = Coil(4, 9, current=1, j_max=1)
        coil2 = Coil(9, -9, current=1, j_max=1)

    def test_simple(self):
        mapper = make_coil_mapper(track, [exclusion], [coil1, coil2])
