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

"""
from typing import Tuple

from EUDEMO_builders.ivc.tools import _cut_vessel_shape, _make_clearance_face

from bluemira.base.designer import Designer
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.wire import BluemiraWire


class PlasmaFaceDesigner(Designer[BluemiraFace]):

    params_cls = None

    def __init__(self, ivc_boundary, wall_boundary, divertor_silhouette, rm_clearance):
        super().__init__()
        self.ivc_boundary = ivc_boundary
        self.wall_boundary = wall_boundary
        self.divertor_silhouette = divertor_silhouette
        self.rm_clearance = rm_clearance

    def run(self) -> Tuple[BluemiraFace]:
        plasma_facing_wire = BluemiraWire(
            [self.wall_boundary, *self.divertor_silhouette]
        )

        in_vessel_face = BluemiraFace([self.ivc_boundary, plasma_facing_wire])

        # Cut a clearance between the blankets and divertor - getting two
        # new faces
        vessel_bbox = in_vessel_face.bounding_box
        rm_clearance_face = _make_clearance_face(
            vessel_bbox.x_min,
            vessel_bbox.x_max,
            self.wall_boundary.bounding_box.z_min,
            self.rm_clearance,
        )

        blanket_face, divertor_face = _cut_vessel_shape(
            in_vessel_face, rm_clearance_face
        )
        return blanket_face, divertor_face
