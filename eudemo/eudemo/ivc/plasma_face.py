# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
Plasma Face Designer
"""
from dataclasses import dataclass
from typing import Dict, Tuple, Union

import numpy as np

from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import boolean_cut, make_polygon
from bluemira.geometry.wire import BluemiraWire


@dataclass
class PlasmaFaceDesignerParams(ParameterFrame):
    """Parameters for running the :class:`.PlasmaFaceDesigner`."""

    div_type: Parameter[str]
    c_rm: Parameter[float]


class PlasmaFaceDesigner(Designer[Tuple[BluemiraFace, BluemiraFace]]):
    """
    Designs the Plasma facing keep out zones

    Parameters
    ----------
    params:
        Plasma face designer parameters
    ivc_boundary:
        IVC boundary keep out zone
    wall_boundary:
        wall boundary keep out zone (cut at divertor)
    divertor_silhouette:
        divertor keep out zone
    """

    param_cls = PlasmaFaceDesignerParams
    params: PlasmaFaceDesignerParams

    def __init__(
        self,
        params: Union[ParameterFrame, Dict],
        ivc_boundary: BluemiraWire,
        wall_boundary: BluemiraWire,
        divertor_silhouette: Tuple[BluemiraWire, ...],
    ):
        super().__init__(params)
        if self.params.div_type.value == "DN":
            raise NotImplementedError("Double Null divertor not implemented")
        self.ivc_boundary = ivc_boundary
        self.wall_boundary = wall_boundary
        self.divertor_silhouette = divertor_silhouette

    def run(self) -> Tuple[BluemiraFace, BluemiraFace]:
        """
        Run method for PlasmaFaceDesigner
        """
        # For double null this and self.divertor_silhouette need a structure
        # to accommodate two divertors
        plasma_facing_wire = BluemiraWire(
            [self.wall_boundary, *self.divertor_silhouette]
        )

        in_vessel_face = BluemiraFace([self.ivc_boundary, plasma_facing_wire])

        # Cut a clearance between the blankets and divertor - getting two
        # new faces
        vessel_bbox = in_vessel_face.bounding_box
        # The minimum z-value of the wall boundary. The boundary should
        # be open at the lower end and the start and end points of the
        # wire should be at the same z. But take the minimum z value of
        # the start and end points.
        # Note we do not use bounding_box here due to a bug: 34228d3
        min_z = min(self.wall_boundary.start_point().z, self.wall_boundary.end_point().z)
        rm_clearance_face = _make_clearance_face(
            vessel_bbox.x_min,
            vessel_bbox.x_max,
            min_z,
            self.params.c_rm.value,
        )

        return _cut_vessel_shape(in_vessel_face, rm_clearance_face)


def _make_clearance_face(
    x_min: float, x_max: float, z: float, thickness: float
) -> BluemiraFace:
    """
    Makes a rectangular face in xz with the given thickness in z.

    The face is intended to be used to cut a remote maintenance
    clearance between blankets and divertor.
    """
    x_coords = np.zeros(4)
    x_coords[:2] = x_min
    x_coords[2:] = x_max

    y_coords = np.zeros(4)

    z_coords = np.zeros(4)
    z_coords[[0, 3]] = z + thickness / 2
    z_coords[[1, 2]] = z - thickness / 2

    return BluemiraFace(make_polygon([x_coords, y_coords, z_coords], closed=True))


def _cut_vessel_shape(
    in_vessel_face: BluemiraFace, rm_clearance_face: BluemiraFace
) -> Tuple[BluemiraFace, BluemiraFace]:
    """
    Cut a remote maintenance clearance into the given vessel shape.
    """
    pieces = boolean_cut(in_vessel_face, [rm_clearance_face])
    blanket_face = pieces[np.argmax([p.center_of_mass[2] for p in pieces])]
    divertor_face = pieces[np.argmin([p.center_of_mass[2] for p in pieces])]
    return blanket_face, divertor_face
