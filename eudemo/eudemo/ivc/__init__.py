# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Module containing builders for the EUDEMO first wall components
"""

from dataclasses import dataclass
from typing import Dict

from bluemira.base.designer import run_designer
from bluemira.base.parameter_frame import ParameterFrame
from bluemira.equilibria import find_OX_points
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.wire import BluemiraWire
from eudemo.ivc.divertor_silhouette import DivertorSilhouetteDesigner
from eudemo.ivc.ivc_boundary import IVCBoundaryDesigner
from eudemo.ivc.plasma_face import PlasmaFaceDesigner
from eudemo.ivc.tools import cut_wall_below_x_point
from eudemo.ivc.wall_silhouette import WallSilhouetteDesigner


@dataclass
class IVCShapes:
    """Collection of geometries used to design/build in-vessel components."""

    blanket_face: BluemiraFace
    divertor_face: BluemiraFace
    outer_boundary: BluemiraWire
    inner_boundary: BluemiraWire


def design_ivc(
    params: ParameterFrame, build_config: Dict, equilibrium: Equilibrium
) -> IVCShapes:
    """
    Run the IVC component designers in sequence.

    Returns
    -------
    blanket_face: BluemiraFace
        Face of the blanket, does not include anything below the
        divertor.
    divertor_face: BluemiraFace
        A face with the shape of divertor in xz.
    ive_boundary: BluemiraWire
        A wire defining the xz boundary of the in-vessel components.
    """
    wall_boundary = run_designer(
        WallSilhouetteDesigner,
        params,
        build_config["Wall silhouette"],
        equilibrium=equilibrium,
    ).create_shape(label="wall")
    _, x_points = find_OX_points(equilibrium.x, equilibrium.z, equilibrium.psi())
    cut_wall_boundary = cut_wall_below_x_point(wall_boundary, x_points[0].z)
    divertor_shapes = DivertorSilhouetteDesigner(
        params,
        equilibrium=equilibrium,
        wall=cut_wall_boundary,
    ).execute()
    ivc_boundary = IVCBoundaryDesigner(params, wall_shape=wall_boundary).execute()
    plasma_face, divertor_face = PlasmaFaceDesigner(
        params,
        ivc_boundary=ivc_boundary,
        wall_boundary=cut_wall_boundary,
        divertor_silhouette=divertor_shapes,
    ).execute()

    # We have already cut the wall boundary once below the x-point in
    # order to generate our blanket face. However, we then cut that
    # blanket face using some thickness (remote maintenance clearance).
    # We want the boundary wire and face to start and end at the same
    # place, so we cut the wire again here.
    wall_boundary = cut_wall_below_x_point(wall_boundary, plasma_face.bounding_box.z_min)
    return IVCShapes(
        blanket_face=plasma_face,
        divertor_face=divertor_face,
        outer_boundary=ivc_boundary,
        inner_boundary=wall_boundary,
    )
