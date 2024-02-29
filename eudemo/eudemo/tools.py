# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
A collection of tools used in the EU-DEMO design.
"""

from typing import List

import numpy as np

from bluemira.base.components import PhysicalComponent
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import slice_shape


def get_inner_cut_point(breeding_blanket_xz, r_inner_cut):
    """
    Get the inner cut point of the breeding blanket geometry.
    """
    cut_plane = BluemiraPlane.from_3_points(
        [r_inner_cut, 0, 0], [r_inner_cut, 0, 1], [r_inner_cut, 1, 1]
    )
    # Get the first intersection with the vertical inner cut plane
    intersections = slice_shape(breeding_blanket_xz.boundary[0], cut_plane)
    intersections = intersections[intersections[:, -1] > 0.0]
    return sorted(intersections, key=lambda x: x[-1])[0]


def make_2d_view_components(
    view: str, azimuthal_angle: float, components: List[PhysicalComponent]
) -> List[PhysicalComponent]:
    """
    Make a 2-D slice of a list of 3-D components

    Parameters
    ----------
    view:
        View to make the components for. From ['xz', 'xy']
    azimuthal_angle:
        Angle at which to cut the x-z plane [degree]. Has no effect for the
        x-y plane
    components:
        List of PhysicalComponents to take slices from

    Returns
    -------
    List of PhysicalComponents in the desired view plane
    """
    azimuthal_angle = np.deg2rad(azimuthal_angle)
    if view == "xz":
        plane = BluemiraPlane.from_3_points(
            [0, 0, 0], [0, 0, 1], [np.cos(azimuthal_angle), np.sin(azimuthal_angle), 0]
        )
    elif view == "xy":
        plane = BluemiraPlane.from_3_points([0, 0, 0], [0, 1, 0], [1, 0, 0])
    else:
        raise ValueError(f"Unrecognised view: {view}, please choose from ['xz', 'xy']")

    view_comps = []
    for comp in components:
        comp_slices = []
        pieces = slice_shape(comp.shape, plane)
        # TODO: slice_shape is unreliable for complex shapes...

        if pieces:
            for i, piece in enumerate(pieces):
                face = BluemiraFace(piece)
                if azimuthal_angle != 0:
                    face.rotate(degree=-np.rad2deg(azimuthal_angle))
                new_comp = PhysicalComponent(
                    f"{comp.name} {i}", face, material=comp.material
                )
                comp_slices.append(new_comp)
            view_comps.append(comp_slices)
    return view_comps
