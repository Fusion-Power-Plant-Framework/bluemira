# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
The API for CGAL.
"""

from bluemira.codes.cgal_ext._guard import cgal_available
from bluemira.codes.cgal_ext.collision_detection import (
    do_polys_collide,
    tri_mesh_to_cgal_mesh,
)

__all__ = [
    "cgal_available",
    "do_polys_collide",
    "tri_mesh_to_cgal_mesh",
]
