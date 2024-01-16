# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Test PlasmaFaceDesigner
"""

from typing import ClassVar

import pytest

from bluemira.geometry.tools import make_polygon
from eudemo.ivc import PlasmaFaceDesigner


class TestPlasmaFaceDesigner:
    _params: ClassVar = {
        "div_type": {"value": "SN", "unit": ""},
        "c_rm": {"value": 0.02, "unit": "m"},
        "lower_port_angle": {"value": 0, "unit": "degree"},
    }

    @classmethod
    def setup_class(cls):
        cls.wall_boundary = make_polygon([
            [1, 0, -10],
            [1, 0, 10],
            [11, 0, 10],
            [11, 0, -10],
        ])
        cls.divertor_silhouette = (
            make_polygon([[1, 0, -10], [6, 0, -15]]),
            make_polygon([[11, 0, -10], [6, 0, -15]]),
        )
        cls.ivc_boundary = make_polygon(
            [[0, 0, -16], [0, 0, 11], [12, 0, 11], [12, 0, -16]], closed=True
        )

    def test_wall_boundary_is_cut_below_x_point_in_z_axis(self):
        designer = PlasmaFaceDesigner(
            self._params, self.ivc_boundary, self.wall_boundary, self.divertor_silhouette
        )

        (blanket_face, divertor_face, div_wall_join_pt) = designer.execute()
        # U shape
        bf_area = 20 * 2 + 1 * 12
        assert pytest.approx(blanket_face.area) == bf_area - designer.params.c_rm.value
        # Rectangle with a triangle cut out
        # cut removed a slice of triangle with rm section
        div_area = 6 * 12 - 10 * 5 * 0.5
        extra_triangles = 0.00005 * 2
        assert (
            pytest.approx(divertor_face.area)
            == div_area - designer.params.c_rm.value - extra_triangles
        )
