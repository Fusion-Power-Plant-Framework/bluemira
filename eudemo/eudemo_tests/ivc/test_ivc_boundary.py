# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Test ivc boundary designer.
"""

from typing import ClassVar

import pytest

from bluemira.base.error import DesignError
from bluemira.geometry.parameterisations import PictureFrame
from bluemira.geometry.tools import make_circle, signed_distance
from eudemo.ivc import IVCBoundaryDesigner


class TestIVCBoundaryDesigner:
    picture_frame: ClassVar = PictureFrame({
        "ro": {"value": 6},
        "ri": {"value": 3},
    }).create_shape()
    params: ClassVar = {
        "tk_bb_ib": {"value": 0.8, "unit": "m"},
        "tk_bb_ob": {"value": 1.1, "unit": "m"},
        "ib_offset_angle": {"value": 45, "unit": "degree"},
        "ob_offset_angle": {"value": 175, "unit": "degree"},
    }

    def test_DesignError_given_wall_shape_not_closed(self):
        wall_shape = make_circle(end_angle=180)

        with pytest.raises(DesignError):
            IVCBoundaryDesigner(self.params, wall_shape)

    def test_design_returns_boundary_that_does_not_intersect_wire(self):
        designer = IVCBoundaryDesigner(self.params, self.picture_frame)

        wire = designer.execute()

        assert signed_distance(wire, self.picture_frame) < 0
