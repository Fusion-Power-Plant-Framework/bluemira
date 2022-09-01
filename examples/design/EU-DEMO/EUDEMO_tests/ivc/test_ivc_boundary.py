import pytest
from EUDEMO_builders.ivc import IVCBoundaryDesigner

from bluemira.base.error import DesignError
from bluemira.geometry.parameterisations import PictureFrame
from bluemira.geometry.tools import make_circle, signed_distance


class TestIVCBoundaryDesigner:

    picture_frame = PictureFrame({"ro": {"value": 6}, "ri": {"value": 3}}).create_shape()
    params = {
        "tk_bb_ib": {"name": "tk_bb_ib", "value": 0.8},
        "tk_bb_ob": {"name": "tk_bb_ob", "value": 1.1},
        "ib_offset_angle": {"name": "ib_offset_angle", "value": 45},
        "ob_offset_angle": {"name": "ob_offset_angle", "value": 175},
    }

    def test_DesignError_given_wall_shape_not_closed(self):
        wall_shape = make_circle(end_angle=180)

        with pytest.raises(DesignError):
            IVCBoundaryDesigner(self.params, wall_shape, 0.0)

    def test_design_returns_boundary_that_does_not_intersect_wire(self):
        designer = IVCBoundaryDesigner(self.params, self.picture_frame, -4)

        wire = designer.execute()

        assert signed_distance(wire, self.picture_frame) < 0
