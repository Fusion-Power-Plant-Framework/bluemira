import os
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.geomtools import circle_seg
from BLUEPRINT.cad.model import CADModel
from BLUEPRINT.cad.component import ComponentCAD
from BLUEPRINT.cad.cadtools import (
    make_face,
    revolve,
)
from bluemira.base.file import get_bluemira_path


class DummyCAD(ComponentCAD):
    def __init__(self):
        ComponentCAD.__init__(
            self,
            "Test Metadata",
        )

    def build(self, **kwargs):
        circles = []
        centre = (4.0, 0.0)
        radius = 1.0
        circle_x, circle_z = circle_seg(radius, centre, angle=360, npoints=50)
        circles.append(Loop(x=circle_x, y=None, z=circle_z))
        centre = (0.5, 0.0)
        radius = 0.25
        circle_x, circle_z = circle_seg(radius, centre, angle=360, npoints=50)
        circles.append(Loop(x=circle_x, y=None, z=circle_z))

        names = ["outer", "inner"]
        for i_circle in range(0, len(circles)):
            face = make_face(circles[i_circle])
            torus = revolve(face, None, 360)
            self.add_shape(torus, name=names[i_circle])


class TestCADModel:
    @classmethod
    def setup_class(cls):
        cls.cad = DummyCAD()
        cls.model = CADModel(1)
        cls.model.add_part(cls.cad)

    def test_stp_assembly_metadata(self):
        # Generate a STP file with metadata
        test_file = "test_metadata.STP"
        self.model.save_as_STEP_assembly(test_file)

        # Fetch comparison file
        data_dir = "BLUEPRINT/cad/test_data"
        compare_path = get_bluemira_path(data_dir, subfolder="tests")
        compare_file = os.sep.join([compare_path, "dummy_model_with_metadata.STP"])

        # Load filelines
        f_test = open(test_file, "r")
        lines_test = f_test.readlines()
        f_test.close()
        f_compare = open(compare_file, "r")
        lines_compare = f_compare.readlines()
        f_compare.close()

        # Compare files, skipping timestamp and colour metadata
        # (The order of colours appears to arbitrarily switch between writes)
        n_lines = len(lines_test)
        assert n_lines == len(lines_compare)
        skip_lines = [3]  # timestamp
        skip_lines.extend(range(6540, 6563))  # colour metadata
        for i_line in range(0, n_lines):
            if i_line in skip_lines:
                continue
            assert lines_test[i_line] == lines_compare[i_line]

        # Clean up
        os.remove(test_file)
