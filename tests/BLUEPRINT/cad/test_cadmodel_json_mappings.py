import filecmp
import os

from bluemira.base.file import get_bluemira_path
from BLUEPRINT.cad.cadtools import make_face, revolve
from BLUEPRINT.cad.component import ComponentCAD
from BLUEPRINT.cad.model import CADModel
from BLUEPRINT.geometry.geomtools import circle_seg, make_box_xz
from BLUEPRINT.geometry.loop import Loop


class DummyCAD(ComponentCAD):
    def __init__(self, name, shapes_in):
        self.shapes = shapes_in
        ComponentCAD.__init__(
            self,
            name,
        )

    def build(self, **kwargs):
        for shape_name, loop in self.shapes.items():
            face = make_face(loop)
            torus = revolve(face, None, 360)
            self.add_shape(torus, name=shape_name)


class TestCADModel:
    @classmethod
    def setup_class(cls):
        cls.model = CADModel(1)

        circles = {}
        centre = (4.0, 0.0)
        radius = 1.0
        circle_x, circle_z = circle_seg(radius, centre, angle=360, npoints=50)
        circles["outer"] = Loop(x=circle_x, y=None, z=circle_z)
        centre = (0.5, 0.0)
        radius = 0.25
        circle_x, circle_z = circle_seg(radius, centre, angle=360, npoints=50)
        circles["inner"] = Loop(x=circle_x, y=None, z=circle_z)
        cad = DummyCAD("Test Metadata", circles)
        cls.model.add_part(cad)

        squares = {}
        squares["lower box"] = make_box_xz(6.0, 6.5, -1.5, -0.5)
        squares["upper box"] = make_box_xz(6.0, 6.5, 0.5, 1.5)
        cad = DummyCAD("More Metadata", squares)
        cls.model.add_part(cad)

    def test_save_component_names(self):
        # Generate a STP file with metadata
        test_file = "test_components.json"
        self.model.save_component_names_as_json(test_file)

        # Fetch comparison file
        data_dir = "BLUEPRINT/cad/test_data"
        compare_path = get_bluemira_path(data_dir, subfolder="tests")
        compare_file = os.sep.join([compare_path, "dummy_model_components.json"])

        # Compare
        assert filecmp.cmp(test_file, compare_file)

        # Clean up
        os.remove(test_file)
