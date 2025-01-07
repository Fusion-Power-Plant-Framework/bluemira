# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from pathlib import Path
from unittest.mock import patch

import freecad  # noqa: F401
import Part
import numpy as np
import pytest
from FreeCAD import Base, closeDocument, newDocument
from scipy.special import ellipe

import bluemira.codes._freecadapi as cadapi
from bluemira.base.constants import EPS
from bluemira.codes.error import FreeCADError
from bluemira.geometry.constants import D_TOLERANCE
from tests._helpers import skipif_import_error


class TestFreecadapi:
    @classmethod
    def setup_class(cls):
        cls.square_points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
        ]
        cls.closed_square_points = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (1.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 0.0),
        ]

    @staticmethod
    def offsetter(wire):
        return cadapi.offset_wire(
            wire,
            0.05,
            join="intersect",
            open_wire=False,
        )

    def test_multi_offset_wire(self):
        circ = cadapi.make_circle(10)
        wire1 = self.offsetter(circ)
        wire2 = self.offsetter(wire1)

        assert circ.Length < wire1.Length < wire2.Length

    def test_multi_offset_wire_without_arranged_edges(self):
        """
        FreeCAD Topological naming bug

        As of 08/2022 FreeCAD has a ordering/naming bug as detailed in #1347.
        The result is that some operations do not work as expected (such as offset_wire)
        Some operations when repeated also raise errors such as DisjointedFace errors.
        arrange_edges tries to reorder the internal edges of a wire to attempt to side
        step the issue.

        FreeCAD aims to fix this for v1 which is due for release in 2023.
        When that happens our work arounds can be removed (including this test)
        """

        circ = cadapi.make_circle(10)
        with patch(
            "bluemira.codes._freecadapi.arrange_edges",
            new=lambda a, b: b,  # noqa: ARG005
        ):
            wire1 = self.offsetter(circ)
            wire2 = self.offsetter(wire1)

        assert circ.Length < wire1.Length
        # these two should break in future, this mean the topo naming may be fixed
        assert circ.Length == wire2.Length
        assert wire1.Length > wire2.Length

    def test_fail_vector_to_numpy(self):
        with pytest.raises(TypeError):
            arr = cadapi.vector_to_numpy(self.square_points)

    def test_fail_points_to_numpy(self):
        with pytest.raises(TypeError):
            arr = cadapi.point_to_numpy(self.square_points)

    def test_single_vector_to_numpy(self):
        inp = np.array((1.0, 0.5, 2.0))
        vector = Base.Vector(inp)
        arr = cadapi.vector_to_numpy(vector)
        comparison = arr == inp
        assert comparison.all()

    def test_vector_to_numpy(self):
        vectors = [Base.Vector(v) for v in self.square_points]
        arr = cadapi.vector_to_numpy(vectors)
        comparison = arr == np.array(self.square_points)
        assert comparison.all()

    def test_point_to_numpy(self):
        vectors = [Part.Point(Base.Vector(v)) for v in self.square_points]
        arr = cadapi.point_to_numpy(vectors)
        comparison = arr == np.array(self.square_points)
        assert comparison.all()

    def test_vertex_to_numpy(self):
        vertexes = [Part.Vertex(Base.Vector(v)) for v in self.square_points]
        arr = cadapi.vertex_to_numpy(vertexes)
        comparison = arr == np.array(self.square_points)
        assert comparison.all()

    def test_make_polygon(self):
        # open wire
        open_wire: Part.Wire = cadapi.make_polygon(self.square_points)
        vertexes = open_wire.Vertexes
        assert len(vertexes) == 4
        assert len(open_wire.Edges) == 3
        arr = cadapi.vertex_to_numpy(vertexes)
        comparison = arr == np.array(self.square_points)
        assert comparison.all()
        assert not open_wire.isClosed()
        # closed wire
        closed_wire: Part.Wire = cadapi.make_polygon(self.closed_square_points)
        vertexes = closed_wire.Vertexes
        assert len(vertexes) == 4
        assert len(closed_wire.Edges) == 4
        arr = cadapi.vertex_to_numpy(vertexes)
        comparison = arr == np.array(self.square_points)
        assert comparison.all()
        assert closed_wire.isClosed()

    def test_make_bezier(self):
        bezier: Part.Wire = cadapi.make_bezier(self.square_points)
        curve = bezier.OrderedEdges[0].Curve
        assert type(curve) is Part.BezierCurve

    def test_interpolate_bspline(self):
        pntslist = self.square_points
        bspline: Part.Wire = cadapi.interpolate_bspline(pntslist)
        curve = bspline.OrderedEdges[0].Curve
        assert type(curve) is Part.BSplineCurve
        # assert that the bspline pass through the points
        # get the points parameter
        params = [curve.parameter(Base.Vector(p)) for p in pntslist]
        # get the points on the curve at the calculated parameters
        test_points = cadapi.vector_to_list([curve.value(par) for par in params])
        # assert that the points on the curve are equal (within a tolerance) to the
        # points used to generate the bspline
        assert np.allclose(
            np.array(test_points) - np.array(pntslist), 0, atol=D_TOLERANCE
        )

    def test_length(self):
        open_wire: Part.Wire = cadapi.make_polygon(self.square_points)
        assert (
            cadapi.length(open_wire)
            == open_wire.Length
            == pytest.approx(3.0, rel=0, abs=EPS)
        )
        closed_wire: Part.Wire = cadapi.make_polygon(self.closed_square_points)
        assert (
            cadapi.length(closed_wire)
            == closed_wire.Length
            == pytest.approx(4.0, rel=0, abs=EPS)
        )

    def test_area(self):
        wire: Part.Wire = cadapi.make_polygon(self.closed_square_points)
        assert cadapi.area(wire) == wire.Area == pytest.approx(0.0, rel=0, abs=EPS)
        face: Part.Face = Part.Face(wire)
        assert cadapi.area(face) == face.Area == pytest.approx(1.0, rel=0, abs=EPS)

    def test_center_of_mass(self):
        wire: Part.Wire = cadapi.make_polygon(self.closed_square_points)
        face: Part.Face = Part.Face(wire)
        com = cadapi.center_of_mass(wire)
        comparison = com == np.array((0.5, 0.5, 0.0))
        assert isinstance(com, np.ndarray)
        assert comparison.all()

    def test_split_circular_wire(self):
        full_circle = cadapi.make_circle(radius=1.0, center=(1, 0, 0), axis=(0, 1, 0))
        arc_of_circ = cadapi.make_circle_arc_3P(
            [0, 0, 0], [1, 1, 0], [2, 0, 0], axis=(0, 1, 0)
        )
        semi_circle_lower, semi_circle_upper = cadapi.split_wire(
            full_circle, [0, 0, 0], EPS * 10
        )
        assert np.allclose(
            cadapi.start_point(semi_circle_upper) - cadapi.start_point(arc_of_circ),
            0,
            atol=D_TOLERANCE,
        )
        assert cadapi.split_wire(arc_of_circ, [2, 0, 0], EPS * 10)[1] is None
        assert (
            list(cadapi.split_wire(full_circle, [2, 0, 0], EPS * 10)).count(None) == 1
        ), (
            "Splitting vertex on the start- AND end-point, "
            "so one of the wires must have zero length."
        )

        with pytest.raises(FreeCADError):
            cadapi.split_wire(full_circle, (3, 0, 0), EPS * 10)
        with pytest.raises(FreeCADError):
            cadapi.split_wire(arc_of_circ, (3, 0, 0), EPS * 10)
        with pytest.raises(FreeCADError):
            cadapi._split_edge(
                arc_of_circ.OrderedEdges[0], 0.0
            )  # angle=0.0 radian is invalid here.

    def test_split_elliptical_wire(self):
        ellipse = cadapi.make_ellipse(
            major_radius=2, minor_radius=1, major_axis=(1, 0, 0), minor_axis=(0, 0, -1)
        )
        _, arc_of_ellipse = cadapi.split_wire(ellipse, [0, 0, -1], EPS * 10)
        assert np.isclose(arc_of_ellipse.Edges[0].LastParameter, 2 * np.pi)
        same_arc, none = cadapi.split_wire(arc_of_ellipse, [2, 0, 0], EPS * 10)

        with pytest.raises(FreeCADError):
            cadapi._split_edge(arc_of_ellipse.OrderedEdges[0], 0.0)

    def test_split_nonperiodic_wire(self):
        closed_wire = cadapi.make_polygon(self.closed_square_points)
        bezier = cadapi.make_bezier(self.square_points)
        bspline = cadapi.interpolate_bspline(self.square_points)
        cadapi.split_wire(closed_wire, self.closed_square_points[1], EPS * 10)
        cadapi.split_wire(bezier, self.square_points[0], EPS * 10)
        cadapi.split_wire(bspline, self.square_points[1], EPS * 10)

    def test_scale_shape(self):
        factor = 2.0
        wire: Part.Wire = cadapi.make_polygon(self.closed_square_points)
        scaled_wire = cadapi.scale_shape(wire.copy(), factor)
        face: Part.Face = Part.Face(scaled_wire)
        assert cadapi.area(face) == pytest.approx(1.0 * factor**2, rel=0, abs=EPS)
        assert (
            cadapi.length(face)
            == cadapi.length(scaled_wire)
            == pytest.approx(4.0 * factor, rel=0, abs=EPS)
        )
        face_from_wire = Part.Face(wire)
        scaled_face = cadapi.scale_shape(face_from_wire.copy(), factor)
        assert cadapi.length(scaled_face) == cadapi.length(face)
        assert cadapi.area(scaled_face) == cadapi.area(face)

    def test_discretise(self):
        wire: Part.Wire = cadapi.make_polygon(self.closed_square_points)
        ndiscr = 10
        points = cadapi.discretise(wire, ndiscr)
        assert len(points) == ndiscr
        length_w = wire.Length
        dl = length_w / float(ndiscr - 1)
        points = cadapi.discretise(wire, dl=dl)
        assert len(points) == ndiscr

    def test_discretise_by_edges(self):
        wire: Part.Wire = cadapi.make_polygon(self.closed_square_points)
        ndiscr = 10
        points = cadapi.discretise_by_edges(wire, ndiscr)

        dl = 0.4
        points1 = cadapi.discretise_by_edges(wire, dl=dl)

        dl = 0.4
        points2 = cadapi.discretise_by_edges(wire, ndiscr=100, dl=dl)
        assert np.allclose(points1 - points2, 0, atol=D_TOLERANCE)

    def test_discretise_vs_discretise_by_edges(self):
        wire1 = cadapi.make_polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
        wire2 = cadapi.make_polygon([[0, 0, 0], [0, 1, 0], [1, 1, 0]])
        wire2.reverse()
        wire = Part.Wire([wire1, wire2])

        # ndiscr is chosen in such a way that both discretise and discretise_by_edges
        # give the same points (so that a direct comparison is possible).
        points1 = cadapi.discretise(wire, ndiscr=5)
        points2 = cadapi.discretise_by_edges(wire, ndiscr=4)

        # assert that points1 and points2 are the same
        assert np.allclose(points1 - points2, 0, atol=D_TOLERANCE)

    def test_start_point_given_polygon(self):
        wire = cadapi.make_polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0]])

        start_point = cadapi.start_point(wire)

        assert isinstance(start_point, np.ndarray)
        np.testing.assert_equal(start_point, np.array([0, 0, 0]))

    def test_end_point_given_polygon(self):
        wire = cadapi.make_polygon([[0, 0, 0], [1, 0, 0], [1, 1, 0]])

        end_point = cadapi.end_point(wire)

        assert isinstance(end_point, np.ndarray)
        np.testing.assert_equal(end_point, np.array([1, 1, 0]))

    def test_catcherror(self):
        @cadapi.catch_caderr(ValueError)
        def func():
            raise FreeCADError("Error")

        with pytest.raises(ValueError):  # noqa: PT011
            func()

    def test_save_cad(self, tmp_path):
        shape = cadapi.extrude_shape(cadapi.make_circle(), (0, 0, 1))
        filename = f"{tmp_path}/tst.stp"

        cadapi.save_cad([shape], filename)
        assert Path(filename).exists
        stp_content = Path(filename).read_text()
        assert "myshape" not in stp_content
        assert "Bluemira" in stp_content
        assert (
            "AP242_MANAGED_MODEL_BASED_3D_ENGINEERING_MIM_LF. {1 0 10303 442 1 1 4"
            in stp_content
        )

        cadapi.save_cad(
            [shape],
            filename,
            labels=["myshape"],
            author="myfile",
            stp_file_scheme="AP214IS",
        )
        assert Path(filename).exists()
        stp_content = Path(filename).read_text()
        assert "myshape" in stp_content  # shape label in file
        assert "Bluemira" in stp_content  # bluemira still in file if author changed
        assert "myfile" in stp_content  # author change
        assert (
            "AUTOMOTIVE_DESIGN { 1 0 10303 214 1 1 1 1 }" in stp_content
        )  # scheme change

    def test_circle_ellipse_arc(self):
        # from make_circle
        arc = cadapi.make_circle(start_angle=0, end_angle=180)
        assert cadapi.length(arc) == np.pi  # check length of created arc
        assert np.allclose(
            cadapi.start_point(arc), [1.0, 0.0, 0.0]
        )  # check start point of arc
        assert np.allclose(
            cadapi.end_point(arc), [-1.0, 0.0, 0.0]
        )  # check end point of arc
        arc2 = cadapi.make_circle(
            start_angle=360, end_angle=180
        )  # same as arc but using start>end
        assert np.allclose(cadapi.discretise(arc, 10), cadapi.discretise(arc2, 10))

        # from make_circle_arc_3P
        arc3 = cadapi.make_circle_arc_3P(
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]
        )
        assert np.allclose(
            cadapi.discretise(arc, 10), cadapi.discretise(arc3, 10)
        )  # check arc3 matches arc
        with pytest.raises(FreeCADError):
            cadapi.make_circle_arc_3P([1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0])

        # from make_ellipse
        arc4 = cadapi.make_ellipse(start_angle=0, end_angle=90)
        ellipse_major_radius = 2
        ellipse_eccentricity = 0.75
        arc_length = ellipse_major_radius * ellipe(
            ellipse_eccentricity
        )  # length of arc using complete elliptic integral
        assert np.isclose(arc_length, cadapi.length(arc4), 6)


# Commented out CADFileTypes dont work with basic shapes tested or needed more
# FreeCAD imported, should be reviewed in future
cad_test_parameterisation = [
    ("ASCII_STEREO_MESH", True),
    ("ADDITIVE_MANUFACTURING", False),  # import not implemented
    ("AUTOCAD_DXF", False),  # import segfault
    ("BINMESH", True),
    ("BREP", True),
    pytest.param("CSG", True, marks=[pytest.mark.xfail(reason="import fails")]),
    ("FREECAD", True),
    ("GLTRANSMISSION", True),
    ("IGES", True),
    ("INVENTOR_V2_1", True),
    ("JSON", False),  # import not implemented
    ("OBJ", True),
    ("OBJ_WAVE", True),
    ("OFF", True),
    ("OPENSCAD", False),  # requires openscad package see openscad.org
    ("PLY_STANFORD", True),
    ("SIMPLE_MODEL", True),
    ("STEP", True),
    ("STEP_ZIP", True),  # Case sensitive extension  # possible import wrong
    ("STL", True),
    ("SVG_FLAT", True),  # returns face
    ("WEBGL", False),  # import not imlemented
    # "THREED_MANUFACTURING",  # segfault
    pytest.param("IFC_BIM", True, marks=[skipif_import_error("ifcopenshell")]),
    pytest.param(
        "IFC_BIM_JSON",  # github.com/buildingSMART/ifcJSON
        False,  # import not implemented
        marks=[skipif_import_error("ifcopenshell", "ifcjson")],
    ),
    pytest.param("DAE", True, marks=[skipif_import_error("collada")]),
    pytest.param("AUTOCAD", True, marks=[pytest.mark.xfail(reason="LibreDWG required")]),
    # TODO @je-cook: Part.Feature has no compatible object type, find compatible object
    # 3713
    pytest.param("ASC", True, marks=[pytest.mark.xfail(reason="No file created")]),
    pytest.param("BDF", True, marks=[pytest.mark.xfail(reason="No file created")]),
    pytest.param("DAT", True, marks=[pytest.mark.xfail(reason="No FEM object")]),
    pytest.param(
        "FENICS_FEM", True, marks=[pytest.mark.xfail(reason="No file created")]
    ),
    pytest.param(
        "FENICS_FEM_XML", True, marks=[pytest.mark.xfail(reason="No file created")]
    ),
    pytest.param("INP", True, marks=[pytest.mark.xfail(reason="No FEM object")]),
    pytest.param("MED", True, marks=[pytest.mark.xfail(reason="No FEM object")]),
    pytest.param("MESHJSON", True, marks=[pytest.mark.xfail(reason="No file created")]),
    pytest.param("MESHPY", True, marks=[pytest.mark.xfail(reason="No file created")]),
    pytest.param("MESHYAML", True, marks=[pytest.mark.xfail(reason="No file created")]),
    pytest.param("PCD", True, marks=[pytest.mark.xfail(reason="No file created")]),
    pytest.param("PLY", True, marks=[pytest.mark.xfail(reason="No file created")]),
    pytest.param(
        "TETGEN_FEM", True, marks=[pytest.mark.xfail(reason="No file created")]
    ),
    pytest.param("UNV", True, marks=[pytest.mark.xfail(reason="No FEM object")]),
    pytest.param("VTK", True, marks=[pytest.mark.xfail(reason="No FEM object")]),
    pytest.param("VTU", True, marks=[pytest.mark.xfail(reason="No FEM object")]),
    pytest.param("YAML", True, marks=[pytest.mark.xfail(reason="No file created")]),
    pytest.param(
        "Z88_FEM_MESH", True, marks=[pytest.mark.xfail(reason="No FEM object")]
    ),
    pytest.param(
        "Z88_FEM_MESH_2", True, marks=[pytest.mark.xfail(reason="No file created")]
    ),
    # Requires imports which requires a full GUI
    # pytest.param("SVG", marks=[pytest.mark.xfail(reason="No DrawingGui found")]),
    # pytest.param("PDF", marks=[pytest.mark.xfail(reason="More GUI required")]),
    # pytest.param("VRML", marks=[pytest.mark.xfail(reason="More GUI required")]),
    # pytest.param("VRML_ZIP", marks=[pytest.mark.xfail(reason="More GUI required")]),
    # pytest.param("WEBGL_X3D", marks=[pytest.mark.xfail(reason="More GUI required")]),
    # pytest.param("X3D", marks=[pytest.mark.xfail(reason="More GUI required")]),
    # pytest.param("X3DZ", marks=[pytest.mark.xfail(reason="More GUI required")]),
]


class TestCADFiletype:
    @classmethod
    def setup_class(cls):
        cls.doc = newDocument("TestCADFileType")
        cls.shape = cls.doc.addObject("Part::FeaturePython", "Circle")
        cls.shape.Shape = cadapi.extrude_shape(cadapi.make_circle(), (0, 0, 1))
        cls.doc.recompute()

    @classmethod
    def teardown_class(cls):
        closeDocument(cls.doc.Name)

    def setup_method(self):
        import FreeCADGui  # noqa: PLC0415

        if not hasattr(FreeCADGui, "subgraphFromObject"):
            FreeCADGui.setupWithoutGUI()

    @pytest.mark.parametrize(("name", "ftype"), cadapi.CADFileType.__members__.items())
    def test_init(self, name, ftype):
        assert cadapi.CADFileType[name] == ftype
        assert cadapi.CADFileType(ftype.value) == ftype

    @pytest.mark.parametrize(("name", "imprt"), cad_test_parameterisation)
    def test_exporter_function_exists_and_creates_a_file_and_imported(
        self, name, imprt, tmp_path
    ):
        filetype = cadapi.CADFileType[name]
        filename = f"{tmp_path / 'tst'}.{filetype.ext}"
        if name != "FREECAD":  # custom function in this case
            assert filetype.exporter.__name__ == "export"

        filetype.exporter([self.shape], filename)
        assert Path(filename).exists()

        if not imprt:
            return

        with cadapi.Document() as doc:
            filetype.importer(filename, doc.doc.Name)

            if filetype not in cadapi.CADFileType.mesh_import_formats():
                objs = doc.doc.Objects
                assert len(objs) == 1
                if filetype is cadapi.CADFileType.BREP:
                    assert objs[0].Label == "tst"
                else:
                    assert objs[0].Label == "Circle"
                if filetype is cadapi.CADFileType.SVG_FLAT:
                    # A flat svg is not going to be 3D
                    assert isinstance(objs[0].Shape, cadapi.apiFace)
                elif filetype is cadapi.CADFileType.BREP:
                    assert isinstance(objs[0].Shape, cadapi.apiCompound)
                else:
                    assert isinstance(objs[0].Shape, cadapi.apiShell)
