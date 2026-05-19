# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import os
from pathlib import Path
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("BLUEMIRA_GEOMETRY_BACKEND", "freecad") != "freecad",
    reason="FreeCAD-API tests; active backend is not freecad",
)

import freecad  # noqa: E402, F401
import Part  # noqa: E402
import numpy as np  # noqa: E402
from FreeCAD import Base, closeDocument, newDocument  # noqa: E402

import bluemira.codes.cadapi._freecad.api as cadapi  # noqa: E402
from bluemira.base.constants import EPS  # noqa: E402
from bluemira.codes.error import FreeCADError  # noqa: E402
from bluemira.geometry.constants import D_TOLERANCE, EPS_FREECAD  # noqa: E402
from tests._helpers import skipif_import_error  # noqa: E402
from tests.codes._shared.backend_api_tests import BackendApiTestsBase  # noqa: E402


class TestFreecadapi(BackendApiTestsBase):
    cadapi = cadapi

    @classmethod
    def setup_class(cls):
        cls.square_points = list(BackendApiTestsBase.square_points)
        cls.closed_square_points = list(BackendApiTestsBase.closed_square_points)

    @staticmethod
    def offsetter(wire):
        return cadapi.offset_wire(wire, 0.05, join="intersect", open_wire=False)

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
            "bluemira.codes.cadapi._freecad.api.arrange_edges",
            new=lambda a, b: b,  # noqa: ARG005
        ):
            wire1 = self.offsetter(circ)
            wire2 = self.offsetter(wire1)

        assert circ.Length < wire1.Length
        # these two should break in future, this mean the topo naming may be fixed
        assert circ.Length == wire2.Length
        assert wire1.Length > wire2.Length

    def test_fail_points_to_numpy(self):
        with pytest.raises(TypeError):
            arr = cadapi.point_to_numpy(self.square_points)

    def test_point_to_numpy(self):
        vectors = [Part.Point(Base.Vector(v)) for v in self.square_points]
        arr = cadapi.point_to_numpy(vectors)
        comparison = arr == np.array(self.square_points)
        assert comparison.all()

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

    def test_split_circular_wire_split_edge(self):
        """FreeCAD-only: ``_split_edge`` is an internal helper not present in
        other backends.
        """
        arc_of_circ = cadapi.make_circle_arc_3P(
            [0, 0, 0], [1, 1, 0], [2, 0, 0], axis=(0, 1, 0)
        )
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

    @pytest.mark.parametrize(
        ("two_pi_offset", "positive_y_axis", "reverse"),
        [
            (0, True, False),
            (0, True, True),
            (0, False, False),
            (0, False, True),
            (1, True, False),
            (1, True, True),
            (1, False, False),
            (1, False, True),
            (-1, True, False),
            (-1, True, True),
            (-1, False, False),
            (-1, False, True),
            (-2, True, False),
            (-2, True, True),
            (-2, False, False),
            (-2, False, True),
        ],
    )
    def test_reverse(self, two_pi_offset: int, *, positive_y_axis: bool, reverse: bool):
        """
        By swapping the start-end points (x2), and choosing opposite rotation axis (x2),
        re-drawing the same arc four times (=4).

        This function also helps justify the logic inside cadapi._convert_edge_to_curve.

        Parameters
        ----------
        two_pi_offset:
            an integer n, so we will offset the end-angle by 2*pi*n.
            Note: This proves that for Part.Circle and Part.Ellipse,

        positive_y_axis:
            whether the rotation axis is [0,1,0] or [0,-1,0]. This proves that flipping
            the rotation axis is equivalent to re-calculating all angles as "clockwise"
            instead of "anti-clockwise".

        reverse:
            whether to run the .reverse() method on the Part.Wire or not.
            Note: this test-method proves that .reverse() does NOT alter any information
            about the `Part.Circle`, only alters the .StartPoint and .EndPoint of
            `Part.ArcOfCircle`, but this information does not gets stored.
        """
        axis = [0, -1 + 2 * int(positive_y_axis), 0]
        circle = cadapi.make_circle(axis=axis).Edges[0].Curve
        assert np.isclose(circle.FirstParameter, 0, rtol=0, atol=EPS_FREECAD)
        assert np.isclose(circle.LastParameter, 2 * np.pi, rtol=0, atol=EPS_FREECAD)
        if positive_y_axis:
            arc_wire = Part.Wire(
                Part.Edge(
                    Part.ArcOfCircle(circle, 0, 0.5 * np.pi + 2 * np.pi * two_pi_offset)
                )
            )
        else:
            arc_wire = Part.Wire(
                Part.Edge(
                    Part.ArcOfCircle(
                        circle, 1.5 * np.pi, 2 * np.pi + 2 * np.pi * two_pi_offset
                    )
                )
            )
            # will get autocorrected to [1.5π, 2π] no matter which 2π interval we input
        if reverse:
            arc_wire.reverse()
        edge = arc_wire.Edges[0]

        if reverse:
            assert arc_wire.Orientation == "Reversed"
            assert edge.Orientation == "Reversed"
        else:
            assert arc_wire.Orientation == "Forward"
            assert edge.Orientation == "Forward"
        # edge.parameter is always increasing.
        if positive_y_axis:
            assert np.isclose(
                edge.parameterAt(edge.firstVertex()),
                0.0 * np.pi,
                rtol=0.0,
                atol=EPS_FREECAD,
            )
            assert np.isclose(
                edge.parameterAt(edge.lastVertex()),
                0.5 * np.pi,
                rtol=0.0,
                atol=EPS_FREECAD,
            )
        else:
            assert np.isclose(
                edge.parameterAt(edge.firstVertex()),
                1.5 * np.pi,
                rtol=0.0,
                atol=EPS_FREECAD,
            )
            assert np.isclose(
                edge.parameterAt(edge.lastVertex()),
                2.0 * np.pi,
                rtol=0.0,
                atol=EPS_FREECAD,
            )
        # regardless of reversing the wire or flipping the rotation axis of the wire,
        # .valueAt uses the underlying .Curve, which is a circle pointed to the RHS.
        np.testing.assert_allclose(
            list(edge.valueAt(0)), (1.0, 0.0, 0.0), rtol=0.0, atol=EPS_FREECAD
        )
        assert np.isclose(
            edge.Curve.parameter(cadapi.apiVector((1.0, 0.0, 0.0))),
            0.0,
            rtol=0.0,
            atol=EPS_FREECAD,
        )

    @pytest.mark.parametrize(("reverse"), [True, False])
    def test_serialise_part_ellipse(self, *, reverse: bool):
        """Checks for invertibility of the serialise function for Part.Ellipse."""
        ellipse = Part.Ellipse()
        ellipse.Axis = cadapi.apiVector([0.1, -0.2, 0.3]).normalize()
        ellipse_arc = Part.ArcOfEllipse(ellipse, 0.0, 5.0)
        arc_of_ellipse = Part.Wire(Part.Edge(ellipse_arc))
        if reverse:
            arc_of_ellipse.reverse()

        reconstructed = cadapi.deserialise_shape(cadapi.serialise_shape(arc_of_ellipse))
        if reverse:
            np.testing.assert_allclose(
                arc_of_ellipse.Edges[0].firstVertex().Point,
                reconstructed.Edges[0].lastVertex().Point,
                rtol=0,
                atol=EPS_FREECAD,
            )
            np.testing.assert_allclose(
                arc_of_ellipse.Edges[0].lastVertex().Point,
                reconstructed.Edges[0].firstVertex().Point,
                rtol=0,
                atol=EPS_FREECAD,
            )
        else:
            np.testing.assert_allclose(
                arc_of_ellipse.Edges[0].firstVertex().Point,
                reconstructed.Edges[0].firstVertex().Point,
                rtol=0,
                atol=EPS_FREECAD,
            )
            np.testing.assert_allclose(
                arc_of_ellipse.Edges[0].lastVertex().Point,
                reconstructed.Edges[0].lastVertex().Point,
                rtol=0,
                atol=EPS_FREECAD,
            )


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

    def test_bad_init(self):
        match = "not a valid CADFileType"
        with pytest.raises(ValueError, match=match):
            cadapi.CADFileType("😇")

        with pytest.raises(ValueError, match=match):
            cadapi.CADFileType("hello")

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
