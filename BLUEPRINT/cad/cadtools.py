# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
CAD functions and operations
"""
# High level imports
import os
from itertools import zip_longest

import matplotlib.pyplot as plt
import numpy as np

# OCC imports
try:
    from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCC.Core.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_MakeSolid,
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_NurbsConvert,
        BRepBuilderAPI_Sewing,
        BRepBuilderAPI_Transform,
    )
    from OCC.Core.BRepGProp import (
        brepgprop_LinearProperties,
        brepgprop_SurfaceProperties,
        brepgprop_VolumeProperties,
    )
    from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.Core.BRepOffsetAPI import BRepOffsetAPI_MakePipe, BRepOffsetAPI_ThruSections
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakePrism, BRepPrimAPI_MakeRevol

    # from OCC.Display.SimpleGui import init_display
    from OCC.Core.Geom import Geom_BezierCurve
    from OCC.Core.GeomAbs import GeomAbs_C2
    from OCC.Core.GeomAPI import GeomAPI_Interpolate, GeomAPI_PointsToBSpline
    from OCC.Core.gp import (
        gp_Ax1,
        gp_Ax2,
        gp_Circ,
        gp_Dir,
        gp_Pln,
        gp_Pnt,
        gp_Trsf,
        gp_Vec,
    )
    from OCC.Core.GProp import GProp_GProps

    # File exporting utility imports
    from OCC.Core.Interface import Interface_Static_SetCVal, Interface_Static_Update
    from OCC.Core.ShapeFix import ShapeFix_Shape, ShapeFix_Wire
    from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCC.Core.StlAPI import StlAPI_Writer
    from OCC.Core.TColgp import (
        TColgp_Array1OfPnt,
        TColgp_Array1OfPnt2d,
        TColgp_Array1OfVec,
        TColgp_HArray1OfPnt,
    )
    from OCC.Core.TColStd import TColStd_HArray1OfBoolean
    from OCC.Core.TopAbs import (
        TopAbs_COMPOUND,
        TopAbs_COMPSOLID,
        TopAbs_EDGE,
        TopAbs_FACE,
        TopAbs_SHELL,
        TopAbs_SOLID,
        TopAbs_VERTEX,
        TopAbs_WIRE,
    )

    # OCC type mapping imports
    from OCC.Core.TopoDS import TopoDS_Builder, TopoDS_Compound, TopoDS_Shape, topods
    from OCC.Core.TopTools import TopTools_ListOfShape
except ImportError:
    from OCC.BRepAlgoAPI import BRepAlgoAPI_Cut, BRepAlgoAPI_Fuse
    from OCC.BRepBuilderAPI import (
        BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace,
        BRepBuilderAPI_MakePolygon,
        BRepBuilderAPI_MakeSolid,
        BRepBuilderAPI_MakeWire,
        BRepBuilderAPI_NurbsConvert,
        BRepBuilderAPI_Sewing,
        BRepBuilderAPI_Transform,
    )
    from OCC.BRepGProp import (
        brepgprop_LinearProperties,
        brepgprop_SurfaceProperties,
        brepgprop_VolumeProperties,
    )
    from OCC.BRepMesh import BRepMesh_IncrementalMesh
    from OCC.BRepOffsetAPI import BRepOffsetAPI_MakePipe, BRepOffsetAPI_ThruSections
    from OCC.BRepPrimAPI import BRepPrimAPI_MakePrism, BRepPrimAPI_MakeRevol

    # from OCC.Display.SimpleGui import init_display
    from OCC.Geom import Geom_BezierCurve
    from OCC.GeomAbs import GeomAbs_C2
    from OCC.GeomAPI import GeomAPI_Interpolate, GeomAPI_PointsToBSpline
    from OCC.gp import gp_Ax1, gp_Ax2, gp_Circ, gp_Dir, gp_Pln, gp_Pnt, gp_Trsf, gp_Vec
    from OCC.GProp import GProp_GProps

    # File exporting utility imports
    from OCC.Interface import Interface_Static_SetCVal, Interface_Static_Update
    from OCC.ShapeFix import ShapeFix_Shape, ShapeFix_Wire
    from OCC.STEPControl import STEPControl_AsIs, STEPControl_Writer
    from OCC.StlAPI import StlAPI_Writer
    from OCC.TColgp import (
        TColgp_Array1OfPnt,
        TColgp_Array1OfPnt2d,
        TColgp_Array1OfVec,
        TColgp_HArray1OfPnt,
    )
    from OCC.TColStd import TColStd_HArray1OfBoolean
    from OCC.TopAbs import (
        TopAbs_COMPOUND,
        TopAbs_COMPSOLID,
        TopAbs_EDGE,
        TopAbs_FACE,
        TopAbs_SHELL,
        TopAbs_SOLID,
        TopAbs_VERTEX,
        TopAbs_WIRE,
    )

    # OCC type mapping imports
    from OCC.TopoDS import TopoDS_Builder, TopoDS_Compound, TopoDS_Shape, topods
    from OCC.TopTools import TopTools_ListOfShape

# Other CAD imports
try:
    from aocxchange.step_ocaf import StepOcafExporter
except ImportError:
    from BLUEPRINT.cad.step_writer import StepWriter as StepOcafExporter

import trimesh

from bluemira.base.file import file_name_maker
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.utilities.tools import flatten_iterable
from BLUEPRINT.base.error import CADError
from BLUEPRINT.cad.display import QtDisplayer
from BLUEPRINT.geometry.geomtools import get_angle_between_points, get_dl

# BLUEPRINT imports
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.utilities.plottools import Plot3D
from BLUEPRINT.utilities.tools import expand_nested_list

MINIMUM_MESH_VOL = 1e-5
TOLERANCE = 1e-6
FIX_TOLERANCE = 1e-3


class ShapePropertyExtractor:
    """
    Utility class for extracting properties from OCC Shape objects
    """

    def __init__(self, shape, tolerance=TOLERANCE):
        self.shape = shape
        self.tolerance = tolerance

    def volume(self):
        """
        Returns
        -------
        volume: float
            The volume of the solid shape [m^3]
        """
        prop = GProp_GProps()
        brepgprop_VolumeProperties(self.shape, prop, self.tolerance)
        return prop

    def surface(self):
        """
        Returns
        -------
        surface: float
            The surface area of the solid shape [m^2]
        """
        prop = GProp_GProps()
        brepgprop_SurfaceProperties(self.shape, prop, self.tolerance)
        return prop

    def length(self):
        """
        Returns
        -------
        length: float
            The length of a wire [m]
        """
        prop = GProp_GProps()
        brepgprop_LinearProperties(self.shape, prop)
        return prop

    def gyration(self, lift_point, direction=None):
        """
        Returns
        -------
        gryation: float
            The radius of gyration of the object about a lift point [m]
        """
        if direction is None:
            direction = [0, 0, 1]

        axis = make_axis(lift_point, direction)
        return self.volume().RadiusOfGyration(axis)


class TopologyMapping:
    """
    Maps a topology type to OCC and returns the correct object. Used when
    handling ambiguous types being returned, such as in sew_shapes
    """

    def __init__(self):
        self.topo_types = {
            TopAbs_VERTEX: topods.Vertex,
            TopAbs_EDGE: topods.Edge,
            TopAbs_FACE: topods.Face,
            TopAbs_WIRE: topods.Wire,
            TopAbs_SHELL: topods.Shell,
            TopAbs_SOLID: topods.Solid,
            TopAbs_COMPOUND: topods.Compound,
            TopAbs_COMPSOLID: topods.CompSolid,
        }

    def __call__(self, shape):
        """
        Map the shape type to the correct class.
        """
        if isinstance(shape, TopoDS_Shape):
            return self.topo_types[shape.ShapeType()](shape)
        raise CADError(f"Shape {shape} does not have a ShapeType() method.")

    def __getitem__(self, item):
        """
        Implement list-like behaviour.
        """
        return self(item)


def show_CAD(*shapes):
    """
    Simple OCC viewing function for quick display of parts

    Parameters
    ----------
    shapes: (Shape, ..)
        Iterable of shape objects to display
    """
    qt_display = QtDisplayer()
    for shape in shapes:
        qt_display.add_shape(shape)
    qt_display.show()
    return qt_display  # Needs to return this!


def get_properties(shape, lift_point=None):
    """

    Parameters
    ----------
    shape: OCC Shape object
        Shape object to extract properties from

    Returns
    -------
    props: dict
        Dictionary of properties
    """
    props = {}
    prop = ShapePropertyExtractor(shape)
    props["Volume"] = prop.volume().Mass()
    props["Area"] = prop.surface().Mass()
    centre = prop.volume().CentreOfMass()
    props["CoG"] = {"x": centre.X(), "y": centre.Y(), "z": centre.Z()}
    if lift_point is not None:
        props["Rg"] = prop.gyration(lift_point)
    return props


# =============================================================================
# Property checking
# =============================================================================


def check_watertight(stl_filename):
    """
    Check watertightness of an STL file.

    Parameters
    ----------
    stl_filename: str
        Path and filename of .stl file to be checked

    Returns
    -------
    watertight: bool
        True if stl part is watertight
    """
    mesh = trimesh.load(stl_filename)
    return mesh.is_watertight


def check_good_STL(stl_filename):  # noqa :N802
    """
    Checks quality of an STL file and tests its applicability to neutronics. \n
    Presently checks:
        - watertightness
        - has volume
        - volume > 0

    Parameters
    ----------
    stl_filename: str
        Path and filename of .stl file to be checked

    Returns
    -------
    good: bool
        True if a good stl file
    """
    if not stl_filename.endswith(".stl"):
        stl_filename += ".stl"
    mesh = trimesh.load(stl_filename)
    return all([mesh.is_watertight, mesh.is_volume, mesh.volume > MINIMUM_MESH_VOL])


def check_STL_folder(folderpath):  # noqa :N802
    """
    Checks the quality of all STL files in a given folder

    Parameters
    ----------
    folderpath: str
        Full path directory name

    Returns
    -------
    result: dict(filename: bool, ..)
        Dictionary of suitability of STL files for use in neutronics
    """
    result = {}
    for _, _, files in os.walk(folderpath):
        for file in files:
            if file.endswith(".stl"):
                f_p = os.sep.join([folderpath, file])
                result[file] = check_good_STL(f_p)
    return result


# =============================================================================
# File exports
# =============================================================================


def save_as_STL(  # noqa :N802
    shape,
    filename="test",
    ascii_mode=False,
    linear_deflection=0.001,
    angular_deflection=0.5,
    scale=1,
):
    """
    Saves a shape in an STL format.

    Parameters
    ----------
    shape: OCC Shape
        The shape object to be saved to an STL file
    filename: str
        Full path filename
    ascii_mode: bool
        True: saves in ASCII format, False: saves in binary (more compact)
    linear_deflection: float (default 0.001)
        The STL face accuracy
    angular_deflection: float (default 0.5)
        More STL tweaks
    scale: float (default 1)
        The scale in which to save the Shape object
    """
    if not filename.endswith(".stl"):
        filename += ".stl"
    filename = file_name_maker(filename)
    if shape.IsNull():
        raise CADError("Shape is null.")
    if scale != 1:
        shape = scale_shape(shape, scale)
    mesh = BRepMesh_IncrementalMesh(
        shape, linear_deflection, False, angular_deflection, True
    )
    if not mesh.IsDone():
        raise CADError("Mesh is not done.")
    writer = StlAPI_Writer()
    writer.SetASCIIMode(ascii_mode)
    writer.Write(shape, filename)
    if not os.path.isfile(filename):
        raise IOError("File not written to disk.")


def save_as_STEP(
    shape, filename="test", partname=None, scale=1, standard="AP214"
):  # noqa :N802
    """
    Saves a shape in an STP format.

    Parameters
    ----------
    shape: OCC Shape
        The shape object to be saved to an STP file
    filename: str
        Full path filename
    partname: str
        The part name in the STEP file
    standard: str (default 'AP214')
        The STP file standard to use
    scale: float (default 1)
        The scale in which to save the Shape object
    """
    if not filename.endswith(".STP"):
        filename += ".STP"

    filename = file_name_maker(filename, lowercase=True)
    if partname is None:
        file = filename.split(os.sep)[-1][:-4]
        partname = "BLUEPRINT_" + file

    if shape.IsNull():
        raise CADError("Shape is null.")

    if scale != 1:
        shape = scale_shape(shape, scale)

    writer = STEPControl_Writer()
    if standard not in ["AP214", "AP203", "AP214IS"]:
        raise CADError("Select an appropriate STEP AP standard.")

    if standard == "AP214":
        standard += "IS"  # This is the string in OCC

    Interface_Static_SetCVal("write.step.schema", standard)
    Interface_Static_Update("write.step.schema")
    Interface_Static_SetCVal("write.step.product.name", partname)
    # Interface_Static_SetCVal('write.step.unit', 'MM')  # Default
    writer.Transfer(shape, STEPControl_AsIs)
    writer.Write(filename)

    if not os.path.isfile(filename):
        raise IOError("File not written to disk.")
    del writer


def save_as_STEP_assembly(
    *shapes, filename="test", partname=None, scale=1
):  # noqa :N802
    """
    Saves a series of Shape objects as a STEP assembly

    Parameters
    ----------
    shapes: (Shape, ..)
        Iterable of shape objects to be saved
    filename: str
        Full path filename of the STP assembly
    scale: float (default 1)
        The scale in which to save the Shape objects
    """
    if not filename.endswith(".STP"):
        filename += ".STP"
    filename = file_name_maker(filename)

    try:
        exporter = StepOcafExporter(filename, partname=partname)
    except TypeError:
        if partname is not None:
            bluemira_warn("Partname not saved to STEP file header")
        exporter = StepOcafExporter(filename)

    shapes = expand_nested_list(shapes)
    for shape in shapes:
        if scale != 1:
            shape = scale_shape(shape, scale)
        exporter.add_shape(shape)
    exporter.write_file()
    if not os.path.isfile(filename):
        raise IOError("File not written to disk.")


def make_compound(topos):
    """
    Make an OCC TopoDS_Compound object out of many shapes

    Parameters
    ----------
    *topos: list of OCC TopoDS_* objects
        A set of objects to be compounded

    Returns
    -------
    compound: OCC TopoDS_Compound object
        A compounded set of shapes
    """
    builder = TopoDS_Builder()
    comp = TopoDS_Compound()
    builder.MakeCompound(comp)
    for topology in topos:
        builder.Add(comp, topology)
    return comp


# =============================================================================
# Shape manipulations
# =============================================================================


def scale_shape(shape, scale):
    """
    Scale an object.

    Parameters
    ----------
    shape: OCC Shape object
        The shape to be scaled
    scale: float
        The scaling factor

    Returns
    -------
    scaled: OCC Shape object
        The scaled Shape object
    """
    scaler = gp_Trsf()
    scaler.SetScaleFactor(scale)
    scaled = BRepBuilderAPI_Transform(shape, scaler)
    return scaled.Shape()


def rotate_shape(shape, axis=None, angle=360):
    """
    Revolves a shape about an axis

    Parameters
    ----------
    shape: OCC Shape object
        The shape to be revolved
    axis: OCC Axis object (defaults to z axis revolution)
        The axis about which to revolve the shape
    angle: float
        The degree of revolution [°]

    Returns
    -------
    shape: OCC Shape object
        The revolved shape
    """
    if axis is None:
        axis = make_axis((0, 0, 0), (0, 0, 1))
    rotate = gp_Trsf()
    rotate.SetRotation(axis, np.radians(angle))
    return BRepBuilderAPI_Transform(shape, rotate).Shape()


def translate_shape(shape, vector):
    """
    Translates a shape by a vector

    Parameters
    ----------
    shape: OCC Shape object
        The shape to be translated
    vector: [float, float, float]
        The vector along which to translate the shape
    """
    translate = gp_Trsf()
    translate.SetTranslation(make_vector(vector))
    return BRepBuilderAPI_Transform(shape, translate).Shape()


def mirror_shape(shape, axis_coord):
    """
    Mirrors a shape about a plane

    Parameters
    ----------
    shape: OCC Shape object
        The shape to be mirrored
    axis_coord: OCC gp_Ax2 object
        The plane about which to mirror the shape

    Returns
    -------
    rshape: OCC Shape object
        The mirrored shape
    """
    mirror = gp_Trsf()
    mirror.SetMirror(axis_coord)
    return BRepBuilderAPI_Transform(shape, mirror).Shape()


# =============================================================================
# 2-D faces
# =============================================================================


def make_wire(loop):
    """
    Makes an OCC Wire object from a BLUEPRINT Loop object

    Parameters
    ----------
    loop: geometry::Loop object
        The loop from which to make a wire

    Returns
    -------
    wire: OCC Wire object
        The OCC wire object
    """
    wire = BRepBuilderAPI_MakeWire()
    points = [gp_Pnt(*pnt) for pnt in loop]
    for i in range(len(loop) - 1):
        edge = BRepBuilderAPI_MakeEdge(points[i], points[i + 1]).Edge()
        wire.Add(edge)
    return wire.Wire()


def make_face(loop, spline=False):
    """
    Makes an OCC Face object from a BLUEPRINT Loop object

    Parameters
    ----------
    loop: Geometry::Loop object
        The loop from which a face must be made
    spline: bool
        Flag for using make_spline_face

    Returns
    -------
    face: OCC Face object
    """
    if spline:
        return make_spline_face(loop)
    wire = _make_wire(loop)
    face = BRepBuilderAPI_MakeFace(wire, True)
    return face.Face()


def _make_wire(loop):
    poly = BRepBuilderAPI_MakePolygon()
    for i in loop:
        poly.Add(gp_Pnt(*i))
    poly.Close()
    return poly.Wire()


def make_shell(shell, spline=False):
    """
    Makes an OCC Face from a BLUEPRINT Shell

    Parameters
    ----------
    shell: geometry::Shell object
        The Shell from which to make a Shell face
    spline: bool
        Flag for using make_spline_face

    Returns
    -------
    face: OCC Face object
    """
    face1 = make_face(shell.outer, spline=spline)
    face2 = make_face(shell.inner, spline=spline)
    return boolean_cut(face1, face2)


# TODO: determine which of the two is best and keep it.
def make_shell2(shell, spline=False):
    """
    Makes an OCC Face from a BLUEPRINT Shell

    Parameters
    ----------
    shell: geometry::Shell object
        The Shell from which to make a Shell face
    spline: bool
        Flag for using make_spline_face

    Returns
    -------
    face: OCC Face object
    """
    if spline:
        wire = _make_spline_wire(shell.outer)
        hole = _make_spline_wire(shell.inner)
    else:
        wire = _make_wire(shell.outer)
        hole = _make_wire(shell.inner)
    face = BRepBuilderAPI_MakeFace(wire, True)
    face.Add(hole)
    return face.Face()


def make_circle(centre, direction, radius):
    """
    Makes an OCC Face for a circle

    Parameters
    ----------
    centre: [float, float, float]
        Centrepoint of the circle
    direction: [float, float, float]
        Direction vector of the circle
    radius: float
        The radius of the circle

    Returns
    -------
    circ: OCC Face object for the circular face
    """
    circ = gp_Circ()
    circ.SetAxis(make_axis(centre, direction))
    circ.SetRadius(radius)
    return _make_OCCface(_make_OCCwire(_make_OCCedge(circ)))


def make_spline_face(loop, **kwargs):
    """
    Creates a Bézier curve from a Loop.

    Parameters
    ----------
    loop: Loop
        The loop to transform.

    Returns
    -------
    face: OCC Face object
    """
    wire = _make_spline_wire(loop, **kwargs)
    face = BRepBuilderAPI_MakeFace(wire)
    return face.Face()


def _make_spline_wire(loop, **kwargs):
    try:  # BLUEPRINT Loop object
        spline = points_to_bspline(loop.xyz, **kwargs)
    except AttributeError:  # numpy array
        spline = points_to_bspline(Loop, **kwargs)

    try:
        closed = spline.GetObject().IsClosed()
    except AttributeError:
        closed = spline.IsClosed()
    if not closed:
        # It turns out this is common and not always an issue...
        # bluemira_warn('CAD::make_spline_face: Open spline!')
        pass
    else:
        # spline.GetObject().SetPeriodic()
        pass

    try:
        edge = BRepBuilderAPI_MakeEdge(spline.GetObject().GetHandle()).Edge()
    except AttributeError:
        try:
            edge = BRepBuilderAPI_MakeEdge(spline.GetHandle()).Edge()
        except AttributeError:
            edge = BRepBuilderAPI_MakeEdge(spline).Edge()
    return BRepBuilderAPI_MakeWire(edge).Wire()


def make_mixed_face(loop, **kwargs):
    """
    Parameters
    ----------
    loop: BLUEPRINT Loop object
        The Loop of coordinates to be converted to a OCC Face object

    Returns
    -------
    face: OCC Face object
        The OCC face of the mixed polygon/spline Loop
    """
    mfm = MixedFaceMaker(loop, **kwargs)
    try:
        mfm.build()

    except CADError:
        bluemira_warn("CAD: MixedFaceMaker failed to build as expected.")
        return make_face(loop, **kwargs)

    # Sometimes there won't be a RuntimeError, and you get a free SIGSEGV for your
    # troubles.
    area = get_properties(mfm.face)["Area"]
    if np.isclose(loop.area, area, rtol=5e-3):
        return mfm.face
    else:
        bluemira_warn("CAD: MixedFaceMaker failed to build as expected.")
        return make_face(loop, **kwargs)


def make_mixed_wire(loop, **kwargs):
    """
    Parameters
    ----------
    loop: BLUEPRINT Loop object
        The Loop of coordinates to be converted to a OCC Face object

    Returns
    -------
    face: OCC Wire object
        The OCC wire of the mixed polygon/spline Loop
    """
    mfm = MixedFaceMaker(loop, **kwargs)
    try:
        mfm.build()
    except CADError:
        bluemira_warn("CAD: MixedFaceMaker failed to build as expected.")
        return make_wire(loop)

    return mfm.wire


def make_mixed_shell(shell, **kwargs):
    """
    Makes an mixed OCC Face from a BLUEPRINT Shell

    Parameters
    ----------
    shell: geometry::Shell object
        The Shell of coordinates to be converted to a OCC Face object

    Returns
    -------
    face: OCC Face object
        The OCC face of the mixed polygon/spline Shell
    """
    true_area = shell.area
    # First try simple wires and making a hole
    outer = make_mixed_wire(shell.outer, **kwargs)
    inner = make_mixed_wire(shell.inner, **kwargs)
    face = BRepBuilderAPI_MakeFace(outer, True)
    face.Add(inner)
    face = face.Face()
    area = get_properties(face)["Area"]
    if np.isclose(true_area, area, rtol=5e-3):
        return face

    # Now try reversed inner wire (usually not necessary)
    inner.Reverse()
    face = BRepBuilderAPI_MakeFace(outer, True)
    face.Add(inner)
    face = face.Face()
    area = get_properties(face)["Area"]
    if np.isclose(true_area, area, rtol=5e-3):
        return face

    # Finally, try boolean cut
    outer = make_mixed_face(shell.outer, **kwargs)
    inner = make_mixed_face(shell.inner, **kwargs)
    return boolean_cut(outer, inner)


class MixedFaceMaker:
    """
    Utility class for the creation of OCC faces that combine splines and
    polygons. This is a decomposition of what would otherwise be a very
    long function.
    Polygons are detected by median length and turning angle.

    Parameters
    ----------
    loop: BLUEPRINT Loop object
        The Loop of coordinates to be converted to a OCC Face object

    Other Parameters
    ----------------
    median_factor: float
        The factor of the median for which to filter segment lengths
        (below median_factor*median_length --> spline)
    n_seg: int
        The minimum number of segments for a spline
    a_acute: float
        The angle [degrees] between two consecutive segments deemed to be too
        acute to be fit with a spline.
    debug: bool
        Whether or not to print debugging information
    """

    median_factor = 2.0  # Filter factor on segment length for distinction
    # between spline and segment
    n_seg = 4  # Minimum length of a spline in segments
    a_acute = 150  # Acute angle filter between segments [degree]
    debug = False  # Debug flag; will print and plot

    def __init__(self, loop, **kwargs):
        for key in ["median_factor", "n_seg", "debug"]:
            if key in kwargs:
                setattr(self, key, kwargs[key])
        self.loop = loop.copy()

        # Constructors
        self.edges = None
        self.wire = None
        self.face = None
        self.p_loops = None
        self.s_loops = None
        self.flag_spline_first = None
        self._debugger = None

    def build(self):
        """
        Carry out the MixedFaceMaker sequence to make a Face
        """
        # Get the vertices of polygonic segments
        p_vertices = self.find_polygon_vertices()

        # identify sequences of polygon indices
        p_sequences = self.get_polygon_sequences(p_vertices)

        # Get the (negative) of the polygon sequences to get spline seqs
        s_sequences = self.get_spline_sequences(p_sequences)

        if self.debug:
            print("p_sequences :", p_sequences)
            print("s_sequences :", s_sequences)

        # Make loops for all the segments
        self.make_subloops(p_sequences, s_sequences)

        if self.debug:
            self.plot()

        # Make the edges for each of the subloops, and daisychain them
        self.make_subedges()

        # Finally, make the OCC face from the wire of the edges
        self.make_OCC_face()

    def find_polygon_vertices(self):
        """
        Finds all vertices in the loop which belong to polygonic edges

        Returns
        -------
        vertices: np.array(dtype=int)
            The vertices of the loop which are polygonic
        """
        # find long segment indices
        segment_lengths = get_dl(*self.loop.d2)
        median = np.median(segment_lengths)

        long_indices = np.where(segment_lengths > self.median_factor * median)[0]

        # find sharp angle indices
        angles = np.zeros(len(self.loop) - 2)
        for i in range(len(self.loop) - 2):
            angles[i] = get_angle_between_points(
                self.loop.d2.T[i], self.loop.d2.T[i + 1], self.loop.d2.T[i + 2]
            )

        if self.loop.closed:
            # Get the angle over the closed joint
            join_angle = get_angle_between_points(
                self.loop.d2.T[-2], self.loop.d2.T[0], self.loop.d2.T[1]
            )
            angles = np.append(angles, join_angle)

        sharp_indices = np.where((angles <= self.a_acute) & (angles != 0))[0]
        # Convert angle numbering to segment numbering (both segments of angle)
        sharp_edge_indices = []
        for index in sharp_indices:
            sharp_edges = [index + 1, index + 2]
            sharp_edge_indices.extend(sharp_edges)
        sharp_edge_indices = np.array(sharp_edge_indices)

        # build ordered set of polygon edge indices
        indices = np.unique(np.append(long_indices, sharp_edge_indices))

        # build ordered set of polygon vertex indices
        vertices = []
        for index in indices:
            if index == len(self.loop):
                # If it is the last index, do not overshoot
                vertices.extend([index])
            else:
                vertices.extend([index, index + 1])
        vertices = np.unique(np.array(vertices, dtype=int))
        return vertices

    def get_polygon_sequences(self, vertices):
        """
        Gets the sequences of polygon segments

        Parameters
        ----------
        vertices: np.array(dtype=int)
            The vertices of the loop which are polygonic

        Returns
        -------
        p_sequences: list([start, end], [start, end])
            The list of start and end tuples of the polygon segments
        """
        sequences = []
        start = vertices[0]
        # Loop over all vertices except last (which is same as first)
        for i, vertex in enumerate(vertices[:-1]):

            delta = vertices[i + 1] - vertex

            # Add the last point in the loop but only if we've already
            # added something to sequences
            if i == len(vertices) - 2 and len(sequences) > 0:
                # end of loop clean-up
                end = vertices[i + 1]
                sequences.append([start, end])
                break

            if delta <= self.n_seg:
                # Spline would be too short, so stitch polygons together
                continue
            else:
                end = vertex
                sequences.append([start, end])
                start = vertices[i + 1]  # reset start index

        if not sequences:
            raise CADError("Not a good candidate for a mixed face ==> spline")

        # Now check the start and end of the loop, to see if a polygon segment
        # bridges the join
        first_p_vertex = sequences[0][0]
        last_p_vertex = sequences[-1][1]

        if first_p_vertex <= self.n_seg:
            if len(self.loop) - last_p_vertex <= self.n_seg:
                start_offset = self.n_seg - first_p_vertex
                end_offset = (len(self.loop) - last_p_vertex) + self.n_seg
                total = start_offset + end_offset
                if total <= self.n_seg:
                    start = sequences[-1][0]
                    end = sequences[0][1]
                    # Remove first sequence
                    sequences = sequences[1:]
                    # Replace last sequence with bridged sequence
                    sequences[-1] = [start, end]

        last_p_vertex = sequences[-1][1]
        if len(self.loop) - last_p_vertex <= self.n_seg:
            # There is a small spline section at the end of the loop, that
            # needs to be bridged
            if sequences[0][0] == 0:
                # There is no bridge -> take action
                start = sequences[-1][0]
                end = sequences[0][1]
                sequences = sequences[1:]
                sequences[-1] = [start, end]

        return sequences

    def get_spline_sequences(self, p_sequences):
        """
        Gets the sequences of spline segments

        Parameters
        ----------
        p_sequences: list([start, end], [start, end])
            The list of start and end tuples of the polygon segments

        Returns
        -------
        s_sequences: list([start, end], [start, end])
            The list of start and end tuples of the spline segments
        """
        s_sequences = []

        # Catch the start, if polygon doesn't start at zero, and there is no
        # bridge
        last = p_sequences[-1]
        if last[0] > last[1]:  # there is a polygon bridge
            pass  # Don't add a spline at the start
        else:
            # Check that the first polygon segment doesn't start at zero
            first = p_sequences[0]
            if first[0] == 0:
                pass
            else:  # It doesn't start at zero and there is no bridge: catch
                s_sequences.append([0, first[0]])

        for i, seq in enumerate(p_sequences[:-1]):
            start = seq[1]
            end = p_sequences[i + 1][0]
            s_sequences.append([start, end])

        # Catch the end, if polygon doesn't end at end
        if last[1] == len(self.loop):
            # NOTE: if this is true, there can't be a polygon bridge
            pass
        else:
            if last[0] > last[1]:  # there is a polygon bridge
                s_sequences.append([last[1], p_sequences[0][0]])
            else:
                s_sequences.append([last[1], len(self.loop)])

        # Check if we need to make a spline bridge
        s_first = s_sequences[0][0]
        s_last = s_sequences[-1][1]
        if (s_first == 0) and (s_last == len(self.loop)):
            # Make a spline bridge
            start = s_sequences[-1][0]
            end = s_sequences[0][1]
            s_sequences = s_sequences[1:]
            s_sequences[-1] = [start, end]

        if s_sequences[0][0] == 0:
            self.flag_spline_first = True
        else:
            self.flag_spline_first = False

        return s_sequences

    def make_subloops(self, p_sequences, s_sequences):
        """
        Creates Loop objects for all the spline and polygon segments

        Parameters
        ----------
        p_sequences: list([start, end], [start, end])
            The list of start and end tuples of the polygon segments
        s_sequences: list([start, end], [start, end])
            The list of start and end tuples of the spline segments
        """
        ploops = []
        sloops = []

        for seg in p_sequences:
            if seg[0] > seg[1]:
                # There is a bridge
                d = np.hstack((self.loop[seg[0] :], self.loop[0 : seg[1] + 1]))
                loop = Loop(*d, enforce_ccw=False)
            else:
                loop = Loop(*self.loop[seg[0] : seg[1] + 1], enforce_ccw=False)
            ploops.append(loop)

        for seg in s_sequences:
            if seg[0] > seg[1]:
                # There is a bridge
                d = np.hstack((self.loop[seg[0] :], self.loop[0 : seg[1] + 1]))
                loop = Loop(*d, enforce_ccw=False)
            else:
                loop = Loop(*self.loop[seg[0] : seg[1] + 1], enforce_ccw=False)
            sloops.append(loop)

        self.s_loops = sloops
        self.p_loops = ploops

    def make_subedges(self):
        """
        Creates a collection of TopoDS Edge objects and orders them
        appropriately
        """
        # First daisy-chain correctly...
        loop_order = []
        if self.flag_spline_first:
            set1, set2 = self.s_loops, self.p_loops
        else:
            set2, set1 = self.s_loops, self.p_loops
        for i, (a, b) in enumerate(zip_longest(set1, set2)):
            if a is not None:
                loop_order.append(set1[i])
            if b is not None:
                loop_order.append(set2[i])

        for i, loop in enumerate(loop_order[:-1]):
            if not (loop[-1] == loop_order[i + 1][0]).all():
                loop_order[i + 1].reverse()
                if i == 0:
                    if not (loop[-1] == loop_order[i + 1][0]).all():
                        loop.reverse()
                        if not (loop[-1] == loop_order[i + 1][0]).all():
                            loop_order[i + 1].reverse()

        if self.flag_spline_first:
            set1 = [self.make_spline(s) for s in self.s_loops]
            set2 = [self.make_polygon(p) for p in self.p_loops]
        else:
            set2 = [self.make_spline(s) for s in self.s_loops]
            set1 = [self.make_polygon(p) for p in self.p_loops]

        edges = []
        for i, (a, b) in enumerate(zip_longest(set1, set2)):
            if a is not None:
                edges.append(a)

            if b is not None:
                edges.append(b)

        self.edges = list(flatten_iterable(edges))
        self._debugger = loop_order

    def make_subwire(self):
        """

        Returns
        -------
        wire: TopoDS_Wire
            The wire for all the edges
        """
        # NOTE: TopTools_ListOfShape is input to BRepBuilderAPI_MakeWire and
        # this should correctly order the edges to the wire.
        e_list = TopTools_ListOfShape()
        for edge in self.edges:
            e_list.Append(edge)
        wire = BRepBuilderAPI_MakeWire()
        wire.Add(e_list)
        wire.Build()
        wire = wire.Wire()
        return fix_wire(wire)

    @staticmethod
    def make_spline(loop):
        """
        Make an OCC Spline from a Loop
        """
        spline = points_to_bspline(loop)
        try:
            edge = BRepBuilderAPI_MakeEdge(spline.GetObject().GetHandle()).Edge()
        except AttributeError:
            try:
                edge = BRepBuilderAPI_MakeEdge(spline.GetHandle()).Edge()
            except AttributeError:
                edge = BRepBuilderAPI_MakeEdge(spline).Edge()
        return edge

    @staticmethod
    def make_polygon(loop):
        """
        Make an OCC Polygon from a Loop
        """
        points = [gp_Pnt(*pnt) for pnt in loop]
        edges = []
        for i in range(len(loop) - 1):
            edge = BRepBuilderAPI_MakeEdge(points[i], points[i + 1]).Edge()
            edges.append(edge)
        return edges

    def make_OCC_face(self):  # noqa :N802
        """
        Makes TopoDS Wire and Face from the collection of edges
        """
        self.wire = self.make_subwire()
        self.face = _make_OCCface(self.wire)

    def plot(self, ax=None):
        """
        Debugging utility
        """
        if self.loop.ndim == 2:
            if ax is None:
                _, (ax1, ax2) = plt.subplots(1, 2)
            self.loop.plot(ax1, fill=False, points=True)

            for loop in self.s_loops:
                loop.plot(ax2, fill=False, edgecolor="r", linewidth=6)
                ax2.plot(*loop.d2.T[0], marker="s", color="r")
                ax2.plot(*loop.d2.T[-1], marker=">", color="r")
            for loop in self.p_loops:
                loop.plot(ax2, edgecolor="k", fill=False, points=True)

        else:
            if ax is None:
                ax = Plot3D()

            for loop in self.s_loops:
                loop.plot(ax, fill=False, edgecolor="r", linewidth=6)
                x, y, z = [loop.x[0]], [loop.y[0]], [loop.z[0]]
                ax.plot(x, y, z, marker="s", color="r")
                x, y, z = [loop.x[0]], [loop.y[0]], [loop.z[0]]
                ax.plot(x, y, z, marker=">", color="r")

            for loop in self.p_loops:
                loop.plot(ax, edgecolor="k", fill=False, points=True)


def fix_wire(wire):
    """
    Fixes a wire with potentially disconnected and poorly ordered edges. Drops
    small edges from the wire.

    Parameters
    ----------
    wire: TopoDS_Wire
        The wire to heal

    Returns
    -------
    wire: TopoDS_Wire
        The healed wire
    """
    sff = ShapeFix_Wire()
    sff.Load(wire)
    sff.FixReorder()
    sff.FixConnected()
    sff.FixSmall(True, FIX_TOLERANCE)
    return sff.Wire()


def sew_shapes(*shapes, tolerance=FIX_TOLERANCE):
    """
    Sew shapes together. NOTE: will not give you a Solid

    Parameters
    ----------
    shapes: iterable (TopoDS_Shape, TopoDS_Shape, ...)
        List of shapes to sew together
    tolerance: float (default = FIX_TOLERANCE)
        The tolerance on vertices to sew together

    Returns
    -------
    sewedshape: TopoDS_Shell
        The sewed Shell object
    """
    sew = BRepBuilderAPI_Sewing(tolerance)
    for shape in shapes:
        if isinstance(shape, list):
            for s in shape:
                sew.Add(s)
        else:
            sew.Add(shape)
    sew.Perform()
    return TopologyMapping()(sew.SewedShape())


def fix_shape(shape, tolerance=FIX_TOLERANCE):
    """
    Fix a shape.

    Parameters
    ----------
    shape: TopoDS_Shape
        The shape to fix
    tolerance: float
        The geometric tolerance to use for fixing.

    Returns
    -------
    fixed_shape: TopoDS_Shape
        The fixed shape
    """
    # TODO: test
    fix = ShapeFix_Shape(shape)
    fix.SetFixFreeShellMode(True)
    fix_shell_tool = fix.FixShellTool()
    try:
        fix_shell_tool.GetObject().SetFixOrientationMode(True)
    except AttributeError:
        fix_shell_tool.SetFixOrientationMode(True)
    fix.LimitTolerance(tolerance)
    fix.Perform()
    return fix.Shape()


# =============================================================================
# Primitive OCC objects
# =============================================================================


def make_axis(origin, direction):
    """
    Makes an OCC axis object

    Parameters
    ----------
    origin: tuple(float*3)
        x, y, z coordinates of the origin of the axis
    direction: tuple (float*3)
        x, y ,z coordinates of the direction vector

    Returns
    -------
    axis: OCC axis object
    """
    return gp_Ax1(gp_Pnt(*origin), gp_Dir(*direction))


def make_2d_axis(origin, direction1, direction2):
    """
    Makes an OCC gp_Ax2 object - used for mirroring

    Parameters
    ----------
    origin: Iterable(3)
        Origin point of the plane
    direction1: Iterable(3)
        Direction of the first plane axis form the origin
    direction2: Iterable(3)
        Direction of the second plane axis form the origin

    Returns
    -------
    ax2: OCC gp_Ax2 object
    """
    return gp_Ax2(gp_Pnt(*origin), gp_Dir(*direction1), gp_Dir(*direction2))


def make_vector(point1, point2=None):
    """
    Makes an OCC vector object

    Parameters
    ----------
    point1: Iterable(3)
        x, y, z coordinates of the first point
    point2: Iterable(3)
        x, y, z coordinates of the second point

    Returns
    -------
    axis: OCC vector object
    """
    if point2 is None:
        if isinstance(point1, gp_Vec):
            return point1
        else:
            return gp_Vec(float(point1[0]), float(point1[1]), float(point1[2]))
    else:
        return gp_Vec(gp_Pnt(*point1), gp_Pnt(*point2))


def make_plane(origin, direction):
    """
    Makes an OCC plane object

    Parameters
    ----------
    origin: tuple(float*3)
        x, y, z coordinates of the origin of the axis
    direction: tuple (float*3)
        x, y, z coordinates of the direction vector

    Returns
    -------
    axis: OCC plane object
    """
    return gp_Pln(gp_Pnt(*origin), gp_Dir(*direction))


# =============================================================================
# Solid operations
# =============================================================================


def revolve(profile, axis=None, angle=360):
    """
    Revolves a profile about an axis

    Parameters
    ----------
    profile: OCC Face object
        The profile to be revolved
    axis: OCC Axis object (defaults to z axis revolution)
        The axis about which to revolve the profile
    angle: float
        The degree of revolution [°]

    Returns
    -------
    shape: OCC Shape object
        The revolved shape
    """
    if axis is None:
        axis = gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))

    return BRepPrimAPI_MakeRevol(profile, axis, np.radians(angle)).Shape()


def extrude(profile, **kwargs):
    """
    Extrudes a profile

    Parameters
    ----------
    profile: OCC Face object
        The profile to be extruded

    Other Parameters
    ----------------
    vec: OCC vector object or (float, float, float)
        The vector along which to extrude the profile

    axis: str from ['x', 'y', 'z']
        The axis along which to extrude
    length: float
        The length along the axis along which to extrude

    Returns
    -------
    extrusion: OCC Shape object
        The extruded shape
    """
    vec = None
    if "length" and "axis" in kwargs:
        if kwargs["axis"] == "z":
            vec = make_vector([0, 0, 0], [0, 0, kwargs["length"]])
        elif kwargs["axis"] == "x":
            vec = make_vector([0, 0, 0], [kwargs["length"], 0, 0])
        elif kwargs["axis"] == "y":
            vec = make_vector([0, 0, 0], [0, kwargs["length"], 0])
    elif "vec" in kwargs:
        if kwargs["vec"].__class__.__name__ != "gp_Vec":
            vec = make_vector(kwargs["vec"])
        else:
            vec = kwargs["vec"]
    else:
        bluemira_warn("Use extrude with correct kwargs")
    if vec is None:
        raise CADError("CAD::extrude kein Vector!")

    extrusion = BRepPrimAPI_MakePrism(profile, vec, True)
    extrusion.Build()
    return extrusion.Shape()


def loft(*profiles):
    """
    Lofts a series of profiles

    Parameters
    ----------
    profiles: iterable (TopoDS_Wire, TopoDS_Wire, ...)
        Set of (ordered) profiles to be lofted

    Returns
    -------
    loft: TopoDS_Solid
        The lofted (solid) object
    """
    b = BRepOffsetAPI_ThruSections()
    f1 = _make_OCCface(profiles[0])
    f2 = _make_OCCface(profiles[-1])
    for p in profiles:
        b.AddWire(p)
    b.Build()
    s = sew_shapes(b.Shape(), f1, f2)
    s = BRepBuilderAPI_MakeSolid(s)
    s.Build()
    return s.Solid()


def sweep(profile, path):
    """
    Sweeps a profile along a path

    Parameters
    ----------
    profile: TopoDS_Face
        The profile to be swept
    path: TopoDS_Wire or Geom_BezierCurve

    Returns
    -------
    sweep: TopoDS_Shape
        The swept solid Shape object
    """
    if isinstance(path, Geom_BezierCurve):
        path = curve_to_wire(path)
    sweeping = BRepOffsetAPI_MakePipe(path, profile)
    sweeping.Build()
    return sweeping.Shape()


# =============================================================================
# Solid shape creation simplifiers
# =============================================================================


def make_box(corner, v1, v2, v3):
    """
    Make an oriented box.

    Parameters
    ----------
    corner: gp_Pnt
        The corner point coordinates
    v1, v2, v3: iterable(3)
        The x, y, and z direction vectors

    Returns
    -------
    box: OCC Shape object
        The oriented prism
    """
    p1 = np.array(corner, dtype=np.float64)
    v1 = np.array(v1, dtype=np.float64)
    v2 = np.array(v2, dtype=np.float64)
    v3 = np.array(v3, dtype=np.float64)
    p2 = p1 + v1
    p3 = p2 + v3
    p4 = p1 + v3
    x = [p1[0], p2[0], p3[0], p4[0]]
    y = [p1[1], p2[1], p3[1], p4[1]]
    z = [p1[2], p2[2], p3[2], p4[2]]
    loop = Loop(x, y, z)
    loop.close()
    face = make_face(loop)
    return extrude(face, vec=v2)


# =============================================================================
# Spline nightmares
# =============================================================================


def make_bezier_curve(points):
    """
    Create a Bézier curve from some points.

    Parameters
    ----------
    points : np.array (N, 3) or list [[x], [y], [z]]
        x, y, z coordinates of nodes for Bézier curve

    Returns
    -------
    crv : OCC.Geom.Geom_BezierCurve
    """
    pnts = point_array_to_TColgp_PntArrayType(points, TColgp_Array1OfPnt)
    crv = Geom_BezierCurve(pnts)
    return crv


def points_to_bspline(
    pnts,
    deg=3,
    periodic=False,
    tangents=None,
    scale=False,
    continuity=GeomAbs_C2,
    tol=TOLERANCE,
):
    """
    Makes a BSplineCurve from an x, y, z array of points

    Parameters
    ----------
    pnts : list or numpy array
        array of x, y, z points
    deg : integer
        degree of the fitted bspline
    periodic : Bool (default=False)
        If true, OCC.GeomAPI_Interpolate will be used instead of the
        GeomAPI_PointsToBspline. Curve tangent vectors can then be enforced at
        the interpolation pnts
    tangents : array (default=None)
        list of [x, y, z] tangent vectors to be specificied at points:
        if only 2 tangents are specified, these will be enforced at the
        start and end points, otherwise tangents should have the same length
        as pnts and will be enforced at each point.
    scale : bool (default=False)
        Will scale the tangents (gives a smoother Periodic curve if False)
    continuity : OCC.GeomAbs.GeomAbs_XX type (default C2)
        The order of continuity (C^0, C^1, C^2, G^0, ....)

    Returns
    -------
    crv : OCC.Geom.BSplineCurve
    """
    if not periodic and (tangents is None):
        crv = _points_to_bspline(pnts, deg=deg, continuity=continuity)
    else:
        typer = TColgp_HArray1OfPnt
        try:
            pnts = point_array_to_TColgp_PntArrayType(pnts, typer).GetHandle()
        except AttributeError:
            pnts = point_array_to_TColgp_PntArrayType(pnts, typer)
        interp = GeomAPI_Interpolate(pnts, periodic, tol)
        if tangents is not None:
            n_tangents = tangents.shape[0]
            if n_tangents == 2:
                interp.Load(gp_Vec(*tangents[0, :]), gp_Vec(*tangents[1, :]), scale)
            else:
                tan_array = TColgp_Array1OfVec(1, n_tangents)
                for i in range(1, n_tangents + 1):
                    tan_array.SetValue(i, gp_Vec(*tangents[i - 1, :]))

                tan_flags = TColStd_HArray1OfBoolean(1, n_tangents)
                tan_flags.Init(True)  # Set all true (enforce all tangents)
                try:
                    tan_flags = tan_flags.GetHandle()
                except AttributeError:
                    pass
                interp.Load(tan_array, tan_flags, scale)
        interp.Perform()
        crv = interp.Curve()
    return crv


def _points_to_bspline(points, deg=3, continuity=GeomAbs_C2):
    """
    Makes a BSplineCurve from an x, y, z array of points

    Parameters
    ----------
    points : list or numpy array
        array of x, y, z points
    deg : integer
        degree of the fitted OCC Bézier spline
    continuity : OCC.GeomAbs.GeomAbs_XX type (default C2)
        The order of continuity (C^0, C^1, C^2, G^0, ....)

    Returns
    -------
    crv : OCC.Geom.BSplineCurve
    """
    typer = TColgp_Array1OfPnt
    pnts = point_array_to_TColgp_PntArrayType(points, typer)
    deg_min, deg_max = deg, deg
    return GeomAPI_PointsToBSpline(pnts, deg_min, deg_max, continuity).Curve()


def curve_to_wire(curve):
    """
    Converts a Bézier curve to an OCC Wire object.

    Parameters
    ----------
    curve: OCC.Geom.BSplineCurve
        The Bézier curve to convert.

    Returns
    -------
    wire: OCC Wire object
        Le cable
    """
    try:
        edge = BRepBuilderAPI_MakeEdge(curve.GetObject().GetHandle()).Edge()
    except AttributeError:
        try:
            edge = BRepBuilderAPI_MakeEdge(curve.GetHandle()).Edge()
        except AttributeError:
            edge = BRepBuilderAPI_MakeEdge(curve).Edge()
    return BRepBuilderAPI_MakeWire(edge).Wire()


def to_NURBS(shape):  # noqa :N802
    """
    Convert shape to NURBS spline
    """
    return BRepBuilderAPI_NurbsConvert(shape).Shape()


# =============================================================================
# Conversions
# =============================================================================


def point_array_to_TColgp_PntArrayType(array, typer=TColgp_Array1OfPnt):  # noqa :N802
    """
    Creates a curve from a numpy array

    Parameters
    ----------
    array : array (Npts x 3) or list
        Array of xyz points for which to fit a bspline
    typer : type of TColgp array
            - TColgp_Array1OfPnt
            - TColgp_HArray1OfPnt

    Returns
    -------
    pt_arr : TCOLgp_Array1OfPnt
        OCC type array of points

    Note
    ----
    Use TColgp_Harray when interpolating a curve from points with the
    GeomAPI_Interpolate. Use TColgp_Array when interpolating a curve
    from points with the GeomAPI_PointsToBspline
    """
    dims = np.shape(array)
    if (dims[0] == 3) and (dims[1] != 3):
        # Detect transposed Loop and quietly handle
        array = array.T
    elif dims[1] != 3:
        raise CADError("Array must have dimension Npnts x 3 (x, y, z)")
    n_points = np.shape(array)[0]
    pt_arr = typer(1, n_points)
    for i, pt in enumerate(array):
        pt_arr.SetValue(i + 1, gp_Pnt(*pt.tolist()))
    return pt_arr


def _points_to_TColgp_Array(pointlist):  # noqa :N802
    """
    Parameters
    ----------
    pointlist: list(*gp_Pnt2d)
        List of points to be converted

    Returns
    -------
    pntarray: TColgp_Array1OfPnt2d
        OCC god knows what
    """
    return _Tcol_dim_1(pointlist, TColgp_Array1OfPnt2d)


def _Tcol_dim_1(pointlist, _type):  # noqa :N802
    """
    Function factory for 1-dimensional TCol* types
    """
    pts = _type(0, len(pointlist) - 1)
    for n, i in enumerate(pointlist):
        pts.SetValue(n, i)
    return pts


def _make_OCCedge(*args):  # noqa :N802
    edge = BRepBuilderAPI_MakeEdge(*args)
    return edge.Edge()


def _make_OCCwire(*args):  # noqa :N802
    if isinstance(args[0], (list, tuple)):
        wire = BRepBuilderAPI_MakeWire()
        for i in args[0]:
            wire.Add(i)
        wire.Build()
        return wire.Wire()
    wire = BRepBuilderAPI_MakeWire(*args)
    return wire.Wire()


def _make_OCCface(*args):  # noqa :N802
    """
    Makes an OCC Face object from an OCC Wire object

    Parameters
    ----------
    args: TopoDS_Wire
        The wire from which a face must be made

    Returns
    -------
    face: OCC Face object
    """
    face = BRepBuilderAPI_MakeFace(*args)
    return face.Face()


def _make_OCCsolid(*args):  # noqa :N802
    """
    Makes an OCC Solid object from topods objects

    Parameters
    ----------
    args: TopoDS_*
        The topods objects to make a solid from

    Returns
    -------
    solid: OCC Solid object
    """
    solid = BRepBuilderAPI_MakeSolid(*args)
    return solid.Solid()


# =============================================================================
# Boolean operations
# =============================================================================


def boolean_cut(shape, cutshape):
    """
    Boolean cut operation.

    Parameters
    ----------
    shape: OCC Shape object
        The shape to cut from
    cutshape: OCC Shape object
        The shape to cut with

    Returns
    -------
    shp: OCC Shape object
        The result of the boolean cut operation
    """
    cut = BRepAlgoAPI_Cut(shape, cutshape)
    try:
        cut.RefineEdges()
        cut.FuseEdges()
    except AttributeError:
        pass
    shp = cut.Shape()
    try:
        cut.Delete()
    except AttributeError:
        pass
    return shp


def boolean_fuse(shape1, shape2):
    """
    Boolean fuse operation.

    Parameters
    ----------
    shape1: OCC Shape object
        The base shape to fuse from
    shape2: OCC Shape object
        The shape to fuse with

    Returns
    -------
    shp: OCC Shape object
        The result of the boolean cut operation
    """
    join = BRepAlgoAPI_Fuse(shape1, shape2)
    try:
        join.RefineEdges()
        join.FuseEdges()
    except AttributeError:
        pass
    shp = join.Shape()
    try:
        join.Delete()
    except AttributeError:
        pass
    return shp
