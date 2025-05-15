# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Supporting functions for the bluemira geometry module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np
from cadquery.assembly import Assembly
from cadquery.occ_impl import geom, shapes
from cadquery.occ_impl.assembly import Color
from cadquery.vis import show, style
from matplotlib import colors

from bluemira.codes.error import CadQueryError
from bluemira.utilities.tools import ColourDescriptor

if TYPE_CHECKING:
    from bluemira.display.palettes import ColorPalette

apiVertex = shapes.Vertex  # noqa: N816
apiVector = geom.Vector  # noqa: N816
apiEdge = shapes.Edge  # noqa: N816
apiWire = shapes.Wire  # noqa: N816
apiFace = shapes.Face  # noqa: N816
apiShell = shapes.Shell  # noqa: N816
apiSolid = shapes.Solid  # noqa: N816
apiShape = shapes.Shape  # noqa: N816
# apiSurface = Part.BSplineSurface
# apiPlacement = Base.Placement
apiPlane = shapes.Plane  # noqa: N816
apiCompound = shapes.Compound  # noqa: N816

WORKING_PRECISION = 1e-5
MIN_PRECISION = 1e-5
MAX_PRECISION = 1e-5
ONE_PERIOD = 2 * np.pi

# ======================================================================================
# Array, List, Vector, Point manipulation
# ======================================================================================


def vector_to_list(vectors: list[apiVector]) -> list[list[float]]:
    """Converts a FreeCAD Base.Vector or list(Base.Vector) into a list"""  # noqa: DOC201
    return [list(v) for v in vectors]


def vector_to_numpy(vectors: list[apiVector]) -> np.ndarray:
    """Converts a FreeCAD Base.Vector or list(Base.Vector) into a numpy array"""  # noqa: DOC201
    return np.array(vector_to_list(vectors))


# ======================================================================================
# Geometry creation
# ======================================================================================


def make_solid(shell: apiShell) -> apiSolid:
    """Make a solid from a shell."""  # noqa: DOC201
    return apiSolid.makeSolid(shell)


def make_shell(faces: list[apiFace]) -> apiShell:
    """Make a shell from faces."""  # noqa: DOC201
    return apiShell.makeShell(faces)


def make_compound(shapes: list[apiShape]) -> apiCompound:
    """
    Make a CadQuery compound object out of many shapes

    Parameters
    ----------
    shapes:
        A set of objects to be compounded

    Returns
    -------
    A compounded set of shapes
    """
    return apiCompound.makeCompound(shapes)


def make_polygon(points: list | np.ndarray) -> apiWire:
    """
    Make a polygon from a set of points.

    Parameters
    ----------
    points:
        list of points. It can be given as a list of 3D tuples, a 3D numpy array,
        or similar.

    Returns
    -------
    :
        A FreeCAD wire that contains the polygon
    """
    # Points must be converted into FreeCAD Vectors
    pntslist = [apiVector(x) for x in points]
    return apiWire.makePolygon(pntslist)


def make_bezier(points: list | np.ndarray) -> apiWire:
    """
    Make a bezier curve from a set of points.

    Parameters
    ----------
    points:
        list of points. It can be given as a list of 3D tuples, a 3D numpy array,
        or similar.

    Returns
    -------
    :
        A FreeCAD wire that contains the bezier curve
    """
    # Points must be converted into FreeCAD Vectors
    pntslist = [apiVector(x) for x in points]
    return apiWire.assembleEdges(apiEdge.makeBezier(pntslist))


def make_bspline(
    poles: npt.ArrayLike,
    mults: npt.ArrayLike,
    knots: npt.ArrayLike,
    *,
    periodic: bool,
    degree: int,
    weights: npt.ArrayLike,
    check_rational: bool,
) -> apiWire:
    """
    Builds a B-Spline by a lists of Poles, Mults, Knots

    Parameters
    ----------
    poles:
        list of poles.
    mults:
        list of integers for the multiplicity
    knots:
        list of knots
    periodic:
        Whether or not the spline is periodic (same curvature at start and end points)
    degree: int
        bspline degree
    weights:
        sequence of float
    check_rational:
        Whether or not to check if the BSpline is rational (not sure)

    Returns
    -------
    :
        A FreeCAD wire that contains the bspline curve

    Notes
    -----
    This function wraps the FreeCAD function of bsplines buildFromPolesMultsKnots
    """
    raise NotImplementedError


def make_bsplinesurface(
    poles: npt.ArrayLike,
    mults_u: npt.ArrayLike,
    mults_v: npt.ArrayLike,
    knot_vector_u: npt.ArrayLike,
    knot_vector_v: npt.ArrayLike,
    degree_u: int,
    degree_v: int,
    weights: npt.ArrayLike,
    *,
    periodic: bool = False,
    check_rational: bool = False,
) -> apiSurface:
    """
    Builds a B-SplineSurface by a lists of Poles, Mults, Knots

    Parameters
    ----------
    poles:
        poles (sequence of Base.Vector).
    mults_u:
        list of integers for the u-multiplicity
    mults_v:
        list of integers for the u-multiplicity
    knot_vector_u:
        list of u-knots
    knot_vector_v:
        list of v-knots
    degree_u:
        degree of NURBS in u-direction
    degree_v:
        degree of NURBS in v-direction
    weights:
        pole weights (sequence of float).
    periodic:
        Whether or not the spline is periodic (same curvature at start and end points)
    check_rational:
        Whether or not to check if the BSpline is rational (not sure)

    Returns
    -------
    :
        A FreeCAD object that contours the bsplinesurface

    Notes
    -----
    This function wraps the FreeCAD function of bsplinesurface buildFromPolesMultsKnots
    """
    raise NotImplementedError


def interpolate_bspline(
    points: list | np.ndarray,
    *,
    closed: bool = False,
    start_tangent: Iterable | None = None,
    end_tangent: Iterable | None = None,
) -> apiWire:
    """
    Make a B-Spline curve by interpolating a set of points.

    Parameters
    ----------
    points:
        list of points. It can be given as a list of 3D tuples, a 3D numpy array,
        or similar.
    closed:
        if True, the first and last points will be connected in order to form a
        closed shape.
    start_tangent:
        Tangency of the BSpline at the first pole. Must be specified with end_tangent
    end_tangent:
        Tangency of the BSpline at the last pole. Must be specified with start_tangent

    Returns
    -------
    :
        A FreeCAD wire that contains the bspline curve

    Raises
    ------
    InvalidCADInputsError
        Not enough points to interpolate
    FreeCADError
        Unable to make spline
    """
    raise NotImplementedError


def make_circle_curve(radius: float, center: apiVector, axis: apiVector) -> Part.Circle:
    """
    Make a Part.Circle with a consistent .Rotation property, by initializing a circle of
    the default size, position and orientation at first.

    Parameters
    ----------
    radius:
        radius of the circle [m]
    center:
        center of the circle [m]
    axis:
        Normalised vector around which the circle spins counter-clockwise.

    Returns
    -------
    circle:
        Part.Circle created by FreeCAD.
    """
    return apiWire.makeCircle(radius, center, axis)


def make_circle(
    radius: float = 1.0,
    center: Iterable[float] = [0.0, 0.0, 0.0],
    start_angle: float = 0.0,
    end_angle: float = 360.0,
    axis: Iterable[float] = [0.0, 0.0, 1.0],
) -> apiWire:
    """
    Create a circle or arc of circle object with given parameters.

    Parameters
    ----------
    radius:
        Radius of the circle
    center:
        Center of the circle
    start_angle:
        Start angle of the arc [degrees]
    end_angle:
        End angle of the arc [degrees]. If start_angle == end_angle, a circle is created,
        otherwise a circle arc is created
    axis:
        Normal vector to the circle plane. It defines the clockwise/anticlockwise
        circle orientation according to the right hand rule. Default [0., 0., 1.].

    Returns
    -------
    :
        FreeCAD wire that contains the arc or circle
    """
    if np.isclose(start_angle, end_angle % 360):
        return make_circle_curve(radius, apiVector(center), apiVector(axis))
    raise NotImplementedError


def make_circle_arc_3P(  # noqa: N802
    p1: Iterable[float],
    p2: Iterable[float],
    p3: Iterable[float],
    axis: Iterable[float] | None = None,
) -> apiWire:
    """
    Create an arc of circle object given three points.

    Parameters
    ----------
    p1:
        Starting point of the circle arc
    p2:
        Middle point of the circle arc
    p3:
        End point of the circle arc

    Returns
    -------
    :
        FreeCAD wire that contains the arc of circle

    Raises
    ------
    FreeCADError
        Raised if the three points are collinear.
    """
    return apiWire.assembleEdges(
        apiEdge.makeThreePointArc(apiVector(p1), apiVector(p2), apiVector(p3))
    )


def make_ellipse(
    center: Iterable[float] = [0.0, 0.0, 0.0],
    major_radius: float = 2.0,
    minor_radius: float = 1.0,
    major_axis: Iterable[float] = [1, 0, 0],
    minor_axis: Iterable[float] = [0, 1, 0],
    start_angle: float = 0.0,
    end_angle: float = 360.0,
) -> apiWire:
    """
    Creates an ellipse or arc of ellipse object with given parameters.

    Parameters
    ----------
    center:
        Center of the ellipse
    major_radius:
        the major radius of the ellipse
    minor_radius:
        the minor radius of the ellipse
    major_axis:
        major axis direction
    minor_axis:
        minor axis direction
    start_angle:
        Start angle of the arc [degrees]
    end_angle:
        End angle of the arc [degrees]. If start_angle == end_angle, an ellipse is
        created, otherwise an arc of ellipse is created

    Returns
    -------
    :
        FreeCAD wire that contains the ellipse or arc of ellipse
    """
    s1 = apiVector(major_axis).normalize().multiply(major_radius) + apiVector(center)
    s2 = apiVector(minor_axis).normalize().multiply(minor_radius) + apiVector(center)
    center = apiVector(center)
    start_angle %= 360.0
    end_angle %= 360.0
    return apiWire.makeEllipse(
        s1,
        s2,
        center,
        major_axis,
        minor_axis,
        start_angle,
        end_angle,
        closed=np.isclose(start_angle, end_angle),
    )


class JoinType(enum.Enum):
    """See Part/PartEnums.py, its not importable"""

    Arc = "arc"
    Tangent = "tangent"
    Intersect = "intersection"


def offset_wire(
    wire: apiWire, thickness: float, join: str = "intersect", *, open_wire: bool = True
) -> apiWire:
    """
    Make an offset from a wire.

    Parameters
    ----------
    wire:
        Wire to offset from
    thickness:
        Offset distance. Positive values outwards, negative values inwards
    join:
        Offset method. "arc" gives rounded corners, and "intersect" gives sharp corners
    open_wire:
        For open wires (counter-clockwise default) whether or not to make an open offset
        wire, or a closed offset wire that encompasses the original wire. This is
        disabled for closed wires.

    Returns
    -------
    :
        Offset wire

    Raises
    ------
    InvalidCADInputsError
        Wire must be planar and cannot be straight
    FreeCADError
        offset failed
    """
    if thickness == 0.0:
        return wire.copy()

    if _wire_is_straight(wire):
        raise InvalidCADInputsError("Cannot offset a straight line.")

    if not _wire_is_planar(wire):
        raise InvalidCADInputsError("Cannot offset a non-planar wire.")

    f_join = JoinType[join.lower().capitalize()]

    if wire.isClosed() and open_wire:
        open_wire = False

    wire.offset2D(thickness, f_join.value)
    if not wire.isClosed() and not open_wire:
        raise CadQueryError("offset failed to close wire")
    return wire


def make_face(wire: apiWire) -> apiFace:
    """
    Make a face given a wire boundary.

    Parameters
    ----------
    wire:
        Wire boundary from which to make a face

    Returns
    -------
    :
        Face created from the wire boundary

    Raises
    ------
    FreeCADError
        If the created face is invalid
    """
    face = apiFace.makeFromWires(wire)
    if face.isValid():
        return face
    face.fix(WORKING_PRECISION, MIN_PRECISION, MAX_PRECISION)
    if face.isValid():
        return face
    raise CadQueryError("An invalid face has been generated")


# ======================================================================================
# Object properties
# ======================================================================================
def _get_api_attr(obj: apiShape, prop: str):
    try:
        return getattr(obj, prop)
    except AttributeError:
        raise FreeCADError(
            f"FreeCAD object {obj} does not have an attribute: {prop}"
        ) from None


def length(obj: apiShape) -> float:
    """Object's length"""  # noqa: DOC201
    return _get_api_attr(obj, "Length")


def area(obj: apiShape) -> float:
    """Object's Area"""  # noqa: DOC201
    return _get_api_attr(obj, "Area")


def volume(obj: apiShape) -> float:
    """Object's volume"""  # noqa: DOC201
    return _get_api_attr(obj, "Volume")


# ======================================================================================
# Geometry visualisation
# ======================================================================================


@dataclass
class DefaultDisplayOptions:
    """CadQuery default display options"""

    colour: ColourDescriptor = ColourDescriptor()
    transparency: float = 1.0
    tolerance: float = 1e-2
    edges: bool = True
    mesh: bool = False
    specular: bool = True
    markersize: float = 5
    linewidth: float = 2
    spheres: bool = False
    tubes: bool = False
    edgecolor: str = "black"
    meshcolor: str = "lightgrey"
    vertexcolor: str = "cyan"

    @property
    def color(self) -> str:
        """See colour"""
        return self.colour

    @color.setter
    def color(self, value: str | tuple[float, float, float] | ColorPalette):
        """See colour"""
        self.colour = value


def show_cad(
    parts: apiShape | list[apiShape],
    options: dict | list[dict | None] | None = None,
    labels: list[str] | None = None,
    **kwargs,
):
    if isinstance(parts, apiShape):
        parts = [parts]

    if options is None:
        options = [None] * len(parts)

    if labels is None:
        labels = [None] * len(parts)

    if len(options) != len(parts) != len(labels):
        raise CadQueryError(
            "If options for display or labels are provided then there must be as "
            "many as there are parts to display."
        )

    options = [{**asdict(DefaultDisplayOptions()), **(o or {})} for o in options]

    show(
        *(
            style(
                Assembly(
                    part,
                    name=label,
                    color=Color(
                        *colors.to_rgba(c=op.pop("colour"), alpha=op.pop("transparency"))
                    ),
                ),
                **op,
            )
            for part, op, label in zip(parts, options, labels, strict=False)
        ),
        title="Bluemira Display",
        **kwargs,
    )
