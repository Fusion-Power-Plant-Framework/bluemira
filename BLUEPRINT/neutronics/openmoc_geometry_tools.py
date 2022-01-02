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
Tools for integrating geometry information into OpenMOC
"""

import numpy as np
import openmoc

# BLUEPRINT imports
from BLUEPRINT.geometry.geomtools import polyarea

###########################################
# Tools for manipulating points and planes.
###########################################


def get_plane_properties_from_points(point1, point2):
    """
    Get the plane properties from two points

    The plane properties are defined by the following equation:
        a * x + b * y + c * z + d = 0

    Assumes 2-D, so z multiplier is always zero.

    Parameters
    ----------
    point1 : (float, float)
        The first point.
    point2 : (float, float)
        The second point.

    Returns
    -------
    a : float
        The x multiplier.
    b : float
        The y multiplier.
    c : float
        The z multiplier (zero for a 2-D plane).
    d : float
        The offset.
    """
    b = point2[0] - point1[0]
    a = point1[1] - point2[1]
    c = 0.0
    d = -1.0 * (a * point1[0] + b * point1[1])
    return a, b, c, d


def get_plane_properties(plane):
    """
    Get the properties of a plane

    The plane properties are defined by the following equation:
        a * x + b * y + c * z + d = 0

    Assumes 2-D, so z multiplier is always zero.

    Parameters
    ----------
    plane : `openmoc.Plane`
        The plane.

    Returns
    -------
    a : float
        The x multiplier.
    b : float
        The y multiplier.
    c : float
        The z multiplier (zero for a 2-D plane).
    d : float
        The offset.
    """
    c = 0
    if isinstance(plane, openmoc.XPlane):
        a = 1
        b = 0
        d = -plane.getX()
    elif isinstance(plane, openmoc.YPlane):
        a = 0
        b = 1
        d = -plane.getY()
    else:
        a = plane.getA()
        b = plane.getB()
        d = plane.getD()
    return a, b, c, d


def get_normalised_plane_properties(plane):
    """
    Get the normalised plane properties

    The normalised plane properties are defined in 2-D by the following equation:

    y = gradient * x + intercept

    This allows co-planar lines to be identified in a normalised manner.

    Parameters
    ----------
    plane : `openmoc.Plane`
        The plane.

    Returns
    -------
    gradient : float
        The dx / dy gradient. If the plane is parallel with the y-axis then inf.
    intercept : float
        The y-axis intercept if the plane is not parallel with the y-axis.
        The x-axis intercept if the plane is parallel with the y-axis.
    """
    a, b, _, d = get_plane_properties(plane)
    if np.isclose(b, 0.0):
        gradient = float("inf")
        if np.isclose(a, 0.0):
            intercept = float("inf")
        else:
            intercept = -d / a
    else:
        gradient = -a / b
        intercept = -d / b
    return gradient, intercept


def evaluate_point_against_plane(plane, point):
    """
    Evaluate the point against the plane

    Calculates the value of the point on the plane according to
    value = a * x + b * y + c * z + d

    Note that in 2-D the value of c is always zero.

    Parameters
    ----------
    plane : `openmoc.Plane`
        The plane.
    point : (float, float)
        The point.

    Returns
    -------
    float
        The point evaluated on the plane.
    """
    a = plane.getA()
    b = plane.getB()
    d = plane.getD()
    return a * point[0] + b * point[1] + d


def get_halfspace(plane, point):
    """
    Get the halfspace occupied by a point against a plane

    Parameters
    ----------
    plane : `openmoc.Plane`
        The plane.
    point : (float, float)
        The point.

    Returns
    -------
    int
        The halfspace occupied by the point against the plane.
    """
    return +1 if evaluate_point_against_plane(plane, point) > 0 else -1


class PlaneHelper:
    """A class to act as a cache of Planes to eliminate any coplanar lines."""

    class _PlaneKey:
        """A class to generate a consistently hashed plane key for 2-D coordinates."""

        DECIMAL_PLACES = 8

        def __init__(self, gradient, intercept):
            self.gradient = gradient
            self.intercept = intercept

        def __eq__(self, other):
            return (
                round(self.gradient, self.DECIMAL_PLACES)
                == round(other.gradient, self.DECIMAL_PLACES)
            ) and (
                round(self.intercept, self.DECIMAL_PLACES)
                == round(other.intercept, self.DECIMAL_PLACES)
            )

        def __hash__(self):
            return hash(
                (
                    round(self.gradient, self.DECIMAL_PLACES),
                    round(self.intercept, self.DECIMAL_PLACES),
                )
            )

    def __init__(self):
        self.planes = {}

    def add_plane(self, plane: openmoc.Plane) -> None:
        """
        Add a plane to the cache

        Parameters
        ----------
        plane : openmoc.Plane
            The plane.
        """
        gradient, intercept = get_normalised_plane_properties(plane)
        if self._PlaneKey(gradient, intercept) not in self.planes:
            self.planes[self._PlaneKey(gradient, intercept)] = plane

    def find_plane(self, plane: openmoc.Plane) -> openmoc.Plane:
        """
        Find a plane in the cache and return it

        Parameters
        ----------
        plane : openmoc.Plane
            The plane to find.

        Returns
        -------
        openmoc.Plane
            The plane pulled from the cache.
        """
        gradient, intercept = get_normalised_plane_properties(plane)
        return self.planes[self._PlaneKey(gradient, intercept)]


#################################
# Triangulation and meshing tools
#################################


def calc_triangle_centroid(point1, point2, point3):
    """
    Calculate the centroid of three points

    Parameters
    ----------
    point1 : (float, float]
        The first point.
    point2 : (float, float)
        The second point.
    point3 : (float, float)
        The third point.

    Returns
    -------
    Tuple[float, float]
        The centroid of the three points.
    """
    x = (point1[0] + point2[0] + point3[0]) / 3.0
    y = (point1[1] + point2[1] + point3[1]) / 3.0
    return x, y


def create_system_cells(universe, sections, name, material, plane_helper):
    """
    Create the cells for the system comprising the provided sections and add to universe

    Parameters
    ----------
    universe : `openmoc.Universe`
        The OpenMOC Universe that the cells will be added to.
    sections : List[`sectionproperties.analysis.cross_section.CrossSection`]
        The `CrossSection` objects representing the system.
    name : str
        The name of the system, as used by OpenMOC to build a dictionary of cells.
    material : `openmoc.Material`
        The material to fill the cells with.
    plane_helper : PlaneHelper
        The cache of planes to draw from.

    Returns
    -------
    system_cells : List[`openmoc.Cell`]
        The `Cell` objects corresponding to this system.
    """
    system_cells = []
    for section_idx, section in enumerate(sections):
        for element_idx, element in enumerate(section.mesh_elements):
            # Get the triangular elements as points
            points = [section.mesh_nodes[node_idx] for node_idx in element[:3]]

            # Get the centroid of this element
            cell_centroid = calc_triangle_centroid(*points)

            # Build the cell
            cell = openmoc.Cell(name=f"{name}_{section_idx}_{element_idx}")

            # Generate the planes bounding this element and update cache
            for point_idx, point in enumerate(points):
                plane = openmoc.Plane(
                    *get_plane_properties_from_points(
                        point, points[(point_idx + 1) % len(points)]
                    ),
                    name=f"{name}_{section_idx}_{element_idx}_{point_idx}",
                )
                plane_helper.add_plane(plane)
                plane = plane_helper.find_plane(plane)
                cell.addSurface(
                    halfspace=get_halfspace(plane, cell_centroid),
                    surface=plane,
                )

            # Make sure the cell is filled
            cell.setFill(material)

            # Add the cell to the universe
            universe.addCell(cell)

            system_cells += [cell]
    return system_cells


def get_source_fsr_map(geometry, source_cells):
    """
    Get the FSR map for the source cells in the specified geometry

    The FSR map provides the link between the UID for the source cells and the
    corresponding FSR ID. This allows a cache to be built that links cells and
    FSRs for more efficient lookup.

    Parameters
    ----------
    geometry : openmoc.Geometry
        The geometry in which the cells can be found
    source_cells : List[openmoc.Cell]
        The cells corresponding to the source.

    Returns
    -------
    far_map : Dict[int, int]
        The mapping between the cell's UID and the corresponding FSR cell.
    """
    source_fsr_map = {}

    num_fsr = geometry.getNumFSRs()
    for fsr_id in range(num_fsr):
        fsr_cell = geometry.findCellContainingFSR(fsr_id)
        for source_cell in source_cells:
            if source_cell.getId() == fsr_cell.getId():
                source_fsr_map[source_cell.getUid()] = fsr_id

    return source_fsr_map


def _find_openmoc_cell_at_point(universe, point):
    """
    Find the OpenMOC cell in the universe that contains the specified point

    Parameters
    ----------
    universe : openmoc.Universe
        The OpenMOC universe to seach for the cell.
    point : (float, float)
        The point that the resulting cell should contain.

    Returns
    -------
    openmoc_cell : openmoc.Cell
        The cell in the universe that contains the point.
    """
    return universe.findCell(openmoc.LocalCoords(*point))


def _calc_points_area(points):
    """
    Calculate the area of the space bounded by the points

    Parameters
    ----------
    points : List[float, float]
        The points bounding the area to be calculated.

    Returns
    -------
    area : float
        The area bound by the points.
    """
    return polyarea(*np.array(points).T)


def _calc_circumference(radius):
    """
    Calculate the circumference at the specified radius

    Parameters
    ----------
    radius : float
        The radius.

    Returns
    -------
    circumference : float
        The circumference.
    """
    return 2 * np.pi * radius


def populate_source_cells(geometry, solver, source, source_cells, source_sections):
    """
    Populate the source cells in the solver

    Scans the elements in the provided source sections to find cells in the geometry.
    Then populates the corresponding FSR with the source strength at the centroid of the
    cell.

    Parameters
    ----------
    geometry : openmoc.Geometry
        The OpenMOC geometry from which universe and FSRs will be obtained.
    solver : openmoc.Solver
        The solver being used for the OpenMOC calculation.
    source : PlasmaSource
        The source from which the source strength will be obtained.
    source_cells : List[openmoc.Cell]
        The cells containing the source.
    source_sections : List[sectionproperties.analysis.cross_section.CrossSection]
        The cross sections corresponding to the source mesh.
    """
    # Build a cache of FSRs corresponding to plasma cells
    source_fsr = get_source_fsr_map(geometry, source_cells)

    universe = geometry.getRootUniverse()
    for section in source_sections:
        for element_idx, element in enumerate(section.mesh_elements):
            # Get the triangular elements for the cell as points
            points = [section.mesh_nodes[node_idx] for node_idx in element[:3]]

            # Find the cell in the universe
            centroid = calc_triangle_centroid(*points)
            source_cell = _find_openmoc_cell_at_point(universe, centroid)

            # Get the source strength and set the source in the FSR
            source_strength = source.get_source_strength_xz(*centroid)
            if not np.isnan(source_strength):
                # Get the ID of the FSR cell from the cache
                fsr_id = source_fsr[source_cell.getUid()]

                # Get the cell properties for normalisation
                area = _calc_points_area(points)
                circumfrence = _calc_circumference(centroid[0])

                # Normalise to the cell volume
                norm_source_strength = source_strength * area * circumfrence
                solver.setFixedSourceByFSR(fsr_id, 1, norm_source_strength)
