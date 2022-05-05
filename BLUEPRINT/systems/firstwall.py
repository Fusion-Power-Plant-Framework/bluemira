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
Flux surface attributes and first wall profile based on heat flux calculation
"""

from typing import Type

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.parameter import ParameterFrame
from bluemira.equilibria.find import find_flux_surface_through_point, find_flux_surfs
from bluemira.geometry._deprecated_loop import Loop
from bluemira.geometry._deprecated_tools import get_intersect, loop_plane_intersect
from bluemira.geometry.error import GeometryError
from bluemira.radiation_transport.advective_transport import ChargedParticleSolver
from BLUEPRINT.base.error import SystemsError
from BLUEPRINT.cad.firstwallCAD import FirstWallCAD
from BLUEPRINT.geometry.boolean import (
    boolean_2d_common_loop,
    boolean_2d_difference,
    boolean_2d_difference_loop,
    boolean_2d_difference_split,
    boolean_2d_union,
    convex_hull,
    simplify_loop,
)
from BLUEPRINT.geometry.geombase import make_plane
from BLUEPRINT.geometry.geomtools import (
    clean_loop_points,
    index_of_point_on_loop,
    make_box_xz,
    rotate_vector_2d,
)
from BLUEPRINT.geometry.offset import offset_clipper
from BLUEPRINT.geometry.shell import Shell
from BLUEPRINT.systems.baseclass import ReactorSystem
from BLUEPRINT.systems.plotting import ReactorSystemPlotter
from BLUEPRINT.utilities.csv_writer import write_csv


def find_outer_point(point_list, x_compare):
    """
    Return the first point in the list which has an x coordinate
    that excedes the comparison value
    """
    return next(filter(lambda p: p[0] > x_compare, point_list))


def find_inner_point(point_list, x_compare):
    """
    Return the first point in the list which has an x coordinate
    that is less than the comparison value
    """
    return next(filter(lambda p: p[0] < x_compare, point_list))


def get_intersection_point(point, norm, loop, compare_x, inner=True):
    """
    Return the point of intersection between the given loop and
    a plane with given norm and that axis at the given point

    Parameters
    ----------
    point : float
        Coord of plane in direction of given norm used to intersect loop
    norm: int
        Direction of plane normal (0 = x, 2 = z)
    loop: Loop
        Loop with which to intersect x-y plane
    compare_x : float
        x-coordinate used to compare intersection points and yield
        inner or outer point
    inner : bool
        Flag to control whether inner or outer point is returned
    """
    # Create a plane with given normal and intersecting axis at point
    cut_plane = make_plane(point, norm)

    # Find intersection between the plane and separatrix
    plane_ints = loop_plane_intersect(loop, cut_plane)

    # Return the inner or outer intersection using the given x point
    # to compare against
    if inner:
        return find_inner_point(plane_ints, compare_x)
    else:
        return find_outer_point(plane_ints, compare_x)


def get_tangent_vector(point_on_loop, loop):
    """
    Find the normalised tangent vector to the given loop at the given
    point on the loop.

    Parameters
    ----------
    point_on_loop: [float,float]
        List of [x,z] coords corresponding to the strike point position
    loop: Loop
        The loop that was used to find the outer stike point, from which
        another point will be taken

    Returns
    -------
    tangent_norm: numpy.array
        Vector in the x-z plane representing the tangent to the loop
        at the given point
    """
    # Retrieve coordinates of the point just before where
    # the loop intersects the given point
    i_before = index_of_point_on_loop(loop, point_on_loop, before=True)
    p_before = [loop.x[i_before], loop.z[i_before]]

    # Create a tangent to the flux loop at the strike point
    tangent = np.array(p_before) - np.array(point_on_loop)

    # Return normalised tangent vector
    tangent_norm = tangent / np.linalg.norm(tangent)

    # Fixing the tangent vector direction to be concordant with the x-axis
    if tangent_norm[0] > 0:
        tangent_norm = -tangent_norm

    return tangent_norm


def make_guide_line(initial_loop, top_limit, bottom_limit):
    """
    Cuts a portion of an initial loop that will work as guide line
    for a more complex shape (e.g. divertor leg).

    Parameters
    ----------
    initial_loop: loop
        Initial loop to cut
    top_limit: [float,float]
        Coordinates of the top limit where the loop will be cut
    bottom_limit: [float,float]
        Coordinates of the bottom limit where the loop will be cut

    Returns
    -------
    guide_line: loop
        portion of the initial loop that will work as guide line
    """
    # Select those points along the initial loop below
    # the top limit
    top_clip_guide_line = np.where(
        initial_loop.z < top_limit[1],
    )
    # Create a new Loop from the points selected along the
    # initial loop
    cut_loop = Loop(
        x=initial_loop.x[top_clip_guide_line],
        y=None,
        z=initial_loop.z[top_clip_guide_line],
    )
    # Select those points along the top-clipped loop above
    # the bottom limit
    bottom_clip_guide_line = np.where(
        cut_loop.z > bottom_limit[1],
    )
    # Create a new Loop from the points selected along the
    # previously top-clipped loop
    guide_line = Loop(
        x=cut_loop.x[bottom_clip_guide_line],
        y=None,
        z=cut_loop.z[bottom_clip_guide_line],
    )
    return guide_line


def make_flux_contour_loops(eq, psi_norm):
    """
    Return an ordered list of loops corresponding to flux contours having
    the given (normalised) psi value.
    The ordering is in decreasing value of min x coord in loop to be
    consisent with convention used in self.separatrix.

    Parameters
    ----------
    eq : Equilibirum
        Equilibrium from which to take the flux field.
    psi_norm : float
        Value normalised flux field to use to define the contours.

    Returns
    -------
    flux_loops : list of Loop
        List of flux contours as Loops
    """
    # Flux field
    psi = eq.psi()

    # Get the contours
    flux_surfs = find_flux_surfs(eq.x, eq.z, psi, psi_norm)

    # Create a dictionary of loops indexed in min x
    flux_dict = {}
    for surf in flux_surfs:
        flux_x = surf[:, 0]
        flux_z = surf[:, 1]
        min_x = np.min(flux_x)
        flux_dict[min_x] = Loop(x=flux_x, y=None, z=flux_z)

    # Sort the dictionary
    sorted_dict = dict(sorted(flux_dict.keys()))

    # Get list of increasing values
    sorted_flux_loops = list(sorted_dict.values())[::-1]

    return sorted_flux_loops


def reshape_curve(
    curve_x_coords,
    curve_z_coords,
    new_starting_point,
    new_ending_point,
    degree,
):
    """
    Force a curve between two new points
    Used to shape the divertor legs following the separatrix curvature
    Mostly useful for the Super-X configuration

    Parameters
    ----------
    curve_x_coords: [float]
        x coordinates of the leading curve (the separatrix)
    curve_z_coords: [float]
        z coordinates of the leading curve (the separatrix)
    new_starting_point: [float, float]
        x and z coordinates of the starting point of the new curve
        (contour of the divertor legs)
    new_ending_point: [float, float]
        x and z coordinates of the ending point of the new curve
    degree: [float]
        Degree of the fitting polynomial. The longer is the leg the harder
        is to control the shape. Changing this value can help

    Returns
    -------
    x: [float]
        x coordinate of points in the new curve
    z: [float]
        z coordinate of points in the new curve
    """
    coeffs = np.polyfit(curve_x_coords, curve_z_coords, degree)
    func = np.poly1d(coeffs)

    new_a_coeff = (new_ending_point[1] - new_starting_point[1]) / (
        func(new_ending_point[0]) - func(new_starting_point[0])
    )
    new_b_coeff = new_starting_point[1] - new_a_coeff * func(new_starting_point[0])

    x = np.linspace(new_starting_point[0], new_ending_point[0], 10)
    z = new_a_coeff * (func(x)) + new_b_coeff
    return (x, z)


def get_non_overlapping(inboard, outboard, divertor_loops, cutters):
    """
    Remove overlaps between offset inboard, outboard, and divertor loops
    by clipping vertically at x point and vertically at the horizontal
    edge of the original divertor loops.

    Parameters
    ----------
    inboard : Loop
        Loop representing the inboard portion of first wall minus divertor.
    outboard : Loop
        Loop representing the outboard portion of first wall minus divertor.
    divertor_loops : list
        List of Loop objects representing the divertor(s).
    cutters: tuple
        Tuple of cutters used to remove overlaps.

    Returns
    -------
    sections : list
        List of Loop objects corresponding to the non-overlapping sections
        of the first wall. Ordering is divertor_loops; inboard; outboard.
    """
    # Extract cutters
    cutter_in = cutters[0]
    cutter_out = cutters[1]
    div_cutters = cutters[2]

    # Cut vertically at the x point
    inboard = boolean_2d_common_loop(inboard, cutter_in)
    outboard = boolean_2d_common_loop(outboard, cutter_out)

    # Cut horizontally where the inboard/outboard meets divertor
    sections = []
    for i_div, div_cutter in enumerate(div_cutters):
        # Cut the divertor
        div = divertor_loops[i_div]
        div_clip = boolean_2d_common_loop(div, div_cutter)

        # Cut the inboard/outboard
        inboard = boolean_2d_difference_loop(inboard, div_cutter)
        outboard = boolean_2d_difference_loop(outboard, div_cutter)

        # Save clipped divertor
        sections.append(div_clip)

    # Save fully clipped inboard / outboard sections
    sections.append(inboard)
    sections.append(outboard)
    return sections


class DivertorBuilder:
    """
    Mental health assistant
    """

    def __init__(self, params, config, inputs, eq):
        self.params = params
        self.config = config
        self.inputs = inputs
        self.equilibrium = eq
        self.points = self.inputs.pop("points")

        separatrix = self.inputs.pop("separatrix")
        if isinstance(separatrix, Loop):
            separatrix = [separatrix]
        self.separatrix = separatrix

        self.inputs["SN"] = self.inputs.get("SN", False)

    def make_divertor(self, fw_loop):
        """
        Create a long legs divertor loop(s)
        usable both for SN and DN divertor

        Parameters
        ----------
        fw_loop : Loop
            first wall preliminary profile

        Returns
        -------
        divertor_loop: Loop
            Loop for the bottom divertor geometry
        """
        # Some shorthands
        z_x_point = self.points["x_point"]["z_low"]
        x_x_point = self.points["x_point"]["x"]

        # Define point where the legs should meet
        # In line with X point but with vertical offset
        middle_point = [x_x_point, z_x_point - self.params.xpt_height]

        # Find the intersection of the first wall loop and
        # the x-y plane containing the lower X point
        z_norm = 2
        fw_int_point = get_intersection_point(
            z_x_point, z_norm, fw_loop, x_x_point, inner=False
        )

        # Determine outermost point for outer divertor leg
        div_top_right = max(fw_int_point[0], x_x_point + self.params.xpt_outer_gap)

        # Determine outermost point for inner divertor leg
        div_top_left = min(fw_int_point[0], x_x_point - self.params.xpt_inner_gap)

        # Pick some flux loops to use to locate strike points and shape the
        # divertor legs
        flux_loops = self.pick_flux_loops()

        # Find the strike points
        inner_strike, outer_strike = self.find_strike_points(flux_loops)

        # Make the outer leg
        outer_leg_x, outer_leg_z = self.make_outer_leg(
            div_top_right, outer_strike, middle_point, flux_loops[0]
        )

        # Make the inner leg
        if len(flux_loops) == 1:
            inner_leg_x, inner_leg_z = self.make_inner_leg(
                div_top_left, inner_strike, middle_point, flux_loops[0]
            )
        else:
            inner_leg_x, inner_leg_z = self.make_inner_leg(
                div_top_left, inner_strike, middle_point, flux_loops[1]
            )

        # Divertor x-coords
        x_div = np.append(inner_leg_x, outer_leg_x)
        x_div = [round(elem, 5) for elem in x_div]

        # Divertor z-coords
        z_div = np.append(inner_leg_z, outer_leg_z)
        z_div = [round(elem, 5) for elem in z_div]

        # Bottom divertor loop
        bottom_divertor = Loop(x=x_div, z=z_div)
        bottom_divertor.close()

        if self.inputs["SN"]:
            return [bottom_divertor]
        else:
            # Flip z coords to get top divertor loop
            top_divertor = Loop(x=bottom_divertor.x, z=-bottom_divertor.z)
            return [bottom_divertor, top_divertor]

    def make_divertor_target(
        self, strike_point, tangent, vertical_target=True, outer_target=True
    ):
        """
        Make a divertor target

        Parameters
        ----------
        strike_point: [float,float]
            List of [x,z] coords corresponding to the strike point position

        tangent: [float,float]
            List of [x,z] coords corresponding to the tangent vector to the
        appropriate flux loop at the strike point

        Returns
        -------
        target_internal_point: [float, float]
            x and z coordinates of the internal end point of target.
            Meaning private flux region (PRF) side
        target_external_point: [float, float]
            x and z coordinates of the external end point of target.
            Meaning scrape-off layer (SOL) side

        The divertor target is a straight line
        """
        # If statement to set the function
        # either for the outer target (if) or the inner target (else)
        if outer_target:
            sign = 1
            theta_target = self.params.theta_outer_target
            target_length_pfr = self.params.tk_outer_target_pfr
            target_length_sol = self.params.tk_outer_target_sol
        else:
            sign = -1
            theta_target = self.params.theta_inner_target
            target_length_pfr = self.params.tk_inner_target_pfr
            target_length_sol = self.params.tk_inner_target_sol

        # Rotate tangent vector to appropriate flux loop to obtain
        # a vector parallel to the outer target

        # if horizontal target
        if not vertical_target:
            target_par = rotate_vector_2d(
                tangent, np.radians(180 + (theta_target * sign))
            )
        # if vertical target
        else:
            target_par = rotate_vector_2d(
                tangent, np.radians(360 - (theta_target * sign))
            )

        # Create relative vectors whose length will be the offset distance
        # from the strike point
        pfr_target_end = -target_par * target_length_pfr * sign
        sol_target_end = target_par * target_length_sol * sign

        # Add the strike point to diffs to get the absolute positions
        # of the end points of the target
        pfr_target_end = pfr_target_end + strike_point
        sol_target_end = sol_target_end + strike_point

        # Return end points
        return (pfr_target_end, sol_target_end)

    def make_outer_leg(self, div_top_right, outer_strike, middle_point, flux_loop):
        """
        Find the coordinates of the outer leg of the divertor.

        Parameters
        ----------
        div_top_right: float
            Top-right x-coordinate of the divertor
        outer_strike: [float,float]
            Coordinates of the outer strike point
        middle_point: [float,float]
            Coordinates of the middle point between the inner and outer legs
        flux_loop: Loop
            Outer flux loop used for shaping.

        Returns
        -------
        divertor_leg: (list, list)
            x and z coordinates of outer leg
        """
        # Find the tangent to the approriate flux loop at the outer strike point
        tangent = get_tangent_vector(outer_strike, flux_loop)

        # Get the outer target points
        (outer_target_pfr_end, outer_target_sol_end) = self.make_divertor_target(
            outer_strike,
            tangent,
            vertical_target=self.inputs["div_vertical_outer_target"],
            outer_target=True,
        )

        # Select the degree of the fitting polynomial and
        # the flux lines that will guide the divertor leg shape
        degree_in = self.outer_leg_pfr_polyfit_degree
        degree_out = self.outer_leg_sol_polyfit_degree
        (
            outer_leg_external_guide_line,
            outer_leg_internal_guide_line,
        ) = self.get_guide_lines(flux_loop)

        # Select the top and bottom limits for the guide lines
        z_x_point = self.points["x_point"]["z_low"]
        outer_leg_external_top_limit = [div_top_right, z_x_point]
        outer_leg_external_bottom_limit = outer_target_sol_end

        outer_leg_internal_top_limit = middle_point
        outer_leg_internal_bottom_limit = outer_target_pfr_end

        # Make the guide lines
        external_guide_line = make_guide_line(
            outer_leg_external_guide_line,
            outer_leg_external_top_limit,
            outer_leg_external_bottom_limit,
        )

        internal_guide_line = make_guide_line(
            outer_leg_internal_guide_line,
            outer_leg_internal_top_limit,
            outer_leg_internal_bottom_limit,
        )

        # Modify the clipped flux line curve (guide line) to start
        # at the middle and end at the internal point of the outer target
        (outer_leg_internal_line_x, outer_leg_internal_line_z,) = reshape_curve(
            internal_guide_line.x,
            internal_guide_line.z,
            [middle_point[0], middle_point[1]],
            outer_target_pfr_end,
            degree_in,
        )

        # Modify the clipped flux line curve to start at the top point of the
        # outer target and end at the external point
        (outer_leg_external_line_x, outer_leg_external_line_z,) = reshape_curve(
            external_guide_line.x,
            external_guide_line.z,
            [div_top_right, z_x_point],
            outer_target_sol_end,
            degree_out,
        )

        # Connect the inner and outer parts of the outer leg
        outer_leg_x = np.append(
            outer_leg_internal_line_x, outer_leg_external_line_x[::-1]
        )
        outer_leg_z = np.append(
            outer_leg_internal_line_z, outer_leg_external_line_z[::-1]
        )

        # Return coordinate arrays
        return (outer_leg_x, outer_leg_z)

    def make_inner_leg(self, div_top_left, inner_strike, middle_point, flux_loop):
        """
        Find the coordinates of the outer leg of the divertor.

        Parameters
        ----------
        div_top_left: float
            Top-left x-coordinate of the divertor
        inner_strike: [float,float]
            Coordinates of the inner strike point
        middle_point: [float,float]
            Coordinates of the middle point between the inner and outer legs
        flux_loop: Loop
            Outer flux loop used for shaping.

        Returns
        -------
        divertor_leg: (list, list)
            x and z coordinates of outer leg
        """
        # Find the tangent to the approriate flux loop at the outer strike point
        tangent = get_tangent_vector(inner_strike, flux_loop)

        # Get the outer target points
        (inner_target_pfr_end, inner_target_sol_end,) = self.make_divertor_target(
            inner_strike,
            tangent,
            vertical_target=self.inputs["div_vertical_inner_target"],
            outer_target=False,
        )

        # Select the degree of the fitting polynomial
        degree = self.inner_leg_polyfit_degree

        # Select those points along the given flux line below the X point
        inner_leg_central_guide_line = flux_loop
        z_x_point = self.points["x_point"]["z_low"]
        top_clip_inner_leg_central_guide_line = np.where(
            inner_leg_central_guide_line.z < z_x_point,
        )

        # Create a new Loop from the points selected along the given flux line
        inner_leg_central_guide_line = Loop(
            x=inner_leg_central_guide_line.x[top_clip_inner_leg_central_guide_line],
            y=None,
            z=inner_leg_central_guide_line.z[top_clip_inner_leg_central_guide_line],
        )

        # Select those points along the top-clipped flux line above the
        # inner target internal point height
        bottom_clip_inner_leg_central_guide_line = np.where(
            inner_leg_central_guide_line.z > inner_target_pfr_end[1],
        )

        # Create a new Loop from the points selected along the flux line
        inner_leg_central_guide_line = Loop(
            x=inner_leg_central_guide_line.x[bottom_clip_inner_leg_central_guide_line],
            y=None,
            z=inner_leg_central_guide_line.z[bottom_clip_inner_leg_central_guide_line],
        )

        # Modify the clipped flux line curve to start at the middle and end
        # at the internal point of the outer target
        (inner_leg_internal_line_x, inner_leg_internal_line_z,) = reshape_curve(
            inner_leg_central_guide_line.x,
            inner_leg_central_guide_line.z,
            [middle_point[0], middle_point[1]],
            inner_target_pfr_end,
            degree,
        )

        # Modify the clipped flux line curve to start at the top point of the
        # outer target and end at the external point
        (inner_leg_external_line_x, inner_leg_external_line_z,) = reshape_curve(
            inner_leg_central_guide_line.x,
            inner_leg_central_guide_line.z,
            [div_top_left, z_x_point],
            inner_target_sol_end,
            degree,
        )

        # Connect the inner and outer parts of the outer leg
        inner_leg_x = np.append(
            inner_leg_external_line_x, inner_leg_internal_line_x[::-1]
        )
        inner_leg_z = np.append(
            inner_leg_external_line_z, inner_leg_internal_line_z[::-1]
        )

        # Return coordinate arrays
        return (inner_leg_x, inner_leg_z)

    def find_koz_flux_loop_ints(self, koz, flux_loops):
        """
        Find intersections between the keep-out-zone loop
        and the given flux loops.  Only upper and lower most
        intersections for each flux line are returned.

        Parameters
        ----------
        koz : Loop
            Loop representing the keep-out-zone
        flux_loops: list of Loop
            List of flux loops used to find intersections

        Returns
        -------
        all_points : list
             List of the [x,z] coordinates of the intersections
        """
        # For each flux loop find the intersections with the koz
        all_points = []
        for loop in flux_loops:
            # Expectation is that flux loop is open
            if loop.closed:
                raise SystemsError(
                    "Selected flux contour is closed, please check psi_norm"
                )

            # Get the intersections
            int_x, int_z = get_intersect(koz, loop)

            # Combine into [x,z] points
            points = list(map(list, zip(int_x, int_z)))
            all_points.extend(points)

        return all_points

    def find_strike_points_from_koz(self, flux_loops):
        """
        Find the lowermost points of intersection between the
        keep-out-zone and the given flux loops.

        Parameters
        ----------
        flux_loops: list of Loop
            List of flux loops used to find intersections

        Returns
        -------
        inner, outer : list, list
            Inner and outer strike points as [x,z] list.

        """
        # Check we have a koz and psi_norm in inputs
        koz = self.inputs["koz"]

        # Shorthand
        z_low = self.points["x_point"]["z_low"]

        # Find all_intersection points between keep-out zone and flux
        # surface having the given psi_norm
        all_ints = self.find_koz_flux_loop_ints(koz, flux_loops)

        # Sort into a dictionary indexed by x-coord and save extremal
        x_min = np.max(koz.x)
        x_max = np.min(koz.x)
        sorted = {}
        for pt in all_ints:
            x_now = pt[0]
            z_now = pt[1]

            # Only accept points below lower X point
            if z_now > z_low:
                continue
            if x_now not in sorted:
                sorted[x_now] = []
            sorted[x_now].append(z_now)

            # save limits
            if x_now > x_max:
                x_max = x_now
            if x_now < x_min:
                x_min = x_now

        if len(sorted) == 0:
            raise SystemsError(
                "No intersections with keep-out zone below lower X-point."
            )

        # Check we have distinct xmin, xmax
        if x_min == x_max:
            raise SystemsError("All keep-out-zone intersections have same x-coord")

        # Sort extremal inner and outer z coords and select lowest
        sorted[x_min].sort()
        sorted[x_max].sort()
        inner_z = sorted[x_min][0]
        outer_z = sorted[x_max][0]

        # Construct the inner and outer points and return
        inner = [x_min, inner_z]
        outer = [x_max, outer_z]
        return inner, outer

    def find_strike_points_from_params(self, flux_loops):
        """
        Find the inner and outer strike points using the relative
        height from the lower X point taken from self.params
        and look for the intersection point with given flux loop(s).

        Parameters
        ----------
        flux_loops : list of Loop
            Loops with which the strike point should intersect.
            For SN case this will be a list with one entry.

        Returns
        -------
        inner,outer : list,list
            Lists of [x,z] coords corresponding to inner and outer
            strike points
        """
        # Some shorthands
        x_x_point = self.points["x_point"]["x"]

        outer_loop, inner_loop = self.get_outer_inner_loops(flux_loops)

        # Get the inner intersection with the separatrix
        inner_strike_x = self.params.inner_strike_r.value
        x_norm = 0
        # Does it make sense to compare x with x-norm??
        inner_strike_z = get_intersection_point(
            inner_strike_x, x_norm, inner_loop, x_x_point, inner=True
        )[2]

        # Get the outer intersection with the separatrix
        outer_strike_x = self.params.outer_strike_r.value
        # Does it make sense to compare x with x-norm??
        outer_strike_z = get_intersection_point(
            outer_strike_x, x_norm, outer_loop, x_x_point, inner=False
        )[2]

        inner_strike = [inner_strike_x, inner_strike_z]
        outer_strike = [outer_strike_x, outer_strike_z]

        return inner_strike, outer_strike

    def find_strike_points(self, flux_loops):
        """
        Find the inner and outer strike points, taking intersections
        from the given inner / outer flux loops

        Parameters
        ----------
        flux_loops: list of Loop
            List of flux loops used to find intersections

        Returns
        -------
        inner,outer : list,list
            Lists of [x,z] coords corresponding to inner and outer
            strike points
        """
        if "strike_pts_from_koz" in self.inputs and self.inputs["strike_pts_from_koz"]:
            return self.find_strike_points_from_koz(flux_loops)
        else:
            return self.find_strike_points_from_params(flux_loops)

    def pick_flux_loops(self):
        """
        Return a list of flux loops to be used to find strike points.

        Returns
        -------
        flux_loops : list of Loop
            List of flux loops used to find intersections
        """
        flux_loops = []
        if (
            "pick_flux_from_psinorm" in self.inputs
            and self.inputs["pick_flux_from_psinorm"]
            and "psi_norm" in self.inputs
        ):
            # If flag in inputs is true, use psi_norm value
            psi_norm = self.params["psi_norm"]
            flux_loops = make_flux_contour_loops(self.equilibrium, psi_norm)
        else:
            # Default: use separatrix
            flux_loops = self.separatrix
        return flux_loops

    def make_divertor_cassette(self, divertor_loops, outer_profile):
        """
        Given the divertor loops create the divertor cassette.

        Parameters
        ----------
        divertor_loops : list
            List of Loop objects representing the divertor
        outer_profile : Loop
            Loop of the outer profile of first wall.

        Returns
        -------
        divertor_cassette : list
            List of Loop objects representing the divertor cassettes
            (one for each divertor)
        """
        # Fetch the vacuum vessel
        vv = self.inputs["vv_inner"]

        # Find the limits of the vacuum vessel
        z_max_vv = np.max(vv.z)
        z_min_vv = np.min(vv.z)
        x_max_vv = np.max(vv.x)
        x_min_vv = np.min(vv.x)

        # Make a cassette for each divertor
        cassettes = []
        for div in divertor_loops:

            # Create an offset divertor with a given thickness
            offset = self.params.tk_div_cass
            div_offset = offset_clipper(div, offset, method="miter")

            # Take the outermost radial limit from offset div
            x_max = np.max(div.x) + offset

            # Take the innermost radial limit from offset div, plus extra thickness
            x_min = np.min(div.x) - offset - self.params.tk_div_cass_in

            # Check we're inside the vacuum vessel
            if x_min < x_min_vv or x_max > x_max_vv:
                raise GeometryError(
                    "Divertor cassette radial limits overlap with vacuum vessel.\n"
                    f"Minimal VV radius {x_min_vv} > smallest casette radius {x_min}\n"
                    f"Maximum VV radius {x_max_vv} < largest casette radius {x_max}\n"
                )

            # z limits: want to be flush with flat edge of (non-offset) divertor
            z_min_div = np.min(div.z)
            z_max_div = np.max(div.z)

            # Lower or upper divertor?
            if z_min_div > 0.0:
                # Upper divertor
                z_min = z_min_div
                z_max = z_max_vv
            else:
                # Lower divertor
                z_max = z_max_div
                z_min = z_min_vv

            # Check z coords are inside acceptable bounds
            if z_max > z_max_vv or z_min < z_min_vv:
                raise GeometryError(
                    "Divertor cassette z-limits overlap with vacuum vessel\n"
                    f"Minimal VV half-height {z_min_vv} > smallest casette half-height {z_min}"
                    f"Maximum VV half-height {z_max_vv} < largest casette half-height {z_max}"
                )

            div_box = make_box_xz(x_min, x_max, z_min, z_max)

            # Want to cut out space between outer leg and box
            cutters = boolean_2d_difference_split(div_box, div_offset)

            # Deduce the correct cutter by its limits
            cutter_select = None
            for cutter in cutters:
                cutter_x_max = np.max(cutter.x)
                cutter_x_min = np.min(cutter.x)
                if z_min_div > 0.0:
                    # Upper divertor
                    cutter_z_lim = np.min(cutter.z)
                    z_lim = z_min_div
                else:
                    # Lower divertor
                    cutter_z_lim = np.max(cutter.z)
                    z_lim = z_max_div

                if (
                    np.isclose(cutter_x_max, x_max)
                    and np.isclose(cutter_z_lim, z_lim)
                    and cutter_x_min > x_min
                ):
                    cutter_select = cutter
                    break

            # If outer leg is vertical, no space to cut: cutter may be None
            if cutter_select:
                # Apply our cutter
                div_box = boolean_2d_difference_loop(div_box, cutter_select)

            # Find the overlap between the vv and the box
            cassette = boolean_2d_common_loop(div_box, vv)

            # Subtract the first wall outer profile
            subtracted = boolean_2d_difference(cassette, outer_profile)
            cassette = subtracted[0]

            # Finally, simplify
            cassette = simplify_loop(cassette)
            cassettes.append(cassette)

        return cassettes

    def get_outer_inner_loops(self, flux_loops):
        """
        Get the outer and inner loops in a weird way
        """
        # SN case: just one loop
        if self.inputs.get("SN", False):
            outer_loop = inner_loop = flux_loops[0]
        else:
            outer_loop = flux_loops[0]
            inner_loop = flux_loops[1]
        return outer_loop, inner_loop

    def set_lfs_point(self, point):
        """
        Set the points at which the last open flux surface on the OMP is...
        """
        self._lfs_point = point


class DEMODivertorBuilder(DivertorBuilder):
    """
    More separation of concerns
    """

    def __init__(self, *args):
        super().__init__(*args)

        if self.inputs["SN"]:
            degree_in = degree_out = self.inputs.get(
                "outer_leg_sol_polyfit_degree",
                self.inputs.get("outer_leg_pfr_polyfit_degree", 2),
            )

        else:
            degree_in = degree_out = self.inputs.get(
                "outer_leg_sol_polyfit_degree",
                self.inputs.get("outer_leg_pfr_polyfit_degree", 1),
            )
        self.outer_leg_sol_polyfit_degree = degree_in
        self.outer_leg_pfr_polyfit_degree = degree_out

        if self.inputs["SN"]:
            degree = self.inputs.get("inner_leg_polyfit_degree", 2)
        else:
            degree = self.inputs.get("inner_leg_polyfit_degree", 1)
        self.inner_leg_polyfit_degree = degree

    def get_guide_lines(self, flux_loop):
        """
        No idea
        """
        return flux_loop, flux_loop


class STEPDivertorBuilder(DivertorBuilder):
    """
    STEP separation of concerns
    """

    def __init__(self, *args):
        super().__init__(*args)

        self.outer_leg_sol_polyfit_degree = self.inputs.get(
            "outer_leg_sol_polyfit_degree", 3
        )
        self.outer_leg_pfr_polyfit_degree = self.inputs.get(
            "outer_leg_pfr_polyfit_degree", 3
        )
        self.inner_leg_polyfit_degree = self.inputs.get("inner_leg_polyfit_degree", 1)

    def get_guide_lines(self, flux_loop):
        """
        A particular nightmare, I believe it would cause issues in the iterative
        "optimisation" procedure, I've tried to find a workaround.
        """
        eq = self.equilibrium
        p = self._lfs_point
        x, z = find_flux_surface_through_point(
            eq.x, eq.z, eq.psi(), p[0], p[2], eq.psi(p[0], p[2])
        )
        lfs_loop = Loop(x=x, z=z)

        return lfs_loop, flux_loop


class FirstWall(ReactorSystem):
    """
    Reactor First Wall (FW) system abstract base class
    """

    config: Type[ParameterFrame]
    inputs: dict
    CADConstructor = FirstWallCAD

    # fmt: off
    base_default_params = [
        ["n_TF", "Number of TF coils", 16, "dimensionless", None, "Input"],
        ["A", "Plasma aspect ratio", 3.1, "dimensionless", None, "Input"],
        ["psi_norm", "Normalised flux value of strike-point contours",
         1, "dimensionless", None, "Input"],
        ['P_sep_particle', 'Separatrix power', 150, 'MW', None, 'Input'],
        ["f_p_sol_near", "near scrape-off layer power rate", 0.65, "dimensionless", None, "Input"],
        ['tk_fw_in', 'Inboard first wall thickness', 0.052, 'm', None, 'Input'],
        ['tk_fw_out', 'Outboard first wall thickness', 0.052, 'm', None, 'Input'],
        ['tk_fw_div', 'First wall thickness around divertor', 0.052, 'm', None, 'Input'],
        ['tk_div_cass', 'Minimum thickness between inner divertor profile and cassette', 0.3, 'm', None, 'Input'],
        ['tk_div_cass_in', 'Additional radial thickness on inboard side relative to to inner strike point', 0.1, 'm', None, 'Input'],
    ]
    # fmt: on

    def __init__(self, config, inputs):
        self.config = config
        self.inputs = inputs

        self._init_params(self.config)
        self.init_equilibrium()

        # De-carbonarisation hack
        inputs = self.inputs.copy()
        inputs["points"] = self.points
        inputs["separatrix"] = self.separatrix
        if self.inputs.get("DEMO_like_divertor", False):
            self.divertor_builder = DEMODivertorBuilder(
                self.params, self.config, inputs, self.equilibrium
            )
        else:
            self.divertor_builder = STEPDivertorBuilder(
                self.params, self.config, inputs, self.equilibrium
            )

        self._plotter = FirstWallPlotter()

    # Setup stuff
    def init_equilibrium(self):
        """
        Some housework to get rid of EqInputs...
        """
        self.equilibrium = self.inputs["equilibrium"]

        lcfs_shift = 0.001
        x_point_shift = self.params.xpt_height

        # Save inputs
        self.lcfs_shift = lcfs_shift
        self.x_point_shift = x_point_shift

        # First find the last closed flux surface
        self.lcfs = self.equilibrium.get_LCFS()
        self.separatrix = self.equilibrium.get_separatrix()
        # Find the local maxima (O) and inflection (X) points in the flux field
        o_point, x_point = self.equilibrium.get_OX_points()
        self.points = {
            "x_point": {
                "x": x_point[0][0],
                "z_low": x_point[0][1],
                "z_up": x_point[1][1],
            },
            "o_point": {"x": o_point[0][0], "z": round(o_point[0][1], 5)},
        }
        if self.points["x_point"]["z_low"] > self.points["x_point"]["z_up"]:
            self.points["x_point"]["z_low"] = x_point[1][1]
            self.points["x_point"]["z_up"] = x_point[0][1]

        # Define the mid-plane as having z-normal and containing O point.
        self.mid_plane = make_plane(self.points["o_point"]["z"], 2)
        self.x_imp_lcfs = np.min(loop_plane_intersect(self.lcfs, self.mid_plane).T[0])

    # Actual run
    def build(self, callback=None):
        """
        Build the 2D profile
        """
        if "profile" in self.inputs:
            self.profile = self.inputs["profile"]

        elif self.inputs.get("FW_optimisation", False) and callback is not None:
            callback(self, hf_limit=0.2, n_iteration_max=5)

        else:
            self.profile = self.make_preliminary_profile()

        self.make_2d_profile()
        self.profile = self.geom["2D profile"].inner

        self.hf_firstwall_params(self.profile)

    # Output and plotting stuff
    def hf_save_as_csv(self, filename="hf_on_the_wall", metadata=""):
        """
        Generate a .csv file with the coordinates of flux line intersections
        with the first wall  and corresponding local heat flux value
        """
        # Collecting in three different (1 level) lists the intersection
        # point coordinates and heat flux values
        input_x = self.x_all_ints
        input_z = self.z_all_ints
        input_hf = self.hf_all_ints

        # The .csv file, besides the header, will have 3 columns and n rows
        # n = number of intersections
        data = np.array([input_x, input_z, input_hf]).T

        header = "Intersection points and relevant hf"
        if metadata != "" and not metadata.endswith("\n"):
            metadata += "\n"
        header = metadata + header
        col_names = ["x", "z", "heat_flux"]
        write_csv(data, filename, col_names, header)

    def plot_hf(self, ax=None, **kwargs):
        """
        Plots the first wall, the separatrix, the SOL discretised as set of
        flux lines, intersections between fw and flux lines, relevant
        local heat flux and, if given as input, the keep out zone

        Parameters
        ----------
        ax: Axes, optional
            The optional Axes to plot onto, by default None.
        """
        koz = self.inputs.get("koz", None)
        separatrix = self.separatrix
        if isinstance(separatrix, list):
            separatrix = separatrix[0]
        self._plotter.plot_hf(
            separatrix,
            self.solver.flux_surfaces,
            self.x_wall,
            self.z_wall,
            self.hf_wall,
            self.geom["2D profile"].inner,
            koz,
            ax=ax,
            **kwargs,
        )

    # Actual heat flux calculation

    def hf_firstwall_params(self, profile):
        """
        Define params to plot the heat flux on the fw (no divertor).

        Parameters
        ----------
        profile: Loop
            A first wall 2D profile

        Returns
        -------
        x_wall: [float]
            x coordinates of first intersections on the wall
        z_wall: [float]
            z coordinates of first intersections on the wall
        hf_wall: [float]
            heat flux values associated to the first flux lines-wall intersections
        """
        self.solver = ChargedParticleSolver(
            self.config, self.equilibrium, dx_mp=self.inputs["dx_mp"]
        )
        x_wall, z_wall, hf_wall = self.solver.analyse(profile)
        self.x_all_ints = x_wall
        self.z_all_ints = z_wall
        self.hf_all_ints = hf_wall
        return x_wall, z_wall, hf_wall

    # Geometry creation and modification methods

    def attach_divertor(self, fw_loop, divertor_loops):
        """
        Attaches a divertor to the first wall

        Parameters
        ----------
        fw_loop: Loop
            first wall profile

        divertor_loops: list
            list of divertor Loops

        Returns
        -------
        fw_diverted_loop: Loop
            Here the first wall also has a divertor geometry
        """
        fw_diverted_loop = fw_loop
        # Attach each disjoint portion of the divertor
        # (e.g. top / bottom)
        for div in divertor_loops:
            union = boolean_2d_union(fw_diverted_loop, div)
            fw_diverted_loop = union[0]

        # Simplify at end
        fw_diverted_loop = simplify_loop(fw_diverted_loop)
        return fw_diverted_loop

    def make_2d_profile(self):
        """
        Create the 2D profile
        """
        # Ensure our starting profile is closed
        self.profile.close()

        # Nightmare spaghetti
        points = loop_plane_intersect(self.profile, self.mid_plane)
        idx = np.argmax(points[:, 0])
        self.divertor_builder.set_lfs_point(points[idx])

        inner_divertor_loops = self.divertor_builder.make_divertor(self.profile)

        # Attach the divertor to the initial profile
        self.inner_profile = self.attach_divertor(self.profile, inner_divertor_loops)

        # Offset the inner profile to make an outer profile
        outer_profile, sections = self.make_outer_wall(
            self.inner_profile,
            inner_divertor_loops,
            self.params.tk_fw_in,
            self.params.tk_fw_out,
            self.params.tk_fw_div,
        )

        # Extract the different sections of the outer wall
        self.divertor_loops = sections[:-2]
        inboard_wall = sections[-2]
        outboard_wall = sections[-1]

        # Make a shell from the inner and outer profile
        fw_shell = Shell(inner=self.inner_profile, outer=outer_profile)

        # Save geom objects
        self.geom["Preliminary profile"] = self.profile
        self.geom["2D profile"] = fw_shell
        self.geom["Inner profile"] = self.inner_profile
        self.geom["Inboard wall"] = inboard_wall
        self.geom["Outboard wall"] = outboard_wall

        # For now, make the divertor cassette here (to be refactored)
        self.divertor_cassettes = self.divertor_builder.make_divertor_cassette(
            self.divertor_loops, self.geom["2D profile"].outer
        )

        n_div = len(self.divertor_loops)
        n_cass = len(self.divertor_cassettes)
        if n_div != n_cass:
            raise SystemsError("Inconsistent number of divertors and cassettes")

        if n_div == 1:
            self.geom["Divertor"] = self.divertor_loops[0]
            self.geom["Divertor cassette"] = self.divertor_cassettes[0]
        elif n_div == 2:
            self.geom["Divertor lower"] = self.divertor_loops[0]
            self.geom["Divertor upper"] = self.divertor_loops[1]
            self.geom["Divertor cassette lower"] = self.divertor_cassettes[0]
            self.geom["Divertor cassette upper"] = self.divertor_cassettes[1]
        else:
            raise SystemsError("Inappropriate number of divertors")

    def make_outer_wall(
        self, inner_wall, divertor_loops, tk_inboard, tk_outboard, tk_div
    ):
        """
        Create the outer wall 2D profile given the inner wall and the divertor.
        Allows for three different thicknesses, on the inboard / outboard side
        and around the divertor.

        Parameters
        ----------
        inner_wall: Loop
            2D profile of the inner wall
        divertor_loops: list
            List of loops which constitute the divertor(s).
        tk_inboard: float
            Thickness of the wall on the inboard side.
        tk_outboard: float
            Thickness of the wall on the outboard side.
        tk_div: float
            Thickness of the wall around the divertor.

        Returns
        -------
        outer_wall : Loop
            2D profile of the outer wall
        """
        # Find the max thickness
        max_tk = 2.0 * max(max(tk_inboard, tk_outboard), tk_div)

        # Create bounding box cutters
        cutters = self.make_cutters(inner_wall, divertor_loops, max_tk)

        # Divide inner wall into inboard / outboard portions
        cutter_in = cutters[0]
        cutter_out = cutters[1]
        inboard = boolean_2d_common_loop(inner_wall, cutter_in)
        outboard = boolean_2d_common_loop(inner_wall, cutter_out)

        # Now subtract divertor loops from inboard / outboard
        for div in divertor_loops:
            inboard = boolean_2d_difference_loop(inboard, div)
            outboard = boolean_2d_difference_loop(outboard, div)

        # Offset each loop with the appropriate thickness
        inboard = offset_clipper(inboard, tk_inboard, method="miter")
        outboard = offset_clipper(outboard, tk_outboard, method="miter")
        offset_divertor_loops = []
        for div in divertor_loops:
            offset_divertor_loops.append(offset_clipper(div, tk_div, method="miter"))

        # Remove the overlaps between the offset sections
        sections = get_non_overlapping(inboard, outboard, offset_divertor_loops, cutters)

        # Now find the union of our offset loops and the original profile
        outer_wall = self.attach_divertor(inner_wall, sections)

        # Subtract the inner profile from each component
        for ii, sec in enumerate(sections):
            section = boolean_2d_difference_loop(sec, inner_wall)
            # Remove duplicate points on the loop
            clean_points = clean_loop_points(section, 1e-4)
            clean_array = np.array(clean_points).T
            clean_loop = Loop(x=clean_array[0], z=clean_array[1])
            clean_loop.close()
            sections[ii] = clean_loop

        # Return both the union and individual sections
        return outer_wall, sections

    def make_cutters(self, inner_wall, divertor_loops, thickness):
        """
        Intermediate step to make bounding-box cutters around
        inboard / outboard / divertor sections of first wall.

        Parameters
        ----------
        inner_wall: Loop
            2D profile of the inner wall
        divertor_loops: list
            List of loops which constitute the divertor(s).
        thickness: float
            Thickness to add around the bounding boxes.

        Returns
        -------
        cutters : tuple of Loop
            Tuple of cutters: (inboard, outboard, [divertor_cutters])
        """
        # Draw boxes to separate inboard / outboard side using x point
        # and  the limits of the inner profile loop
        x_x_point = self.points["x_point"]["x"]
        x_max = np.max(inner_wall.x) + thickness
        x_min = np.min(inner_wall.x) - thickness
        z_max = np.max(inner_wall.z) + thickness
        z_min = np.min(inner_wall.z) - thickness
        cutter_in = make_box_xz(x_min, x_x_point, z_min, z_max)
        cutter_out = make_box_xz(x_x_point, x_max, z_min, z_max)

        # Make a cutters for each divertor
        div_cutters = []
        for div in divertor_loops:
            x_max = np.max(div.x) + thickness
            x_min = np.min(div.x) - thickness

            # Lower or upper divertor? We want to be flush with the
            # original horizontal join
            z_div_max = np.max(div.z)
            z_div_min = np.min(div.z)
            if z_div_min > 0.0:
                # Upper divertor
                z_min = z_div_min
                z_max = z_div_max + thickness
            else:
                # Lower divertor
                z_max = z_div_max
                z_min = z_div_min - thickness

            div_cutter = make_box_xz(x_min, x_max, z_min, z_max)
            div_cutters.append(div_cutter)

        return (cutter_in, cutter_out, div_cutters)

    def horizontal_clipper(self, loop, vertical_reference=None, top_limit=None):
        """
        Loop clipper.
        Removes bottom and top part of a loop. The bottom limit for the cut
        is the lower x point, while the top limit can be specified. If it is
        not, the upper x point is assigned.
        A vertical plane can be assigned to keep either the part part of the
        loop on the right or on the left of such additional geometrical limit.

        Parameters
        ----------
        loop: Loop
            Loop to cut
        vertical_reference: [float, float]
            Reference axis against which to cut
        top_limit: float
            z coordinate of the top limit

        Returns
        -------
        Loop: Loop
            New modified loop
        """
        if vertical_reference is not None:
            new_loop = Loop(loop.x[vertical_reference], z=loop.z[vertical_reference])
        else:
            new_loop = loop

        if top_limit is None:
            top_limit = self.points["x_point"]["z_up"] + self.x_point_shift

        clip_bottom = np.where(
            new_loop.z > self.points["x_point"]["z_low"] - self.x_point_shift
        )

        new_loop = Loop(new_loop.x[clip_bottom], z=new_loop.z[clip_bottom])

        clip_top = np.where(new_loop.z < top_limit)

        return Loop(new_loop.x[clip_top], z=new_loop.z[clip_top])

    def vertical_clipper(self, loop, x_ref, horizontal_refernce=None):
        """
        Loop clipper.
        According to an x reference coordinate, removes the part of the loop,
        either on the right or on the left of the x-point, which does not
        contain such point.
        A horizontal plane can be assigned to keep either the top part or the
        bottom partof the loop.

        Parameters
        ----------
        loop: Loop
            Loop to cut
        x_ref: float
            x coordinate of the reference point
        horizontal_refernce: [float, float]
            Reference axis against which to cut

        Returns
        -------
        Loop: Loop
            New modified loop
        """
        if horizontal_refernce is not None:
            new_loop = Loop(loop.x[horizontal_refernce], z=loop.z[horizontal_refernce])
        else:
            new_loop = loop

        if x_ref > self.points["x_point"]["x"]:
            clip_right = np.where(new_loop.x > self.points["x_point"]["x"])
            new_loop = Loop(x=new_loop.x[clip_right], z=new_loop.z[clip_right])
        elif x_ref < self.points["x_point"]["x"]:
            clip_left = np.where(new_loop.x < self.points["x_point"]["x"])
            new_loop = Loop(x=new_loop.x[clip_left], z=new_loop.z[clip_left])

        return new_loop

    def make_preliminary_profile(self):
        """
        Generate a preliminary first wall profile in case it is not given as input
        """
        raise NotImplementedError

    @property
    def xz_plot_loop_names(self):
        """
        The x-z loop names to plot.
        """
        names = ["Inboard wall", "Outboard wall", "Divertor", "Divertor cassette"]
        return names


class FirstWallSN(FirstWall):
    """
    Reactor First Wall (FW) system

    First Wall design for a SN configuration
    The user needs to change the default parameters according to the case

    Attributes
    ----------
    self.flux_surfaces: [Loop]
        Set of flux surfaces to discretise the SOL
    self.flux_surface_width_omp: [float]
        Thickness of flux sirfaces
    """

    # fmt: off
    default_params = FirstWall.base_default_params + [
        ['tk_sol_ob', 'Outboard SOL thickness', 0.225, 'm', None, 'Input'],
        ["fw_psi_n", "Normalised psi boundary to fit FW to", 1.01, "dimensionless", None, "Input"],
        ["fw_lambda_q_near", "Lambda q near SOL", 0.05, "m", None, "Input"],
        ["fw_lambda_q_far", "Lambda q far SOL", 0.05, "m", None, "Input"],
        ["f_lfs_lower_target", "Fraction of SOL power deposited on the LFS lower target", 0.75, "dimensionless", None, "Input"],
        ["f_hfs_lower_target", "Fraction of SOL power deposited on the HFS lower target", 0.25, "dimensionless", None, "Input"],
        ["f_lfs_upper_target", "Fraction of SOL power deposited on the LFS upper target", 0.0, "dimensionless", "DN only", "Input"],
        ["f_hfs_upper_target", "Fraction of SOL power deposited on the HFS upper target", 0.0, "dimensionless", "DN only", "Input"],

        # Parameters used in make_divertor_loop
        ["xpt_outer_gap", "Gap between x-point and outer wall", 1, "m", None, "Input"],
        ["xpt_inner_gap", "Gap between x-point and inner wall", 1, "m", None, "Input"],
        ["outer_strike_r", "Outer strike point major radius", 9, "m", None, "Input"],
        ["inner_strike_r", "Inner strike point major radius", 6.5, "m", None, "Input"],
        ["tk_outer_target_sol", "Outer target length between strike point and SOL side",
         0.7, "m", None, "Input"],
        ["tk_outer_target_pfr", "Outer target length between strike point and PFR side",
         0.3, "m", None, "Input"],
        ["theta_outer_target",
         "Angle between flux line tangent at outer strike point and SOL side of outer target",
         50, "deg", None, "Input"],
        ["theta_inner_target",
         "Angle between flux line tangent at inner strike point and SOL side of inner target",
         30, "deg", None, "Input"],
        ["tk_inner_target_sol", "Inner target length SOL side", 0.3, "m", None, "Input"],
        ["tk_inner_target_pfr", "Inner target length PFR side", 0.5, "m", None, "Input"],
        ["xpt_height", "x-point vertical_gap", 0.4, "m", None, "Input"],
    ]
    # fmt: on

    def hf_firstwall_params(self, profile):
        """
        Perform the heat flux calculation on a FW silhouette
        """
        x, z, hf = super().hf_firstwall_params(profile)
        idx = np.where(z > self.points["x_point"]["z_low"] + self.x_point_shift)[0]
        x_wall = x[idx]
        z_wall = z[idx]
        hf_wall = hf[idx]

        self.x_wall = x_wall
        self.z_wall = z_wall
        self.hf_wall = hf_wall
        return (x_wall, z_wall, hf_wall)

    # Geometry creation

    def make_preliminary_profile(self):
        """
        Generate a preliminary first wall profile in case it is not given as input

        Returns
        -------
        fw_loop: Loop
            Here the first wall is without divertor. The wall is cut at the X-point
        """
        dx_loop = self.lcfs.offset(self.params.tk_sol_ob)
        psi_n_loop = self.equilibrium.get_flux_surface(
            self.params.fw_psi_n,
        )

        fw_limit = self.points["x_point"]["z_low"] - self.x_point_shift
        clip1 = np.where(dx_loop.z > fw_limit)
        clip2 = np.where(psi_n_loop.z > fw_limit)

        # Adding divertor entrance limits
        x_left = self.points["x_point"]["x"] - self.params.xpt_inner_gap
        x_right = self.points["x_point"]["x"] + self.params.xpt_outer_gap

        new_loop1 = Loop(dx_loop.x[clip1], z=dx_loop.z[clip1])
        new_loop1.insert([x_left, 0, self.points["x_point"]["z_low"]])
        new_loop2 = Loop(psi_n_loop.x[clip2], z=psi_n_loop.z[clip2])
        new_loop2.insert([x_right, 0, self.points["x_point"]["z_low"]])

        hull = convex_hull([new_loop1, new_loop2])

        fw_loop = Loop(x=hull.x, z=hull.z)
        self.preliminary_profile = fw_loop

        return fw_loop

    def modify_fw_profile(self, profile, x_int_hf, z_int_hf):
        """
        Modify the first wall to reduce the heat flux

        Parameters
        ----------
        profile: Loop
            First wall to optimise
        x_int_hf: float
            x coordinate at the intersection
        z_int_hf: float
            z coordinate at the intersection
        heat_flux: float
            heat fluxe at the intersection

        Returns
        -------
        new_fw_profile: Loop
            Optimised profile
        """
        clipped_loops = []
        self.loops = self.equilibrium.get_flux_surface_through_point(x_int_hf, z_int_hf)

        for loop in self.loops:
            if loop_plane_intersect(loop, self.mid_plane) is not None:
                new_loop = self.horizontal_clipper(
                    loop, top_limit=self.points["o_point"]["z"]
                )
                clipped_loops.append(new_loop)

        if len(clipped_loops) == 0:
            new_fw_profile = profile

        elif len(clipped_loops) == 1:
            loop = self.vertical_clipper(clipped_loops[0], x_int_hf)
            hull = convex_hull([profile, loop])
            new_fw_profile = Loop(x=hull.x, z=hull.z)
            new_fw_profile.close()

        elif len(clipped_loops) == 2:
            hull = convex_hull([profile, clipped_loops[1]])
            new_fw_profile = Loop(x=hull.x, z=hull.z)
            new_fw_profile.close()

        return new_fw_profile


class FirstWallDN(FirstWall):
    """
    Reactor First Wall (FW) system

    First Wall design for a DN configuration
    The user needs to change the default parameters according to the case


    Attributes
    ----------
    self.flux_surfaces: [[Loop]]
        Set of flux surfaces to discretise the SOL
        Each flus surfaces can have either one or two loops (lfs and hfs)
    self.flux_surface_lfs: [Loop]
        All the fs parts at the lfs
    self.flux_surface_hfs: [Loop]
        All the fs parts at the hfs
        Expected len(self.flux_surface_lfs) > len(self.flux_surface_hfs)
    self.flux_surface_width_omp: [float]
        Thickness of flux sirfaces on the lfs
    self.flux_surface_width_imp: [float]
        Thickness of flux sirfaces on the hfs

    """

    # fmt: off
    default_params = FirstWall.base_default_params + [
        ['tk_sol_ib', 'Inboard SOL thickness', 0.225, 'm', None, 'Input'],
        ['tk_sol_ob', 'Outboard SOL thickness', 0.225, 'm', None, 'Input'],
        ["fw_psi_n", "Normalised psi boundary to fit FW to", 1, "dimensionless", None, "Input"],
        ["fw_lambda_q_near_omp", "Lambda_q near SOL omp", 0.003, "m", None, "Input"],
        ["fw_lambda_q_far_omp", "Lambda_q far SOL omp", 0.1, "m", None, "Input"],
        ["fw_lambda_q_near_imp", "Lambda_q near SOL imp", 0.003, "m", None, "Input"],
        ["fw_lambda_q_far_imp", "Lambda_q far SOL imp", 0.1, "m", None, "Input"],
        # These are now wrapped into a single analysis_tweak term: dx_mp
        # It's fast enough that there is no need to discretise differently
        # ["dr_near_omp", "fs thickness near SOL", 0.001, "m", None, "Input"],
        # ["dr_far_omp", "fs thickness far SOL", 0.005, "m", None, "Input"],
        # These seem to be inconsistent with the above, or at least could be set as such
        # Do not appear to be used anyway

        ["f_lfs_lower_target", "Fraction of SOL power deposited on the LFS lower target", 0.9 * 0.5, "dimensionless", None, "Input"],
        ["f_hfs_lower_target", "Fraction of SOL power deposited on the HFS lower target", 0.1 * 0.5, "dimensionless", None, "Input"],
        ["f_lfs_upper_target", "Fraction of SOL power deposited on the LFS upper target", 0.9 * 0.5, "dimensionless", "DN only", "Input"],
        ["f_hfs_upper_target", "Fraction of SOL power deposited on the HFS upper target", 0.1 * 0.5, "dimensionless", "DN only", "Input"],

        # These are now deprecated, in favour of just doing the mupltication in the
        # inputs above
        # ["p_rate_omp", "power sharing omp", 0.9, "%", None, "Input"],
        # ["p_rate_imp", "power sharing imp", 0.1, "%", None, "Input"],

        ["hf_limit", "heat flux material limit", 0.5, "MW/m^2", None, "Input"],
        # External inputs to draw the divertor
        ["xpt_outer_gap", "Gap between x-point and outer wall", 0.7, "m", None, "Input"],
        ["xpt_inner_gap", "Gap between x-point and inner wall", 0.7, "m", None, "Input"],
        ["outer_strike_r", "Outer strike point major radius", 10.3, "m", None, "Input"],
        ["inner_strike_r", "Inner strike point major radius", 8, "m", None, "Input"],
        ["tk_outer_target_sol", "Outer target length between strike point and SOL side",
         0.4, "m", None, "Input"],
        ["tk_outer_target_pfr", "Outer target length between strike point and PFR side",
         0.4, "m", None, "Input"],
        ["theta_outer_target",
         "Angle between flux line tangent at outer strike point and SOL side of outer target",
         20, "deg", None, "Input"],
        ["theta_inner_target",
         "Angle between flux line tangent at inner strike point and SOL side of inner target",
         30, "deg", None, "Input"],
        ["tk_inner_target_sol", "Inner target length SOL side", 0.2, "m", None, "Input"],
        ["tk_inner_target_pfr", "Inner target length PFR side", 0.2, "m", None, "Input"],
        ["xpt_height", "x-point vertical_gap", 0.4, "m", None, "Input"],
    ]
    # fmt: on

    def hf_firstwall_params(self, profile):
        """
        Perform the heat flux calculation on a FW silhouette
        """
        x, z, hf = super().hf_firstwall_params(profile)
        idx = np.where(
            (z < self.points["x_point"]["z_up"]) & (z > self.points["x_point"]["z_low"])
        )[0]

        x_wall = x[idx]
        z_wall = z[idx]
        hf_wall = hf[idx]

        self.x_wall = x_wall
        self.z_wall = z_wall
        self.hf_wall = hf_wall
        return (x_wall, z_wall, hf_wall)

    # Geometry creation
    def make_preliminary_profile(self):
        """
        Generate a preliminary first wall profile shape
        The preliminary first wall is drawn by offsetting the lcfs
        The top and bottom part of the wall are forced to end where
        the divertor is supposed to start. Relevant point are external inputs

        Returns
        -------
        fw_loop: Loop
            Here the first wall is without divertor. The wall is cut at the X-point
        """
        dx_loop_lfs = self.lcfs.offset(self.params.tk_sol_ob)
        clip_lfs = np.where(
            dx_loop_lfs.x > self.points["x_point"]["x"],
        )
        new_loop_lfs = Loop(
            dx_loop_lfs.x[clip_lfs],
            z=dx_loop_lfs.z[clip_lfs],
        )

        dx_loop_hfs = self.lcfs.offset(self.params.tk_sol_ib)
        clip_hfs = np.where(
            dx_loop_hfs.x < self.points["x_point"]["x"],
        )
        new_loop_hfs = Loop(
            dx_loop_hfs.x[clip_hfs],
            z=dx_loop_hfs.z[clip_hfs],
        )

        # Adding divertor entrance limits
        x_left = self.points["x_point"]["x"] - self.params.xpt_inner_gap
        x_right = self.points["x_point"]["x"] + self.params.xpt_outer_gap
        new_loop_hfs.insert([x_left, 0, self.points["x_point"]["z_low"]])
        new_loop_hfs.insert([x_left, 0, self.points["x_point"]["z_up"]])
        new_loop_lfs.insert([x_right, 0, self.points["x_point"]["z_low"]])
        new_loop_lfs.insert([x_right, 0, self.points["x_point"]["z_up"]])

        dx_loop = convex_hull([new_loop_lfs, new_loop_hfs])
        psi_n_loop = self.equilibrium.get_flux_surface(self.params.fw_psi_n)
        bottom_limit = self.points["x_point"]["z_low"] - self.x_point_shift
        top_limit = self.points["x_point"]["z_up"] + self.x_point_shift

        clip_n_loop_up = np.where(psi_n_loop.z > bottom_limit)
        new_psi_n_loop = Loop(
            psi_n_loop.x[clip_n_loop_up], z=psi_n_loop.z[clip_n_loop_up]
        )

        clip_n_loop_low = np.where(new_psi_n_loop.z < top_limit)
        new_psi_n_loop = Loop(
            new_psi_n_loop.x[clip_n_loop_low], z=new_psi_n_loop.z[clip_n_loop_low]
        )

        clip_dx_loop_up = np.where(dx_loop.z > bottom_limit)
        new_dx_loop = Loop(dx_loop.x[clip_dx_loop_up], z=dx_loop.z[clip_dx_loop_up])

        clip_dx_loop_low = np.where(new_dx_loop.z < top_limit)
        new_dx_loop = Loop(
            new_dx_loop.x[clip_dx_loop_low], z=new_dx_loop.z[clip_dx_loop_low]
        )

        hull = convex_hull([new_psi_n_loop, new_dx_loop])

        fw_loop = Loop(x=hull.x, z=hull.z)
        return fw_loop

    def modify_fw_profile(self, profile, x_int_hf, z_int_hf):
        """
        Modify the fw to reduce hf

        Parameters
        ----------
        profile: Loop
            First wall to optimise
        x_int_hf: float
            x coordinate at the inetersection
        z_int_hf: float
            z coordinate at the intersection
        heat_flux: float
            heat fluxe at the intersection

        Returns
        -------
        new_fw_profile: Loop
            Optimised profile
        """
        clipped_loops = []
        self.loops = self.equilibrium.get_flux_surface_through_point(x_int_hf, z_int_hf)

        for loop in self.loops:
            if loop_plane_intersect(loop, self.mid_plane) is not None:

                if (
                    loop_plane_intersect(loop, self.mid_plane)[0][0]
                    > self.points["o_point"]["x"]
                ):
                    clip_vertical = np.where(loop.x > self.points["x_point"]["x"])

                    clipped_loops.append(
                        self.horizontal_clipper(loop, vertical_reference=clip_vertical)
                    )

                elif loop_plane_intersect(loop, self.mid_plane)[0][0] < self.points[
                    "o_point"
                ]["x"] and loop_plane_intersect(loop, self.mid_plane)[0][0] > (
                    self.x_imp_lcfs - self.params.tk_sol_ib
                ):
                    clip_vertical = np.where(loop.x < self.points["x_point"]["x"])

                    clipped_loops.append(
                        self.horizontal_clipper(loop, vertical_reference=clip_vertical)
                    )

        if len(clipped_loops) == 0:
            new_fw_profile = profile

        elif len(clipped_loops) == 1:
            hull = convex_hull([profile, clipped_loops[0]])
            new_fw_profile = Loop(x=hull.x, z=hull.z)
            new_fw_profile.close()

        elif len(clipped_loops) == 2:
            hull = convex_hull(clipped_loops)
            hull_loop = Loop(x=hull.x, z=hull.z)
            hull_fw = convex_hull([profile, hull_loop])
            new_fw_profile = Loop(x=hull_fw.x, z=hull_fw.z)
            new_fw_profile.close()

        return new_fw_profile

    @property
    def xz_plot_loop_names(self):
        """
        The x-z loop names to plot.
        """
        names = [
            "Inboard wall",
            "Outboard wall",
            "Divertor upper",
            "Divertor cassette upper",
            "Divertor lower",
            "Divertor cassette lower",
        ]
        return names


class FirstWallPlotter(ReactorSystemPlotter):
    """
    The plotter for a First Wall and relevant Heat Flux distribution
    """

    def __init__(self):
        super().__init__()
        self._palette_key = "FW"

    def plot_xz(self, plot_objects, ax=None, **kwargs):
        """
        Plot the first wall in x-z.
        """
        super().plot_xz(plot_objects, ax=ax, **kwargs)

    def plot_hf(
        self,
        separatrix,
        loops,
        x_int,
        z_int,
        hf_int,
        fw_profile,
        koz=None,
        ax=None,
        **kwargs,
    ):
        """
        Plots the 2D heat flux distribution.

        Parameters
        ----------
        separatrix: Union[Loop, MultiLoop]
            The separatrix loop(s) (Loop for SN, MultiLoop for DN)
        loops: [MultiLoop]
            The flux surface loops
        x_int: [float]
            List of all the x coordinates at the intersections of concern
        z_int: [float]
            List of all the z coordinates at the intersections of concern
        hf_int: [float]
            List of all hf values at the intersections of concern
        fw_profile: Loop
            Inner profile of a First wall
        koz: Loop
            Loop representing the keep-out-zone
        ax: Axes, optional
            The optional Axes to plot onto, by default None.
        """
        fw_profile.plot(ax=ax, fill=False, edgecolor="k", linewidth=1)
        for loop in loops:
            loop.plot(ax=ax, fill=False, edgecolor="r", linewidth=0.2)

        separatrix.plot(ax=ax, fill=False, edgecolor="r", linewidth=1)
        if koz is not None:
            koz.plot(ax=ax, fill=False, edgecolor="g", linewidth=1)
        ax = plt.gca()
        cs = ax.scatter(x_int, z_int, s=25, c=hf_int, cmap="viridis", zorder=100)
        bar = plt.gcf().colorbar(cs, ax=ax)
        bar.set_label("Heat Flux [MW/m^2]")
