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
import numpy as np
import matplotlib.pyplot as plt
from typing import Type

from bluemira.base.parameter import ParameterFrame

from BLUEPRINT.base.baseclass import ReactorSystem
from BLUEPRINT.base.error import SystemsError, GeometryError
from BLUEPRINT.cad.firstwallCAD import FirstWallCAD
from bluemira.equilibria.find import find_flux_surfs
from BLUEPRINT.geometry.loop import Loop, MultiLoop
from BLUEPRINT.geometry.shell import Shell
from BLUEPRINT.geometry.boolean import (
    convex_hull,
    boolean_2d_union,
    boolean_2d_difference,
    boolean_2d_difference_loop,
    boolean_2d_difference_split,
    boolean_2d_common_loop,
    simplify_loop,
)
from BLUEPRINT.geometry.geomtools import (
    get_intersect,
    check_linesegment,
    loop_plane_intersect,
    index_of_point_on_loop,
    make_box_xz,
)
from BLUEPRINT.geometry.geombase import make_plane
from BLUEPRINT.geometry.geomtools import rotate_vector_2d
from functools import partial
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


class EqInputs:
    """
    Class to extract equilibrium parameters relevant to the first wall

    Parameters
    ----------
    lcfs_shift: float
        Sometimes the separatrix is not well defined.
        This parameter take an open flux surface, close to the lcfs
        to replace the separatrix
    x_point_shift:
        A shift to slightly move away from the x-point and avoid singularities

    Attributes
    ----------
    self.lcfs_shift: float
        Save input parameter used to set separatrix
    self.x_point_shift: float
        Save input parameter used to set separatrix
    self.lcfs: Loop
        Contour corresponding to last closed flux surface
    self.points: dict
        Store the x,z coordinates of the O and X points in the flux field
    self.mid_plane: Plane
        A plane having z-normal and containing the O point
    self.x_omp_lcfs: float
        Outer x-coordinate of the intersection between the last closed flux
        surface and the mid-plane
    self.x_imp_lcfs: float
        Inner x-coordinate of the intersection between the last closed flux
        surface and the mid-plane
    self.sep: separatrix: Union[Loop, MultiLoop]
        The separatrix loop(s) (Loop for SN, MultiLoop for DN)

    Notes
    -----
    According to the refinement of the eqdsk file, the separatrix
    is extrapolated differently.
    For the SN, there are apparently no issues and the separatrix always
    comes as a closed loop that contains the main plasma and an
    open loop for the two legs. The shared point is the x-point.
    For the DN, if the separatrix is extrapolated correctly, it is a
    Multiloop with two open loops that share two points: upper and lower
    x-point. The two loops are the half separatrix at the lfs and
    half separatrix at the hfs.
    If the separatrix is not extrapolated correctly, this turns out to be
    an upside down "SN like separatrix". Thus, only the upper legs exist.
    """

    def __init__(self, lcfs_shift=0.001, x_point_shift=0.1):

        # Save inputs
        self.lcfs_shift = lcfs_shift
        self.x_point_shift = x_point_shift

        # First find the last closed flux surface
        self.lcfs = self.equilibrium.get_LCFS()

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

        # Find the intersection between the mid-plane and the last closed
        # flux surface
        mp_ints = loop_plane_intersect(self.lcfs, self.mid_plane)

        # Find the outer and inner mid-plane intersections
        self.x_omp_lcfs = find_outer_point(mp_ints, self.points["o_point"]["x"])[0]
        self.x_imp_lcfs = find_inner_point(mp_ints, self.points["o_point"]["x"])[0]

        # Here we check if it is a DN. If it is a DN, we have two mirrored x-points.
        if round(self.points["x_point"]["z_low"], 3) == -round(
            self.points["x_point"]["z_up"], 3
        ):
            # Here we pick the separatrix (right or wrong).
            self.sep = self.equilibrium.get_separatrix()

            # We check the separatrix against the lower x-point and we move
            # away from it to avoid a singularity.
            x_point_limit = self.points["x_point"]["z_low"] - self.x_point_shift

            # Here we take the lowest point contained in the separatrix loop.
            sep_limit = min(self.sep[0].z)

            # If we find a point, below the fixed limit, it means that the lower
            # legs are present, and the extrapolated sepatrix is correct.
            if sep_limit < x_point_limit:
                self.separatrix = self.sep

            # If we do not find a point below the fixed limit, it means that the
            # lower legs are not present. The extrapolated separatrix is not correct.
            else:

                # We need to "make" a separatrix.
                # As separatrix we take the first open flux surface (Ideally,
                # the first flux surfaces outside the LCFS).
                loops = self.equilibrium.get_flux_surface_through_point(
                    self.x_omp_lcfs + self.lcfs_shift, 0
                )

                # We re-order the loops as conventionally is done for the
                # separatrix and we make it a MultiLoop, as conventionally is
                # done for the separatrix.
                loops.reverse()
                self.separatrix = MultiLoop(loops[:2])

        # Here we check if it is a SN. If it is a SN, we do have more than one
        # x-points, but they are not mirrored.
        elif round(self.points["x_point"]["z_low"], 3) != -round(
            self.points["x_point"]["z_up"], 3
        ):
            # As the SN does not give concerns, we just have to pick
            # the separatrix.
            self.separatrix = [self.equilibrium.get_separatrix()]


class FluxSurface(EqInputs):
    """
    Create a flux surface
    Evaluate all needed attributes which are useful for to the heat flux calculation

    Parameters
    ----------
    equilibrium: eqdsk, geqdsk or json
    x: float
       x coordinate of the midplane point from which the fs has to pass
    z: float
       z coordinate of the midplane point from which the fs has to pass

    Attributes
    ----------
    self.loops: [Loop]
        Each flux surfaces (fs) can have more than one loop
    self.loop_hfs: Loop
        In case of double null, it is the part of the fs on the inboard
    self.loop_lfs: Loop
        In case of single null, it is the active part of the fs. Yhe one
        carrying power
        In case of double null, it is the part of the fs on the ouboard
    self.x_omp: float
        x coordinate of the intersection between midplane and fs
        on the ouboard
    self.z_omp: float
        z coordinate of the intersection between midplane and fs
        on the ouboard
    self.x_imp: float
        x coordinate of the intersection between midplane and fs
        on the inboard
    self.z_imp: float
        z coordinate of the intersection between midplane and fs
        on the inboard
    self.Bt_omp, self.Bp_omp, self.B_omp: float, float, float
        Toroidal, Poloidal and Total magnetic field values at the
        intersection between fs and midplane on the outboard
    self.Bt_imp, self.Bp_imp, self.B_imp: float, float, float
        Toroidal, Poloidal and Total magnetic field values at the
        intersection between fs and midplane on the inboard
    self.dr_omp, self.dr_imp: float, float
        Raidal distance between fs and the lcfs respectively at the
        outboard and inboard
    """

    def __init__(self, equilibrium, x, z):
        self.equilibrium = equilibrium
        super().__init__(self)

        self.loops = self.equilibrium.get_flux_surface_through_point(x, z)

        for loop in self.loops:
            mid_plane_intersection = loop_plane_intersect(loop, self.mid_plane)
            if mid_plane_intersection is not None:
                if len(mid_plane_intersection) == 1:  # DN
                    int_coord = mid_plane_intersection[0]
                    if int_coord[0] < self.points["o_point"]["x"]:
                        (
                            self.loop_hfs,
                            self.x_imp,
                            self.z_imp,
                            self.Bt_imp,
                            self.Bp_imp,
                            self.B_imp,
                            self.dr_imp,
                        ) = self.define_key_attributes(int_coord, loop, omp=False)
                    if int_coord[0] > self.points["o_point"]["x"]:
                        (
                            self.loop_lfs,
                            self.x_omp,
                            self.z_omp,
                            self.Bt_omp,
                            self.Bp_omp,
                            self.B_omp,
                            self.dr_omp,
                        ) = self.define_key_attributes(int_coord, loop, omp=True)
                if len(mid_plane_intersection) == 2:  # SN
                    for int_coord in mid_plane_intersection:
                        if int_coord[0] > self.points["o_point"]["x"]:
                            (
                                self.loop_lfs,
                                self.x_omp,
                                self.z_omp,
                                self.Bt_omp,
                                self.Bp_omp,
                                self.B_omp,
                                self.dr_omp,
                            ) = self.define_key_attributes(int_coord, loop, omp=True)

    def define_key_attributes(self, mp_int_coords, loop, omp=True):
        """
        Define key attributes for a flux surface

        Parameters
        ----------
        mp_int_coords: [float, float, float]
            x, y, z of intersection between midplane and flux surface
        loop: Loop
            flux surface
        omp: in case of double null, the attributes need to be assigned to
            the inner midplane as well. In this case, one should set omp = False

        Returns
        -------
        loop: Loop
            flux surface loop
        x_mp: float
            x coordinate of the midplane intersection
        z_mp: float
            z coordinate of the midplane intersection
        Bt_mp: float
            Toroidal magnetic field at the intersection
        Bp_mp: float
            Poloidal magnetic field at the intersection
        B_mp: float
            Magnetic field at the intersection
        dr_mp: float
            midplane distance between fs and lcfs
        """
        loop = loop
        x_mp = mp_int_coords[0]
        z_mp = mp_int_coords[2]
        Bt_mp = self.equilibrium.Bt(x_mp)
        Bp_mp = self.equilibrium.Bp(x_mp, z_mp)
        B_mp = np.sqrt(Bt_mp ** 2 + Bp_mp ** 2)
        dr_mp = x_mp - self.x_omp_lcfs
        if omp:
            dr_mp = x_mp - self.x_omp_lcfs
        else:
            dr_mp = -(x_mp - self.x_imp_lcfs)
        return (loop, x_mp, z_mp, Bt_mp, Bp_mp, B_mp, dr_mp)

    def find_intersections(self, flux_surface, fw_profile):
        """
        Find intersection between a flux surface and a given first wall profile

        Parameters
        ----------
        flux_surface: Loop
            A single flux surface. Not a set
        fw_profile: Loop
            A first wall 2D profile

        Returns
        -------
        x_int : np.array (n intersections)
            x coordinate of intersections
        z_int : np.array (n intersections)
            z coordinate of intersections
        """
        x_int, z_int = get_intersect(fw_profile, flux_surface)
        return (x_int, z_int)

    def assign_lfs_hfs_sn(self, x_int, z_int):
        """
        Assign intersection points either to the low field side or the high field side
        Applicable to the SN configuration

        Parameters
        ----------
        x_int : [float]
            x coordinate of intersections
        z_int : [float]
            z coordinate of intersections
        theta_coord: [float]
            theta coordinate of intersections

        Returns
        -------
        int_points_lfs : np.array, np.array, np.array
            x, z and theta coordinates of all the intersections on the lfs
        int_points_hfs : np.array, np.array, np.array
            x, z and theta coordinates of all the intersections on the hfs
        """
        lfs_ind = np.where(
            (x_int > self.points["x_point"]["x"]) & (z_int < self.points["o_point"]["z"])
        )
        hfs_ind = np.where(
            ~(
                (x_int > self.points["x_point"]["x"])
                & (z_int < self.points["o_point"]["z"])
            )
        )

        int_points_lfs = np.zeros((2, lfs_ind[0].size))
        int_points_hfs = np.zeros((2, hfs_ind[0].size))

        for no, i in enumerate(lfs_ind[0]):
            int_points_lfs[0, no] = x_int[i]
            int_points_lfs[1, no] = z_int[i]

        for no, i in enumerate(hfs_ind[0]):
            int_points_hfs[0, no] = x_int[i]
            int_points_hfs[1, no] = z_int[i]

        return (
            int_points_lfs,
            int_points_hfs,
        )

    def assign_top_bottom(self, x_int, z_int):
        """
        Assign intersection points either to the top part of the chamber or the bottom
        Applicable to the DN configuration

        Parameters
        ----------
        x_int : [float]
            x coordinate of intersections
        z_int : [float]
            z coordinate of intersections

        Returns
        -------
        top_intersections: np.array, np.array
            x and z coordinates of all the intersections on the
            top part of the chamber
        bottom_intersections: np.array, np.array
            x and z coordinates of all the intersections on the
            bottom part of the chamber
        """
        top_intersections = np.zeros(2)
        bottom_intersections = np.zeros(2)

        top_ind = np.where((z_int > self.z_omp))
        bottom_ind = np.where((z_int < self.z_omp))

        top_intersections = x_int[top_ind], z_int[top_ind]
        bottom_intersections = x_int[bottom_ind], z_int[bottom_ind]

        return (
            top_intersections,
            bottom_intersections,
        )

    def cut_flux_line_portion(self, loop, point_1, point_2):
        """
        Cuts a flux line (loop) between two end points

        Parameters
        ----------
        loop : Loop
            loop object
        point_1 : [float, float]
            Initial point in [x, z] coordinates
        point_2 : [float, float]
            Final point in [x, z] coordinates

        Returns
        -------
        new_loop: Loop
            Portion of the initial loop
        """
        d_ref = self.distance_between_two_points_on_a_loop(loop, point_1, point_2)
        d_loop = []
        for x, z in zip(loop.x, loop.z):
            d = self.distance_between_two_points_on_a_loop(loop, point_1, [x, z])
            d_loop.append(d)

        p_ind = np.where(d_loop < d_ref)
        if len(p_ind[0]) > 1:
            new_loop = Loop(x=loop.x[p_ind], z=loop.z[p_ind])
        else:
            new_loop = loop
        return new_loop

    def distance_between_two_points_on_a_loop(self, loop, point_1, point_2):
        """
        Calcultes the distance between two generic points on a loop

        Parameters
        ----------
        loop : Loop
            loop object
        point_1 : [float, float]
            Initial point in [x, z] coordinates
        point_2 : [float, float]
            Final point in [x, z] coordinates

        Returns
        -------
        distance: float
            The distance between the points along the given path (loop)
        """
        coeffs = np.polyfit(loop.x, loop.z, 1)
        func = np.poly1d(coeffs)

        new_a_coeff = (point_2[1] - point_1[1]) / (func(point_2[0]) - func(point_1[0]))
        new_b_coeff = point_1[1] - new_a_coeff * func(point_1[0])

        x_p = np.linspace(point_1[0], point_2[0], 10)
        z_p = new_a_coeff * (func(x_p)) + new_b_coeff

        i = np.arange(9) if x_p.size > 9 else np.arange(x_p.size)
        distance = sum(np.hypot(x_p[i + 1] - x_p[i], z_p[i + 1] - z_p[i]))

        return distance

    def flux_surface_sub_loop(self, loop, double_null=True):
        """
        Splits the flux line (loop) in two parts.
        If SN, the split is meant to be between lfs and hfs.
        If DN, the split is meant to be between the part
        above the mid-plane and below the mid-plane.

        Parameters
        ----------
        loop : Loop
            flux surface loop
        double_null : Boolean
            Boolean set as True (default) means DN. If False, it is SN.

        Returns
        -------
        clipped_loop_up : Loop
            Part of the loop above the mid-plane.
            This represents the hfs in case of SN
        clipped_loop_down : Loop
            Part of the loop below the mid-plane.
            This represents the lfs in case of SN
        """
        # First we check if the flux surface is not empty
        if loop.point_inside([self.x_omp, self.z_omp], True) or loop.point_inside(
            [self.x_imp, self.z_imp], True
        ):
            # In case of DN, we split the flux surface between up and down
            if double_null:
                clip_up = np.where(loop.z > self.z_omp - 0.01)
                clipped_loop_up = Loop(loop.x[clip_up], z=loop.z[clip_up])

                clip_down = np.where(loop.z < self.z_omp + 0.01)
                clipped_loop_down = Loop(loop.x[clip_down], z=loop.z[clip_down])

            # Only alternative case, at the moment, is SN
            else:
                clip_hfs = np.where(
                    ~(
                        (loop.x > self.points["x_point"]["x"])
                        & (loop.z < self.points["o_point"]["z"])
                    )
                )
                clipped_loop_hfs = Loop(loop.x[clip_hfs], z=loop.z[clip_hfs])
                clipped_loop_up = clipped_loop_hfs

                clip_lfs = np.where(
                    (loop.x > self.points["x_point"]["x"])
                    & (loop.z < self.points["o_point"]["z"])
                )
                clipped_loop_lfs = Loop(loop.x[clip_lfs], z=loop.z[clip_lfs])
                clipped_loop_down = clipped_loop_lfs

        return clipped_loop_up, clipped_loop_down

    def find_first_intersection(
        self,
        loop,
        intersection_points_x,
        intersection_points_z,
        lfs=True,
        double_null=True,
    ):
        """
        Find the first intersection point between the flux line and the first wall.

        Parameters
        ----------
        loop: Loop
            Portion of loop, with initial (or final) on the mid-plane.
        intersection_points_x : [float]
            List of x coordinates of the intersection points between
            the given loop and the first wall.
        intersection_points_z : [float]
            List of z coordinates of the intersection points between
            the given loop and the first wall.
        lfs: Boolean (default = True)
            Referral magnetic field region
        double_null: Boolean (default = True)
            Referral plasma configuration

        Returns
        -------
        first_intersection: [float, float]
            x and z coordinates of the first intersection.
        """
        if (double_null and lfs) or not double_null:
            p0 = [self.x_omp, self.z_omp]
        else:
            p0 = [self.x_imp, self.z_imp]

        p_dist = []
        for x, z in zip(intersection_points_x, intersection_points_z):
            p_next = [x, z]
            d = self.distance_between_two_points_on_a_loop(loop, p0, p_next)
            p_dist.append(d)

        first_int_p_ind = np.argmin(p_dist)

        first_intersection = [
            intersection_points_x[first_int_p_ind],
            intersection_points_z[first_int_p_ind],
        ]

        return first_intersection

    def calculate_q_par_omp(self, Psol_near, Psol_far, lambdaq_near, lambdaq_far):
        """
        Calculate the parallel power density at the outer midplane

        Parameters
        ----------
        Psol_near: float
            Power entering the near SOL at the midplane
        Psol_far: float
            Power entering the far SOL at the midplane
        lambda_near: float
            Decay length of near SOL
        lambda_far: float
            Decay length of far SOL

        Returns
        -------
        qpar_omp: float
            Parallel contribution of the power carried by the fs at the omp
        """
        q_omp = (
            (Psol_near * np.exp(-self.dr_omp / lambdaq_near))
            / (2 * np.pi * self.x_omp * lambdaq_near)
        ) + (
            (Psol_far * np.exp(-self.dr_omp / lambdaq_far))
            / (2 * np.pi * self.x_omp * lambdaq_far)
        )
        qpar_omp = q_omp * self.B_omp / self.Bp_omp
        return qpar_omp

    def calculate_q_par_imp(self, Psol_near, Psol_far, lambdaq_near, lambdaq_far):
        """
        Calculate the parallel power density at the inner midplane

        Parameters
        ----------
        Psol_near: float
            Power entering the near SOL at the midplane
        Psol_far: float
            Power entering the far SOL at the midplane
        lambda_near: float
            Decay length of near SOL
        lambda_far: float
            Decay length of far SOL

        Returns
        -------
        qpar_imp: float
            Parallel contribution of the power carried by the fs at the imp
        """
        q_imp = (
            (Psol_near * np.exp(-self.dr_imp / lambdaq_near))
            / (2 * np.pi * self.x_imp * lambdaq_near)
        ) + (
            (Psol_far * np.exp(-self.dr_imp / lambdaq_far))
            / (2 * np.pi * self.x_imp * lambdaq_far)
        )
        qpar_imp = q_imp * self.B_imp / self.Bp_imp
        return qpar_imp

    def calculate_q_par_local(self, x_int, z_int, qpar_omp):
        """
        Calculate the parallel power density associated to a given point
        In case of double null, this is meant for the lfs

        Parameters
        ----------
        x_int: float
            x coordinate of intersection point
        z_int: float
            z coordinate of intersection point
        qpar_omp: float
            Parallel contribution of the power carried by the fs at the omp

        Returns
        -------
        qpar_local: float
            Parallel power density associated to the intersection point
        """
        Bp_local = self.equilibrium.Bp(x_int, z_int)
        self.f = (self.x_omp * self.Bp_omp) / (x_int * Bp_local)
        qpar_local = (
            qpar_omp * (self.Bp_omp / self.B_omp) * (self.x_omp / x_int) * (1 / self.f)
        )
        return qpar_local

    def calculate_q_par_local_hfs(self, x_int, z_int, qpar_imp):
        """
        Calculate the parallel power density associated to a given point
        Apllicable to the DN

        Parameters
        ----------
        x_int: float
            x coordinate of intersection point
        z_int: float
            z coordinate of intersection point
        qpar_imp: float
            Parallel contribution of the power carried by the fs at the imp

        Returns
        -------
        qpar_local_hfs: float
            Parallel power density associated to the intersection point at the hfs
        """
        Bp_local = self.equilibrium.Bp(x_int, z_int)
        self.f_hfs = (self.x_imp * self.Bp_imp) / (x_int * Bp_local)
        qpar_local_hfs = (
            qpar_imp
            * (self.Bp_imp / self.B_imp)
            * (self.x_imp / x_int)
            * (1 / self.f_hfs)
        )
        return qpar_local_hfs

    def calculate_incindent_angle(self, loop, x_int, z_int, fw_profile):
        """
        Calculate the incindent angle, on the poloidal plane, for a given
        intersection between a flux surface and the first wall

        Parameters
        ----------
        loop: Loop
            flux surface given as loop object
        x_int: float (single value)
            x coordinate of intersection point
        z_int : float (single value)
            z coordinate of intersection point
        fw_profile: Loop
            first wall 2D profile

        Returns
        -------
        incindent_angle: float
            Angle formed at the intersection between fs and first wall (deg)
        incindent_angle_rad: float
            Angle formed at the intersection between fs and first wall (rad)
        """
        x_int = x_int
        z_int = z_int
        fs_p_ind = np.roll(np.arange(loop.x.size), -1)
        arr = np.array([[loop.x, loop.z], [loop.x[fs_p_ind], loop.z[fs_p_ind]]])
        int_ind = np.where(
            [
                check_linesegment(arr[0, :, i], arr[1, :, i], [x_int, z_int])
                for i in range(loop.x.size)
            ]
        )
        p_before_int = (loop.x[int_ind][0], loop.z[int_ind][0])

        if len(p_before_int) != 0:

            fw_p_ind = np.roll(np.arange(fw_profile.x.size), -1)
            arr = np.array(
                [
                    [fw_profile.x, fw_profile.z],
                    [fw_profile.x[fw_p_ind], fw_profile.z[fw_p_ind]],
                ]
            )
            int_ind = np.where(
                [
                    check_linesegment(arr[0, :, i], arr[1, :, i], [x_int, z_int])
                    for i in range(fw_profile.x.size)
                ]
            )
            p_before_int_wall = (fw_profile.x[int_ind][0], fw_profile.z[int_ind][0])

            v0 = p_before_int - np.array([x_int, z_int])
            v1 = p_before_int_wall - np.array([x_int, z_int])

            v0_u = v0 / np.linalg.norm(v0)
            v1_u = v1 / np.linalg.norm(v1)
            incindent_angle = (np.arccos(np.clip(np.dot(v0_u, v1_u), -1.0, 1.0))) * (
                180 / np.pi
            )
            incindent_angle_rad = (
                ((np.arccos(np.clip(np.dot(v0_u, v1_u), -1.0, 1.0))) * (180 / np.pi))
            ) * (np.pi / 180)

        return (incindent_angle, incindent_angle_rad)

    def calculate_heat_flux_onto_fw_surface(self, qpar, incindent_angle):
        """
        Project the power density carried by a flux surface onto the fw surface

        Parameters
        ----------
        qpar: float
            Parallel power density associated to a given point
        incindent_angle: float
            Incident angle in deg

        Returns
        -------
        qpar * np.sin(incindent_angle): float
            This is the actual heat flux -> Power over a surface
        """
        return qpar * np.sin(incindent_angle)


class FirstWall(EqInputs, ReactorSystem):
    """
    Reactor First Wall (FW) system abstract base class
    """

    config: Type[ParameterFrame]
    inputs: dict
    CADConstructor = FirstWallCAD

    # fmt: off
    base_default_params = [
        ["n_TF", "Number of TF coils", 16, "N/A", None, "Input"],
        ["A", "Plasma aspect ratio", 3.1, "N/A", None, "Input"],
        ["psi_norm", "Normalised flux value of strike-point contours",
         1, "N/A", None, "Input"],
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
        self.init_params()
        self.init_equilibrium()
        self.build()
        self.build_fs_to_plot()
        self._plotter = FirstWallPlotter()

    def init_params(self):
        """
        Initialise First Wall parameters from config.
        """
        self._init_params(self.config)

    def init_equilibrium(self):
        """
        Initialise equilibrium inputs.
        """
        self.equilibrium = self.inputs["equilibrium"]
        EqInputs.__init__(self)

    def build(self):
        """
        Build the 2D profile
        """
        if "profile" in self.inputs:
            self.profile = self.inputs["profile"]
            self.make_flux_surfaces()

        elif self.inputs.get("FW_optimisation", False):
            self.profile = self.optimise_fw_profile()

        else:
            self.profile = self.make_preliminary_profile()
            self.make_flux_surfaces()

        self.make_2d_profile()

    def build_fs_to_plot(self):
        """
        Build flux surfaces to be plotted in the heat flux plot.
        """
        self.hf_firstwall_params(self.inner_profile)
        fs_loops = self.lfs_flux_line_portion + self.hfs_flux_line_portion
        self.fs = MultiLoop(fs_loops)
        self.geom["fs"] = self.fs

    def hf_firstwall_params(self):
        """
        Generate the parameters to calculate the heat flux
        """
        raise NotImplementedError

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

    def make_preliminary_profile(self):
        """
        Generate a preliminary first wall profile in case it is not given as input
        """
        raise NotImplementedError

    def make_flux_surfaces(self, step_size=0.005, profile=None):
        """
        Generate a set of flux surfaces placed between lcfs and fw
        """
        raise NotImplementedError

    def find_intersections(self):
        """
        Find the intersections between all the flux surfaces and the first wall
        """
        raise NotImplementedError

    def find_first_intersections(self):
        """
        Find the first intersections between all the flux surfaces and the first wall
        """
        raise NotImplementedError

    def optimise_fw_profile(self, hf_limit=0.2, n_iteration_max=5):
        """
        Optimises the initial preliminary profile in terms of heat flux.
        The divertor will be attached to this profile.

        Parameters
        ----------
        n_iteration_max: integer
            Max number of iterations after which the optimiser is stopped.
        hf_limit: float
            Heat flux limit for the optimisation.

        Returns
        -------
        profile: Loop
            Optimised profile
        """
        initial_profile = self.make_preliminary_profile()
        self.preliminary_profile = initial_profile
        self.make_flux_surfaces(profile=initial_profile)

        profile = initial_profile
        for i_it in range(n_iteration_max):

            x_wall, z_wall, hf_wall = self.hf_firstwall_params(profile)

            for x_hf, z_hf, hf in zip(x_wall, z_wall, hf_wall):
                if hf > hf_limit:
                    profile = self.modify_fw_profile(profile, x_hf, z_hf)

            heat_flux_max = max(hf_wall)
            print(heat_flux_max)
            self.optimised_profile = profile
            if heat_flux_max < hf_limit:
                break

        return profile

    def reshape_curve(
        self,
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
            target_par = rotate_vector_2d(tangent, np.radians(theta_target * sign))
        # if vertical target
        else:
            target_par = rotate_vector_2d(tangent, np.radians(-theta_target * sign))

        # Create relative vectors whose length will be the offset distance
        # from the strike point
        pfr_target_end = -target_par * target_length_pfr * sign
        sol_target_end = target_par * target_length_sol * sign

        # Swap if we got the wrong way round
        if outer_target:
            swap_points = sol_target_end[0] < pfr_target_end[0]
        # for the inner target
        else:
            swap_points = (
                not vertical_target and sol_target_end[0] > pfr_target_end[0]
            ) or (vertical_target and sol_target_end[0] < pfr_target_end[0])

        if swap_points:
            tmp = pfr_target_end
            pfr_target_end = sol_target_end
            sol_target_end = tmp

        # Add the strike point to diffs to get the absolute positions
        # of the end points of the target
        pfr_target_end = pfr_target_end + strike_point
        sol_target_end = sol_target_end + strike_point

        # Return end points
        return (pfr_target_end, sol_target_end)

    def get_tangent_vector(self, point_on_loop, loop):
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
        return tangent_norm

    def make_guide_line(self, initial_loop, top_limit, bottom_limit):
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
        tangent = self.get_tangent_vector(outer_strike, flux_loop)

        # Get the outer target points
        (
            outer_target_internal_point,
            outer_target_external_point,
        ) = self.make_divertor_target(
            outer_strike,
            tangent,
            vertical_target=self.inputs["div_vertical_outer_target"],
            outer_target=True,
        )

        # Select the degree of the fitting polynomial and
        # the flux lines that will guide the divertor leg shape
        if self.inputs.get("DEMO_DN", False):
            degree_in = degree_out = self.inputs.get(
                "outer_leg_sol_polyfit_degree",
                self.inputs.get("outer_leg_pfr_polyfit_degree", 1),
            )
            outer_leg_external_guide_line = outer_leg_internal_guide_line = flux_loop
        elif self.inputs.get("SN", False):
            degree_in = degree_out = self.inputs.get(
                "outer_leg_sol_polyfit_degree",
                self.inputs.get("outer_leg_pfr_polyfit_degree", 2),
            )
            outer_leg_external_guide_line = outer_leg_internal_guide_line = flux_loop
        else:
            degree_out = self.inputs.get("outer_leg_sol_polyfit_degree", 3)
            degree_in = self.inputs.get("outer_leg_pfr_polyfit_degree", 3)
            outer_leg_external_guide_line = self.flux_surface_lfs[-1]
            outer_leg_internal_guide_line = flux_loop

        # Select the top and bottom limits for the guide lines
        z_x_point = self.points["x_point"]["z_low"]
        outer_leg_external_top_limit = [div_top_right, z_x_point]
        outer_leg_external_bottom_limit = outer_target_external_point

        outer_leg_internal_top_limit = middle_point
        outer_leg_internal_bottom_limit = outer_target_internal_point

        # Make the guide lines
        external_guide_line = self.make_guide_line(
            outer_leg_external_guide_line,
            outer_leg_external_top_limit,
            outer_leg_external_bottom_limit,
        )

        internal_guide_line = self.make_guide_line(
            outer_leg_internal_guide_line,
            outer_leg_internal_top_limit,
            outer_leg_internal_bottom_limit,
        )

        # Modify the clipped flux line curve (guide line) to start
        # at the middle and end at the internal point of the outer target
        (outer_leg_internal_line_x, outer_leg_internal_line_z,) = self.reshape_curve(
            internal_guide_line.x,
            internal_guide_line.z,
            [middle_point[0], middle_point[1]],
            outer_target_internal_point,
            degree_in,
        )

        # Modify the clipped flux line curve to start at the top point of the
        # outer target and end at the external point
        (outer_leg_external_line_x, outer_leg_external_line_z,) = self.reshape_curve(
            external_guide_line.x,
            external_guide_line.z,
            [div_top_right, z_x_point],
            outer_target_external_point,
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
        tangent = self.get_tangent_vector(inner_strike, flux_loop)

        if self.inputs.get("DEMO_DN", False):
            degree = self.inputs.get("inner_leg_polyfit_degree", 1)
        else:
            degree = self.inputs.get("inner_leg_polyfit_degree", 2)

        # Get the outer target points
        (
            inner_target_internal_point,
            inner_target_external_point,
        ) = self.make_divertor_target(
            inner_strike,
            tangent,
            vertical_target=self.inputs["div_vertical_inner_target"],
            outer_target=False,
        )

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
            inner_leg_central_guide_line.z > inner_target_internal_point[1],
        )

        # Create a new Loop from the points selected along the flux line
        inner_leg_central_guide_line = Loop(
            x=inner_leg_central_guide_line.x[bottom_clip_inner_leg_central_guide_line],
            y=None,
            z=inner_leg_central_guide_line.z[bottom_clip_inner_leg_central_guide_line],
        )

        # Modify the clipped flux line curve to start at the middle and end
        # at the internal point of the outer target
        (inner_leg_internal_line_x, inner_leg_internal_line_z,) = self.reshape_curve(
            inner_leg_central_guide_line.x,
            inner_leg_central_guide_line.z,
            [middle_point[0], middle_point[1]],
            inner_target_internal_point,
            degree,
        )

        # Modify the clipped flux line curve to start at the top point of the
        # outer target and end at the external point
        (inner_leg_external_line_x, inner_leg_external_line_z,) = self.reshape_curve(
            inner_leg_central_guide_line.x,
            inner_leg_central_guide_line.z,
            [div_top_left, z_x_point],
            inner_target_external_point,
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

        if isinstance(self, FirstWallSN):
            return [bottom_divertor]

        elif isinstance(self, FirstWallDN):
            # Flip z coords to get top divertor loop
            x_div_top = bottom_divertor.x
            z_div_top = [z * -1 for z in bottom_divertor.z]
            top_divertor = Loop(x=x_div_top, z=z_div_top)
            return [bottom_divertor, top_divertor]

    def make_divertor_demo_like(self, fw_loop):
        """
        Create a DEMO like divertor loop for the single null configuration.

        Parameters
        ----------
        fw_loop: Loop
            first wall preliminary profile

        Returns
        -------
        divertor_loop: list
            List of Loops for the divertor geometry (single entry for SN)
        """
        # Some shorthands
        z_low = self.points["x_point"]["z_low"]
        x_x_point = self.points["x_point"]["x"]

        # Pick some flux loops to use to locate strike points
        flux_loops = self.pick_flux_loops()

        # Find the strike points
        inner, outer = self.find_strike_points(flux_loops)

        # Find the intersection of the first wall loop and
        # the x-y plane containing the lower X point
        z_norm = 2
        fw_int_point = get_intersection_point(
            z_low, z_norm, fw_loop, x_x_point, inner=False
        )

        # Define the left and right limits of the divertor entrance
        # relative to the separatrix x point given gap parameters
        div_left = x_x_point - self.params.xpt_inner_gap
        div_right = max(x_x_point + self.params.xpt_outer_gap, fw_int_point[0])

        # Define the x coordinates for the divertor
        x_div = [
            div_left,
            inner[0] - self.params.tk_inner_target_sol,
            inner[0] + self.params.tk_inner_target_pfr,
            outer[0] - self.params.tk_outer_target_pfr,
            outer[0] + self.params.tk_outer_target_sol,
            div_right,
        ]

        # Define the z coordinates for the divertor
        z_div = [z_low, inner[1], inner[1], outer[1], outer[1], z_low]

        # Create the loop and return as a list
        divertor_loop = Loop(x=x_div, z=z_div)
        divertor_loop.close()

        return [divertor_loop]

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

        # Make a divertor
        if self.inputs.get("DEMO_like_divertor", False):
            inner_divertor_loops = self.make_divertor_demo_like(self.profile)
        # It makes a long legs dievertor
        else:
            inner_divertor_loops = self.make_divertor(self.profile)

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

        # For now, make the divertor cassette here (to be refactored)
        self.divertor_cassettes = self.make_divertor_cassette(self.divertor_loops)

        # Make a shell from the inner and outer profile
        fw_shell = Shell(inner=self.inner_profile, outer=outer_profile)

        # Save geom objects
        self.geom["2D profile"] = fw_shell
        self.geom["Inboard wall"] = inboard_wall
        self.geom["Outboard wall"] = outboard_wall

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

    def make_flux_contour_loops(self, eq, psi_norm):
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
        sorted_dict = dict(sorted(flux_dict.items()))

        # Get list of increasing values
        sorted_flux_loops = list(sorted_dict.values())

        # By convention, want decreasing list
        sorted_flux_loops.reverse()
        return sorted_flux_loops

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

        # SN case: just one loop
        if self.inputs.get("SN", False):
            outer_loop = inner_loop = flux_loops[0]
        else:
            outer_loop = flux_loops[0]
            inner_loop = flux_loops[1]

        # Get the inner intersection with the separatrix
        inner_strike_x = self.params.inner_strike_r
        x_norm = 0
        # Does it make sense to compare x with x-norm??
        inner_strike_z = get_intersection_point(
            inner_strike_x, x_norm, inner_loop, x_x_point, inner=True
        )[2]

        # Get the outer intersection with the separatrix
        outer_strike_x = self.params.outer_strike_r
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
            flux_loops = self.make_flux_contour_loops(self.equilibrium, psi_norm)
        else:
            # Default: use separatrix
            flux_loops = self.separatrix
        return flux_loops

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
        if type(separatrix) is list:
            separatrix = separatrix[0]
        self._plotter.plot_hf(
            separatrix,
            self.geom["fs"],
            self.x_wall,
            self.z_wall,
            self.hf_wall,
            self.inner_profile,
            koz,
            ax=ax,
            **kwargs,
        )

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
        inboard = inboard.offset_clipper(tk_inboard, method="miter")
        outboard = outboard.offset_clipper(tk_outboard, method="miter")
        offset_divertor_loops = []
        for div in divertor_loops:
            offset_divertor_loops.append(div.offset_clipper(tk_div, method="miter"))

        # Remove the overlaps between the offset sections
        sections = self.get_non_overlapping(
            inboard, outboard, offset_divertor_loops, cutters
        )

        # Subtract the inner profile from each component
        for i, sec in enumerate(sections):
            sections[i] = boolean_2d_difference_loop(sec, inner_wall)

        # Now find the union of our offset loops and the original profile
        outer_wall = self.attach_divertor(inner_wall, sections)

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

    def get_non_overlapping(self, inboard, outboard, divertor_loops, cutters):
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

    def make_divertor_cassette(self, divertor_loops):
        """
        Given the divertor loops create the divertor cassette.

        Parameters
        ----------
        divertor_loops : list
            List of Loop objects representing the divertor

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
            div_offset = div.offset_clipper(offset, method="miter")

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

            # Subtract the divertor (get an inner and outer piece)
            subtracted = boolean_2d_difference(cassette, div)
            if not len(subtracted) == 2:
                raise GeometryError("Unexpected number of loops")
            # Select the outer one
            if np.max(subtracted[0].x) > np.max(subtracted[1].x):
                cassette = subtracted[0]
            else:
                cassette = subtracted[1]

            # Finally, simplify
            cassette = simplify_loop(cassette)
            cassettes.append(cassette)

        return cassettes

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
        # ["plasma_type", "Type of plasma", "SN", "N/A", None, "Input"],
        ["fw_dx", "Minimum distance of FW to separatrix", 0.3, "m", None, "Input"],
        ["fw_psi_n", "Normalised psi boundary to fit FW to", 1.01, "N/A", None, "Input"],
        ["fw_p_sol_near", "near Scrape off layer power", 50, "MW", None, "Input"],
        ["fw_p_sol_far", "far Scrape off layer power", 50, "MW", None, "Input"],
        ["fw_lambda_q_near", "Lambda q near SOL", 0.05, "m", None, "Input"],
        ["fw_lambda_q_far", "Lambda q far SOL", 0.05, "m", None, "Input"],
        ["f_outer_target", "Power fraction", 0.75, "N/A", None, "Input"],
        ["f_inner_target", "Power fraction", 0.25, "N/A", None, "Input"],
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

    def make_preliminary_profile(self):
        """
        Generate a preliminary first wall profile in case it is not given as input

        Returns
        -------
        fw_loop: Loop
            Here the first wall is without divertor. The wall is cut at the X-point
        """
        dx_loop = self.lcfs.offset(self.params.fw_dx)
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

    def make_flux_surfaces(self, step_size=0.02, profile=None):
        """
        Generate a set of flux surfaces placed between lcfs and fw

        Parameters
        ----------
        step_size: float
            Defines the thickness of each flux surface at the midplane

        """
        self.flux_surfaces = []
        x_omp = self.x_omp_lcfs + self.lcfs_shift
        double_step = 2 * step_size
        if profile is None:
            profile = self.profile

        # Find intersections between the profile and mid-plane
        profile_ints = loop_plane_intersect(profile, self.mid_plane)

        # Retrieve x of outer intersection
        profile_x_omp = find_outer_point(profile_ints, self.points["o_point"]["x"])[0]

        while x_omp < profile_x_omp:
            flux_surface = FluxSurface(
                self.equilibrium, x_omp, self.points["o_point"]["z"]
            )
            if hasattr(flux_surface, "x_omp"):
                x_omp = flux_surface.x_omp
                self.flux_surfaces.append(flux_surface)
                if (
                    flux_surface.x_omp - flux_surface.x_omp_lcfs
                ) < self.params.fw_lambda_q_near:
                    x_omp += step_size
                else:
                    x_omp += double_step

        if len(self.flux_surfaces) == 0:
            raise ValueError(f"fs for initial psi = {self.params.fw_psi_n} outside fw")

        for fs in self.flux_surfaces:  # exclude empty flux surfaces
            if hasattr(fs, "loop_lfs"):
                x_int, z_int = fs.find_intersections(fs.loop_lfs, profile)
                if len(x_int) == 0:
                    self.flux_surfaces.remove(fs)

        if len(self.flux_surfaces) == 0:
            raise ValueError(
                "No intersections found between Flux Surfaces and First Wall."
            )

        self.flux_surface_width_omp = []
        dr_0 = self.flux_surfaces[0].x_omp - self.x_omp_lcfs
        self.flux_surface_width_omp.append(dr_0)
        for i in range(len(self.flux_surfaces)):
            dr_omp = (
                self.flux_surfaces[(i + 1) % len(self.flux_surfaces)].dr_omp
                - self.flux_surfaces[i].dr_omp
            )
            self.flux_surface_width_omp.append(dr_omp)
        self.flux_surface_width_omp = self.flux_surface_width_omp[:-1]

    def q_parallel_calculation(self):
        """
        Calculate q parallel at OMP for all the flux surfaces

        Returns
        -------
        qpar_omp: [float]
            Parallel contribution of the power carried by all the fs at the omp
        """
        qpar_omp = []

        for fs in self.flux_surfaces:
            if hasattr(fs, "dr_omp"):
                q = fs.calculate_q_par_omp(
                    self.params.fw_p_sol_near,
                    self.params.fw_p_sol_far,
                    self.params.fw_lambda_q_near,
                    self.params.fw_lambda_q_far,
                )
                qpar_omp.append(q)

        power_entering_omp = []
        for q, dr, fs in zip(qpar_omp, self.flux_surface_width_omp, self.flux_surfaces):
            p = q / (fs.B_omp / fs.Bp_omp) * dr * fs.x_omp
            power_entering_omp.append(p)
        integrated_power_entering_omp = 2 * np.pi * (sum(power_entering_omp))
        self.power_correction_factor_omp = integrated_power_entering_omp / (
            self.params.fw_p_sol_near + self.params.fw_p_sol_far
        )

        return qpar_omp

    def find_intersections(self, profile):
        """
        Find intersections between all the flux
        surfaces and a given first wall profile

        Parameters
        ----------
        profile: Loop
            A first wall 2D profile

        Returns
        -------
        intersections_x : np.array (n intersections)
            x coordinate of intersections
        intersections_z : np.array (n intersections)
            z coordinate of intersections
        """
        intersections_x = []
        intersections_z = []

        for fs in self.flux_surfaces:
            x_int, z_int = fs.find_intersections(fs.loop_lfs, profile)
            intersections_x.append(x_int)
            intersections_z.append(z_int)

        return (
            intersections_x,
            intersections_z,
        )

    def find_first_intersections(
        self,
        intersections_x,
        intersections_z,
    ):
        """
        Find first intersections between all the flux
        surfaces and a given first wall profile

        Parameters
        ----------
        intersections_x : np.array (n intersections)
            x coordinate of intersections
        intersections_z : np.array (n intersections)
            z coordinate of intersections

        Returns
        -------
        lfs_first_intersections : [float, float] (n intersections)
            x, z coordinates of first intersections at lfs
        hfs_first_intersections : [float, float] (n intersections)
            x, z coordinates of first intersections at hfs
        """
        lfs_first_intersections = []
        hfs_first_intersections = []
        self.lfs_flux_line_portion = []
        self.hfs_flux_line_portion = []

        for x, z, fs in zip(
            intersections_x,
            intersections_z,
            self.flux_surfaces,
        ):
            clipped_loop_hfs, clipped_loop_lfs = fs.flux_surface_sub_loop(
                fs.loop_lfs, double_null=False
            )
            lfs_intersections, hfs_intersections = fs.assign_lfs_hfs_sn(x, z)

            first_int_lfs = fs.find_first_intersection(
                clipped_loop_lfs,
                lfs_intersections[0],
                lfs_intersections[1],
                lfs=True,
                double_null=False,
            )
            first_int_hfs = fs.find_first_intersection(
                clipped_loop_hfs,
                hfs_intersections[0],
                hfs_intersections[1],
                lfs=False,
                double_null=False,
            )

            flux_line_lfs = fs.cut_flux_line_portion(
                clipped_loop_lfs, [fs.x_omp, fs.z_omp], first_int_lfs
            )
            flux_line_hfs = fs.cut_flux_line_portion(
                clipped_loop_hfs, [fs.x_omp, fs.z_omp], first_int_hfs
            )
            self.lfs_flux_line_portion.append(flux_line_lfs)
            self.hfs_flux_line_portion.append(flux_line_hfs)

            lfs_first_intersections.append(first_int_lfs)
            hfs_first_intersections.append(first_int_hfs)

        return (
            lfs_first_intersections,
            hfs_first_intersections,
        )

    def calculate_parameters_for_heat_flux(
        self, qpar_omp, fw_profile, lfs_first_intersections, hfs_first_intersections
    ):
        """
        Calculate the parameters for the heat flux calculation
        The parameters are collected by flux surface
        len(flux_surface_list) == len(parameter_list)

        Parameters
        ----------
        qpar_omp: [float]
            Parallel contribution of the power carried by all the fs at the omp
        fw_profile: Loop
            Loop object of the first wall
        lfs_first_intersections : [float, float] (n intersections)
            x, z coordinates of first intersections at lfs
        hfs_first_intersections : [float, float] (n intersections)
            x, z coordinates of first intersections at hfs

        Returns
        -------
        qpar_local_lfs: [float]
            q parallel local for each first intersection at the lfs
        qpar_local_hfs: [float]
            q parallel local for each first intersection at the lfs
        incindent_angle_lfs: [float, float]
            incident angle in deg and rad for each first intersection at the lfs
        incindent_angle_hfs: [float, float]
            incident angle in deg and rad for each first intersection at the hfs
        f_lfs_list: [float]
            flux exapnsion for each fs at the intersection point at the lfs
        f_hfs_list: [float]
            flux exapnsion for each fs at the intersection point at the hfs
        """
        qpar_local_lfs = []
        qpar_local_hfs = []

        incindent_angle_lfs = []
        incindent_angle_hfs = []

        f_lfs_list = []  # target flux expansion at the lfs
        f_hfs_list = []  # target flux expansion at the hfs

        for fs, q, list_int in zip(
            self.flux_surfaces, qpar_omp, lfs_first_intersections
        ):

            # q parallel local at lfs
            q_local_lfs = fs.calculate_q_par_local(
                list_int[0],
                list_int[1],
                q / self.power_correction_factor_omp,
            )
            qpar_local_lfs.append(q_local_lfs)

            angle = fs.calculate_incindent_angle(
                fs.loop_lfs,
                list_int[0],
                list_int[1],
                fw_profile,
            )
            incindent_angle_lfs.append(angle)
            # flux expansion
            f_lfs = fs.f
            f_lfs_list.append(f_lfs)

        for fs, q, list_int in zip(
            self.flux_surfaces, qpar_omp, hfs_first_intersections
        ):

            # q parallel local at lfs
            q_local_hfs = fs.calculate_q_par_local(
                list_int[0],
                list_int[1],
                q / self.power_correction_factor_omp,
            )
            qpar_local_hfs.append(q_local_hfs)

            angle = fs.calculate_incindent_angle(
                fs.loop_lfs,
                list_int[0],
                list_int[1],
                fw_profile,
            )
            incindent_angle_hfs.append(angle)
            # flux expansion
            f_hfs = fs.f
            f_hfs_list.append(f_hfs)

        return (
            qpar_local_lfs,
            qpar_local_hfs,
            incindent_angle_lfs,
            incindent_angle_hfs,
            f_lfs_list,
            f_hfs_list,
        )

    def calculate_heat_flux_lfs_hfs(
        self,
        qpar_local_lfs,
        qpar_local_hfs,
        incindent_angle_lfs,
        incindent_angle_hfs,
    ):
        """
        Heat flux calculation lfs and hfs

        Parameters
        ----------
        qpar_local_lfs: [float]
            q parallel local for each intersection at the lfs
        qpar_local_hfs: [float]
            q parallel local for each intersection at the hfs
        incindent_angle_lfs: [float, float]
            incident angle in deg and rad for each first intersection at the lfs
        incindent_angle_hfs: [float, float]
            incident angle in deg and rad for each first intersection at the hfs

        Returns
        -------
        heat_flux_lfs: [float]
            Heat flux carried by each fs calculated at the intersectio point (lfs)
        heat_flux_hfs: [float]
            Heat flux carried by each fs calculated at the intersectio point (hfs)
        """
        heat_flux_lfs = []
        for q, angle_rad, fs in zip(
            qpar_local_lfs, incindent_angle_lfs, self.flux_surfaces
        ):
            hf = (
                fs.calculate_heat_flux_onto_fw_surface(q, angle_rad[1])
                * self.params.f_outer_target
            )
            heat_flux_lfs.append(hf)

        heat_flux_hfs = []
        for q, angle_rad, fs in zip(
            qpar_local_hfs, incindent_angle_hfs, self.flux_surfaces
        ):
            hf = (
                fs.calculate_heat_flux_onto_fw_surface(q, angle_rad[1])
                * self.params.f_inner_target
            )
            heat_flux_hfs.append(hf)

        return (heat_flux_lfs, heat_flux_hfs)

    def collect_intersection_coordinates_and_heat_flux(
        self,
        lfs_first_int,
        heat_flux_lfs,
        hfs_first_int,
        heat_flux_hfs,
    ):
        """
        Collect all the final parameters under single lists

        Parameters
        ----------
        lfs_first_int: [float, float, float]
            x, z and theta coordinates of all the first intersections at the lfs
        heat_flux_lfs: [float]
            Heat flux values at all the intersection points at the lfs
        hfs_first_int: [float, float, float]
            x, z and theta coordinates of all the first intersections at the hfs
        heat_flux_hfs: [float]
            Heat flux values at all the intersection points at the hfs

        Returns
        -------
        x_int_hf: [float]
            List of all the x coordinates at the inetrsections
        z_int_hf: [float]
            List of all the z coordinates at the intersections
        heat_flux: [float]
            List of all the heat fluxes
        """
        x_int_hf = []
        z_int_hf = []
        heat_flux = []

        for list_xz, hf in zip(lfs_first_int, heat_flux_lfs):
            if list_xz is not None:
                x_int_hf.append(list_xz[0])
                z_int_hf.append(list_xz[1])
                heat_flux.append(hf)
        for list_xz, hf in zip(hfs_first_int, heat_flux_hfs):
            if list_xz is not None:
                x_int_hf.append(list_xz[0])
                z_int_hf.append(list_xz[1])
                heat_flux.append(hf)
        return (x_int_hf, z_int_hf, heat_flux)

    @property
    def xz_plot_loop_names(self):
        """
        The x-z loop names to plot.
        """
        names = ["Inboard wall", "Outboard wall", "Divertor", "Divertor cassette"]
        return names

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
        qpar_omp = self.q_parallel_calculation()

        intersections_x, intersections_z = self.find_intersections(profile)

        lfs_first_ints, hfs_first_ints = self.find_first_intersections(
            intersections_x, intersections_z
        )

        (
            qpar_local_lfs,
            qpar_local_hfs,
            glancing_angle_lfs,
            glancing_angle_hfs,
            f_lfs,
            f_hfs,
        ) = self.calculate_parameters_for_heat_flux(
            qpar_omp, profile, lfs_first_ints, hfs_first_ints
        )

        heat_flux_lfs, heat_flux_hfs = self.calculate_heat_flux_lfs_hfs(
            qpar_local_lfs,
            qpar_local_hfs,
            glancing_angle_lfs,
            glancing_angle_hfs,
        )

        (
            self.x_all_ints,
            self.z_all_ints,
            self.hf_all_ints,
        ) = self.collect_intersection_coordinates_and_heat_flux(
            lfs_first_ints,
            heat_flux_lfs,
            hfs_first_ints,
            heat_flux_hfs,
        )

        x_wall = []
        z_wall = []
        hf_wall = []
        for x, z, hf in zip(self.x_all_ints, self.z_all_ints, self.hf_all_ints):
            if z > self.points["x_point"]["z_low"] + self.x_point_shift:
                x_wall.append(x)
                z_wall.append(z)
                hf_wall.append(hf)
        self.x_wall = x_wall
        self.z_wall = z_wall
        self.hf_wall = hf_wall
        return (x_wall, z_wall, hf_wall)

    def modify_fw_profile(self, profile, x_int_hf, z_int_hf):
        """
        Modify the first wall to reduce the heat flux

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
        ["fw_psi_init", "Initial psi norm value", 1, "N/A", None, "Input"],
        ["fw_dpsi_n_near", "Step size of psi in near SOL", 0.1, "N/A", None, "Input"],
        ["fw_dpsi_n_far", "Step size of psi in far SOL", 0.1, "N/A", None, "Input"],
        ["fw_dx_omp", "Initial offset from LCFS omp", 0.2, "m", None, "Input"],
        ["fw_dx_imp", "Initial offset from LCFS imp", 0.05, "m", None, "Input"],
        ["fw_psi_n", "Normalised psi boundary to fit FW to", 1, "N/A", None, "Input"],
        ["fw_p_sol_near", "near Scrape off layer power", 90, "MW", None, "Input"],
        ["fw_p_sol_far", "far Scrape off layer power", 50, "MW", None, "Input"],
        ["p_rate_omp", "power sharing omp", 0.9, "%", None, "Input"],
        ["p_rate_imp", "power sharing imp", 0.1, "%", None, "Input"],
        ["fw_lambda_q_near_omp", "Lambda_q near SOL omp", 0.003, "m", None, "Input"],
        ["fw_lambda_q_far_omp", "Lambda_q far SOL omp", 0.1, "m", None, "Input"],
        ["fw_lambda_q_near_imp", "Lambda_q near SOL imp", 0.003, "m", None, "Input"],
        ["fw_lambda_q_far_imp", "Lambda_q far SOL imp", 0.1, "m", None, "Input"],
        ["dr_near_omp", "fs thickness near SOL", 0.001, "m", None, "Input"],
        ["dr_far_omp", "fs thickness far SOL", 0.005, "m", None, "Input"],
        ["f_lfs_lower_target", "Power fraction lfs lower", 0.5, "N/A", None, "Input"],
        ["f_lfs_upper_target", "Power fraction lfs upper", 0.5, "N/A", None, "Input"],
        ["f_hfs_lower_target", "Power fraction hfs lower", 0.5, "N/A", None, "Input"],
        ["f_hfs_upper_target", "Power fraction hfs upper", 0.5, "N/A", None, "Input"],
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

    def init_params(self):
        """
        Initialise First Wall DN parameters from config.
        """
        super().init_params()
        self.fw_p_sol_near_omp = self.params.fw_p_sol_near * self.params.p_rate_omp
        self.fw_p_sol_far_omp = self.params.fw_p_sol_far * self.params.p_rate_omp
        self.fw_p_sol_near_imp = self.params.fw_p_sol_near * self.params.p_rate_imp
        self.fw_p_sol_far_imp = self.params.fw_p_sol_far * self.params.p_rate_imp

    def init_equilibrium(self):
        """
        Initialise double null equilibrium inputs.
        """
        self.equilibrium = self.inputs["equilibrium"]
        EqInputs.__init__(self, x_point_shift=self.params.xpt_height)

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
        dx_loop_lfs = self.lcfs.offset(self.params.fw_dx_omp)
        clip_lfs = np.where(
            dx_loop_lfs.x > self.points["x_point"]["x"],
        )
        new_loop_lfs = Loop(
            dx_loop_lfs.x[clip_lfs],
            z=dx_loop_lfs.z[clip_lfs],
        )

        dx_loop_hfs = self.lcfs.offset(self.params.fw_dx_imp)
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

    def make_flux_surfaces(self, profile=None):
        """
        Generate a set of flux surfaces placed between lcfs and fw

        """
        self.flux_surfaces = []
        x_omp = self.x_omp_lcfs + self.params.dr_near_omp
        if profile is None:
            profile = self.profile

        # Find intersections between the profile and mid-plane
        profile_ints = loop_plane_intersect(profile, self.mid_plane)

        # Retrieve x of outer / innter  intersection
        profile_x_omp = find_outer_point(profile_ints, self.points["o_point"]["x"])[0]
        profile_x_imp = find_inner_point(profile_ints, self.points["o_point"]["x"])[0]

        while x_omp < profile_x_omp:
            flux_surface = FluxSurface(
                self.equilibrium, x_omp, self.points["o_point"]["z"]
            )
            x_omp = flux_surface.x_omp
            self.flux_surfaces.append(flux_surface)
            if (
                flux_surface.x_omp - flux_surface.x_omp_lcfs
            ) < self.params.fw_lambda_q_near_omp:
                x_omp += self.params.dr_near_omp
            else:
                x_omp += self.params.dr_far_omp

        self.flux_surface_hfs = []
        self.flux_surface_lfs = []

        for fs in self.flux_surfaces:
            if (
                hasattr(fs, "loop_hfs")
                and loop_plane_intersect(fs.loop_hfs, self.mid_plane)[0][0]
                > profile_x_imp
            ):
                self.flux_surface_hfs.append(fs.loop_hfs)

        for fs in self.flux_surfaces:
            if hasattr(fs, "loop_lfs") and fs.find_intersections(fs.loop_lfs, profile):
                self.flux_surface_lfs.append(fs.loop_lfs)

        self.flux_surface_width_omp = []
        dr_0_omp = self.flux_surfaces[0].x_omp - self.x_omp_lcfs
        self.flux_surface_width_omp.append(dr_0_omp)

        for i in range(len(self.flux_surfaces)):
            dr_omp = (
                self.flux_surfaces[(i + 1) % len(self.flux_surfaces)].dr_omp
                - self.flux_surfaces[i].dr_omp
            )
            self.flux_surface_width_omp.append(dr_omp)

        if self.flux_surface_width_omp[-1] < 0:
            self.flux_surface_width_omp = self.flux_surface_width_omp[:-1]

        if len(self.flux_surface_hfs) != 0:
            self.flux_surface_width_imp = []
            dr_0_imp = -(self.flux_surfaces[0].x_imp - self.x_imp_lcfs)
            self.flux_surface_width_imp.append(dr_0_imp)

            for i, fs in zip(range(len(self.flux_surfaces)), self.flux_surfaces):
                if hasattr(self.flux_surfaces[i], "dr_imp") and hasattr(
                    self.flux_surfaces[(i + 1) % len(self.flux_surfaces)], "dr_imp"
                ):

                    dr_imp = (
                        self.flux_surfaces[(i + 1) % len(self.flux_surfaces)].dr_imp
                        - self.flux_surfaces[i].dr_imp
                    )

                    self.flux_surface_width_imp.append(dr_imp)
            if self.flux_surface_width_imp[-1] < 0:
                self.flux_surface_width_imp = self.flux_surface_width_imp[:-1]

    def q_parallel_calculation(self):
        """
        Calculate q parallel at OMP and IMP for all the flux surfaces

        Returns
        -------
        qpar_omp: [float]
            Parallel contribution of the power carried by all the fs at the omp
        qpar_imp: [float]
            Parallel contribution of the power carried by all the fs at the imp
        """
        qpar_omp = []
        qpar_imp = []

        for fs in self.flux_surfaces:
            if hasattr(fs, "dr_omp"):
                q = fs.calculate_q_par_omp(
                    self.fw_p_sol_near_omp,
                    self.fw_p_sol_far_omp,
                    self.params.fw_lambda_q_near_omp,
                    self.params.fw_lambda_q_far_omp,
                )
                qpar_omp.append(q)

        power_entering_omp = []
        for q, dr, fs in zip(qpar_omp, self.flux_surface_width_omp, self.flux_surfaces):
            p = q / (fs.B_omp / fs.Bp_omp) * dr * fs.x_omp
            power_entering_omp.append(p)
        integrated_power_entering_omp = 2 * np.pi * (sum(power_entering_omp))
        self.power_correction_factor_omp = integrated_power_entering_omp / (
            self.fw_p_sol_near_omp + self.fw_p_sol_far_omp
        )

        for fs in self.flux_surfaces:
            if hasattr(fs, "dr_imp"):
                q = fs.calculate_q_par_imp(
                    self.fw_p_sol_near_imp,
                    self.fw_p_sol_far_imp,
                    self.params.fw_lambda_q_near_imp,
                    self.params.fw_lambda_q_far_imp,
                )
                qpar_imp.append(q)

        power_entering_imp = []
        for q, dr, fs in zip(qpar_imp, self.flux_surface_width_imp, self.flux_surfaces):
            power_entering_imp.append(q / (fs.B_imp / fs.Bp_imp) * dr * fs.x_imp)
        integrated_power_entering_imp = 2 * np.pi * (sum(power_entering_imp))
        self.power_correction_factor_imp = integrated_power_entering_imp / (
            self.fw_p_sol_near_imp + self.fw_p_sol_far_imp
        )
        return (qpar_omp, qpar_imp)

    def find_intersections(self, profile):
        """
        Find the intersections between all the flux surfaces and the first wall

        Parameters
        ----------
        profile: Loop
            A first wall 2D profile

        Returns
        -------
        lfs_intersections_x: [[float]]
            x coordinate of all the intersections of all the fs at lfs
        lfs_intersections_z: [[float]]
            z coordinate of all the intersections of all the fs at lfs
        hfs_intersections_x: [[float]]
            x coordinate of all the intersections of all the fs at hfs
        hfs_intersections_z: [[float]]
            z coordinate of all the intersections of all the fs at hfs
        """
        lfs_intersections_x = []
        lfs_intersections_z = []
        hfs_intersections_x = []
        hfs_intersections_z = []

        for loop, fs in zip(self.flux_surface_hfs, self.flux_surfaces):
            x_int, z_int = fs.find_intersections(loop, profile)
            hfs_intersections_x.append(x_int)
            hfs_intersections_z.append(z_int)

        for loop, fs in zip(self.flux_surface_lfs, self.flux_surfaces):
            x_int, z_int = fs.find_intersections(loop, profile)
            lfs_intersections_x.append(x_int)
            lfs_intersections_z.append(z_int)

        return (
            lfs_intersections_x,
            lfs_intersections_z,
            hfs_intersections_x,
            hfs_intersections_z,
        )

    def find_first_intersections(
        self,
        lfs_intersections_x,
        lfs_intersections_z,
        hfs_intersections_x,
        hfs_intersections_z,
    ):
        """
        Find first intersections between all the flux
        surfaces and a given first wall profile

        Parameters
        ----------
        lfs_intersections_x: np.array (n intersections)
            x coordinate of intersections
        lfs_intersections_z: np.array (n intersections)
            z coordinate of intersections
        hfs_intersections_x: np.array (n intersections)
            x coordinate of intersections
        hfs_intersections_z: np.array (n intersections)
            z coordinate of intersections

        Returns
        -------
        lfs_down_first_intersections: [float, float] (n intersections)
            x, z coordinates of first intersections at lfs
        lfs_up_first_intersections: [float, float] (n intersections)
            x, z coordinates of first intersections at lfs
        hfs_down_first_intersections: [float, float] (n intersections)
            x, z coordinates of first intersections at hfs
        hfs_up_first_intersections: [float, float] (n intersections)
            x, z coordinates of first intersections at hfs
        """
        lfs_down_first_intersections = []
        lfs_up_first_intersections = []
        hfs_down_first_intersections = []
        hfs_up_first_intersections = []
        self.lfs_flux_line_portion = []
        self.hfs_flux_line_portion = []

        for x, z, loop, fs in zip(
            lfs_intersections_x,
            lfs_intersections_z,
            self.flux_surface_lfs,
            self.flux_surfaces,
        ):
            clipped_loop_up, clipped_loop_down = fs.flux_surface_sub_loop(loop)
            lfs_top_intersections, lfs_bottom_intersections = fs.assign_top_bottom(x, z)

            first_int_down = fs.find_first_intersection(
                clipped_loop_down,
                lfs_bottom_intersections[0],
                lfs_bottom_intersections[1],
                lfs=True,
            )
            first_int_up = fs.find_first_intersection(
                clipped_loop_up,
                lfs_top_intersections[0],
                lfs_top_intersections[1],
                lfs=True,
            )

            flux_line_lfs_down = fs.cut_flux_line_portion(
                clipped_loop_down, [fs.x_omp, fs.z_omp], first_int_down
            )
            flux_line_lfs_up = fs.cut_flux_line_portion(
                clipped_loop_up, [fs.x_omp, fs.z_omp], first_int_up
            )
            self.lfs_flux_line_portion.append(flux_line_lfs_down)
            self.lfs_flux_line_portion.append(flux_line_lfs_up)

            lfs_down_first_intersections.append(first_int_down)
            lfs_up_first_intersections.append(first_int_up)

        for x, z, loop, fs in zip(
            hfs_intersections_x,
            hfs_intersections_z,
            self.flux_surface_hfs,
            self.flux_surfaces,
        ):
            clipped_loop_up, clipped_loop_down = fs.flux_surface_sub_loop(loop)
            hfs_top_intersections, hfs_bottom_intersections = fs.assign_top_bottom(x, z)

            first_int_down = fs.find_first_intersection(
                clipped_loop_down,
                hfs_bottom_intersections[0],
                hfs_bottom_intersections[1],
                lfs=False,
            )
            first_int_up = fs.find_first_intersection(
                clipped_loop_up,
                hfs_top_intersections[0],
                hfs_top_intersections[1],
                lfs=False,
            )

            flux_line_hfs_down = fs.cut_flux_line_portion(
                clipped_loop_down, [fs.x_imp, fs.z_imp], first_int_down
            )
            flux_line_hfs_up = fs.cut_flux_line_portion(
                clipped_loop_up, [fs.x_imp, fs.z_imp], first_int_up
            )
            self.hfs_flux_line_portion.append(flux_line_hfs_down)
            self.hfs_flux_line_portion.append(flux_line_hfs_up)

            hfs_down_first_intersections.append(first_int_down)
            hfs_up_first_intersections.append(first_int_up)

        return (
            lfs_down_first_intersections,
            lfs_up_first_intersections,
            hfs_down_first_intersections,
            hfs_up_first_intersections,
        )

    def calculate_parameters_for_heat_flux(
        self, qpar_midplane, list_first_intersections, list_flux_surfaces, profile
    ):
        """
        Calculate the key parameters to calculate the heat flux

        The parameters are collected by flux surface
        len(flux_surface_list) == len(parameter_list)

        Parameters
        ----------
        qpar_midplane: [float]
            Parallel contribution of the power carried by all the fs at the midplane
        list_first_intersections: [float, float]
            x and z coordinate of first intersections
        list_flux_surfaces: [Loop]
            List of flux surfaces
        profile: Loop
            First wall profile

        Returns
        -------
        list_qpar_target: [float]
            q parallel at the intersection point
        list_incident_angle: [float]
            incident angle between fs and first wall
        list_flux_expansion: [float]
            flux expansion at the intersection between fs and first wall
        """
        list_qpar_target = []
        list_incident_angle = []
        list_flux_expansion = []

        for first_intersection, qpar, loop, fs in zip(
            list_first_intersections,
            qpar_midplane,
            list_flux_surfaces,
            self.flux_surfaces,
        ):

            qpar_target = partial(
                fs.calculate_q_par_local,
                first_intersection[0],
                first_intersection[1],
                qpar / self.power_correction_factor_omp,
            )
            angle = partial(
                fs.calculate_incindent_angle,
                loop,
                first_intersection[0],
                first_intersection[1],
                profile,
            )

            functions = [qpar_target, angle, lambda: fs.f]
            lists = [list_qpar_target, list_incident_angle, list_flux_expansion]
            for function, to_append in zip(functions, lists):
                test = np.where(
                    len(first_intersection) != 0,
                    function(),
                    np.nan,
                )
                to_append.append(test)

        return (list_qpar_target, list_incident_angle, list_flux_expansion)

    def define_flux_surfaces_parameters_to_calculate_heat_flux(
        self,
        profile,
        qpar_omp,
        qpar_imp,
        lfs_down_first_intersections,
        lfs_up_first_intersections,
        hfs_down_first_intersections,
        hfs_up_first_intersections,
    ):
        """
        Collect by regions the key parameters to calculate the heat flux
        Regions are in the order: lfs bottom, lfs top, hfs bottom, hfs top
        Specific method for double null

        Parameters
        ----------
        qpar_omp: [float]
            Parallel contribution of the power carried by all the fs at the omp
        qpar_imp: [float]
            Parallel contribution of the power carried by all the fs at the imp
        lfs_down_first_intersections: [float, float]
            x, z coordinates of first intersections at lfs bottom
        lfs_up_first_intersections: [float, float]
            x, z coordinates of first intersections at lfs top
        hfs_down_first_intersections: [float, float]
            x, z coordinates of first intersections at hfs bottom
        hfs_up_first_intersections: [float, float]
            x, z coordinates of first intersections at hfs top

        Returns
        -------
        qpar_local_lfs[0]: [float]
            Local q parallel for each fs at the intersection point at lfs bottom
        qpar_local_lfs[1]: [float]
            Local q parallel for each fs at the intersection point at lfs top
        qpar_local_hfs[0]: [float]
            Local q parallel for each fs at the intersection point at hfs bottom
        qpar_local_hfs[1]: [float]
            Local q parallel for each fs at the intersection point at hfs top
        incindent_angle_lfs[0]: [float]
            Incindent angle (deg) at the intersection between fs and fw at lfs bottom
        incindent_angle_lfs[1]: [float]
            Incindent angle (deg) at the intersection between fs and fw at lfs top
        incindent_angle_hfs[0]: [float]
            Incindent angle (deg) at the intersection between fs and fw at hfs bottom
        incindent_angle_hfs[1]: [float]
            Incindent angle (deg) at the intersection between fs and fw at hfs top
        f_lfs_list[0]: [float]
            Flux expansion at the intersections at lfs bottom
        f_lfs_list[1]: [float]
            Flux expansion at the intersections at lfs top
        f_hfs_list[0]: [float]
            Flux expansion at the intersections at hfs bottom
        f_hfs_list[1]: [float]
            Flux expansion at the intersections at hfs top
        """
        qpar_local_lfs = []
        incindent_angle_lfs = []
        f_lfs_list = []
        qpar_local_hfs = []
        incindent_angle_hfs = []
        f_hfs_list = []

        for intersection in [lfs_down_first_intersections, lfs_up_first_intersections]:
            (q, angle, f) = self.calculate_parameters_for_heat_flux(
                qpar_omp,
                intersection,
                self.flux_surface_lfs,
                profile,
            )
            qpar_local_lfs.append(q)
            incindent_angle_lfs.append(angle)
            f_lfs_list.append(f)

        for intersection in [hfs_down_first_intersections, hfs_up_first_intersections]:
            (q, angle, f) = self.calculate_parameters_for_heat_flux(
                qpar_imp,
                intersection,
                self.flux_surface_hfs,
                profile,
            )
            qpar_local_hfs.append(q)
            incindent_angle_hfs.append(angle)
            f_hfs_list.append(f)

        return (
            [qpar_local_lfs, qpar_local_hfs],
            [incindent_angle_lfs, incindent_angle_hfs],
            [f_lfs_list, f_hfs_list],
        )

    def calculate_heat_fluxes(self, qpar_local, incident_angles, power_sharing_factor):
        """
        Calculate the heat flux for all the intersection points

        Parameters
        ----------
        qpar_local: [float]
            q parallel local for each intersection
        incindent_angle: [float]
            incident angle in deg for each first intersection at the lfs
        power_sharing_factor: float
            This rate depends on the region. The different values are given
            in the parameter set at the beginning of the class

        Returns
        -------
        heat_flux: [float]
            Heat flux carried by each fs calculated at the intersection point
        """
        heat_fluxes = []

        for q, angle_rad, fs in zip(qpar_local, incident_angles, self.flux_surfaces):
            hf = np.where(
                len(angle_rad) != 0,
                fs.calculate_heat_flux_onto_fw_surface(q, angle_rad[1])
                * power_sharing_factor,
                np.nan,
            )
            heat_fluxes.append(hf)

        return heat_fluxes

    def collect_intersection_coordinates_and_heat_flux(
        self, list_first_intersections, list_heat_flux
    ):
        """
        Collect intersection coordinates and corresponding hf values

        Parameters
        ----------
        list_first_intersections: List [float, float]
            x, z coordinates of all the first intersections
        list_heat_flux: [float]
            Heat flux values at the intersections

        Returns
        -------
        x_int_hf: [float]
            List of all the x coordinates at the intersections
        z_int_hf: [float]
            List of all the z coordinates at the intersections
        heat_flux: [float]
            List of all the heat fluxes
        """
        x_int_hf = []
        z_int_hf = []
        heat_flux = []
        heat_map = [x_int_hf, z_int_hf, heat_flux]

        for list_xz, hf in zip(list_first_intersections, list_heat_flux):
            if len(list_xz) != 0:
                attrs = [list_xz[0], list_xz[1], hf]
                for no, list_params in enumerate(heat_map):
                    list_params.append(attrs[no])
            else:
                for list_params in heat_map:
                    list_params.append(np.nan)

        return (x_int_hf, z_int_hf, heat_flux)

    def calculate_heat_flux(
        self,
        lfs_down_int,
        lfs_up_int,
        hfs_down_int,
        hfs_up_int,
        qpar_local_lfs_down,
        qpar_local_lfs_up,
        qpar_local_hfs_down,
        qpar_local_hfs_up,
        incindent_angle_lfs_down,
        incindent_angle_lfs_up,
        incindent_angle_hfs_down,
        incindent_angle_hfs_up,
    ):
        """
        Specific method for double null to collect all the heat flux values
        To calculate the heat flux for the four regions

        Collect al the final parameters under single lists

        Parameters
        ----------
        lfs_down_int: [float, float]
            x, z coordinates of first intersections at lfs bottom
        lfs_up_int: [float, float]
            x, z coordinates of first intersections at lfs top
        hfs_down_int: [float, float]
            x, z coordinates of first intersections at hfs bottom
        hfs_up_int: [float, float]
            x, z coordinates of first intersections at hfs top
        qpar_local_lfs_down: [float]
            q parallel local at lfs bottom first intersection
        qpar_local_lfs_up: [float]
            q parallel local at lfs top first intersection
        qpar_local_hfs_down: [float]
            q parallel local at hfs bottom first intersection
        qpar_local_hfs_up: [float]
            q parallel local at hfs top first intersection
        incindent_angle_lfs_down: [float]
            incident angle (deg) at lfs bottom first intersections
        incindent_angle_lfs_up: [float]
            incident angle (deg) at lfs top first intersections
        incindent_angle_hfs_down: [float]
            incident angle (deg) at hfs bottom first intersections
        incindent_angle_hfs_up: [float]
            incident angle (deg) at hfs top first intersections

        Returns
        -------
        x_int_hf: [float]
            List of all the x coordinates at the intersections
        z_int_hf: [float]
            List of all the z coordinates at the intersections
        th_int_hf: [float]
            List of all the theta coordinates at the intersections
        heat_flux: [float]
            List of all the heat fluxes
        """
        heat_fluxes = [
            self.calculate_heat_fluxes(  # heat_flux_lfs_down
                qpar_local_lfs_down,
                incindent_angle_lfs_down,
                self.params.f_lfs_lower_target,
            ),
            self.calculate_heat_fluxes(  # heat_flux_lfs_up
                qpar_local_lfs_up, incindent_angle_lfs_up, self.params.f_lfs_upper_target
            ),
            self.calculate_heat_fluxes(  # heat_flux_hfs_down
                qpar_local_hfs_down,
                incindent_angle_hfs_down,
                self.params.f_hfs_lower_target,
            ),
            self.calculate_heat_fluxes(  # heat_flux_hfs_up
                qpar_local_hfs_up, incindent_angle_hfs_up, self.params.f_hfs_upper_target
            ),
        ]
        # Collecting intersection point coordinates and heat fluxes
        x_int_hf = []
        z_int_hf = []
        heat_flux = []
        for heat_f, first_int in zip(
            heat_fluxes, [lfs_down_int, lfs_up_int, hfs_down_int, hfs_up_int]
        ):
            x, z, hf = self.collect_intersection_coordinates_and_heat_flux(
                first_int, heat_f
            )
            x_int_hf.append(x)
            z_int_hf.append(z)
            heat_flux.append(hf)

        return (x_int_hf, z_int_hf, heat_flux)

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

    def hf_firstwall_params(self, profile):
        """
        Define params to plot the heat flux on the fw (no divertor).
        """
        qpar_omp, qpar_imp = self.q_parallel_calculation()
        lfs_hfs_intersections = self.find_intersections(profile)
        first_intersections = self.find_first_intersections(*lfs_hfs_intersections)
        (
            qpar_local_lfs_hfs,
            incindent_angle_lfs_hfs,
            f_list_lfs_hfs,
        ) = self.define_flux_surfaces_parameters_to_calculate_heat_flux(
            profile,
            qpar_omp,
            qpar_imp,
            *first_intersections,
        )

        x_coord_ints, z_coord_ints, hf_ints = self.calculate_heat_flux(
            *first_intersections,
            *qpar_local_lfs_hfs[0],
            *qpar_local_lfs_hfs[1],
            *incindent_angle_lfs_hfs[0],
            *incindent_angle_lfs_hfs[1],
        )

        self.x_all_ints = list(np.concatenate(x_coord_ints).flat)
        self.z_all_ints = list(np.concatenate(z_coord_ints).flat)
        self.hf_all_ints = list(np.concatenate(hf_ints).flat)

        x_wall = []
        z_wall = []
        hf_wall = []
        for x, z, hf in zip(self.x_all_ints, self.z_all_ints, self.hf_all_ints):
            if (
                z < self.points["x_point"]["z_up"]
                and z > self.points["x_point"]["z_low"]
            ):
                x_wall.append(x)
                z_wall.append(z)
                hf_wall.append(hf)
        self.x_wall = x_wall
        self.z_wall = z_wall
        self.hf_wall = hf_wall
        return (x_wall, z_wall, hf_wall)

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
                    self.x_imp_lcfs - self.params.fw_dx_imp
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
        loops.plot(ax=ax, fill=False, edgecolor="r", linewidth=0.2)
        separatrix.plot(ax=ax, fill=False, edgecolor="r", linewidth=1)
        if koz is not None:
            koz.plot(ax=ax, fill=False, edgecolor="g", linewidth=1)
        ax = plt.gca()
        cs = ax.scatter(x_int, z_int, s=25, c=hf_int, cmap="viridis", zorder=100)
        bar = plt.gcf().colorbar(cs, ax=ax)
        bar.set_label("Heat Flux [MW/m^2]")
