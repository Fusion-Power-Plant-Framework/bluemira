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
from typing import Type
from BLUEPRINT.base import ReactorSystem, ParameterFrame
from BLUEPRINT.cad.firstwallCAD import FirstWallCAD
from BLUEPRINT.geometry.loop import Loop, MultiLoop
from BLUEPRINT.geometry.shell import Shell
from BLUEPRINT.geometry.boolean import (
    convex_hull,
    boolean_2d_union,
    simplify_loop,
)
from BLUEPRINT.geometry.geomtools import (
    get_intersect,
    check_linesegment,
    loop_plane_intersect,
)
from BLUEPRINT.geometry.geombase import Plane
from BLUEPRINT.geometry.geomtools import lineq
from BLUEPRINT.geometry.geomtools import rotate_vector_2d
from functools import partial


class EqInputs:
    """
    Load a equilibrium file
    """

    def load_equilibrium(self, lcfs_shift=0.001, x_point_shift=0.5):
        """
        Extract basic equilibrium information

        Parameters
        ----------
        lcfs_shift: Sometime the separatrix is not well defined.
                    This parameter take an open flux surface, close to the lcfs
                    to replace the separatrix
        x_point_shift: A shift to slightly move away from the x-point
                    and avoid singularities
        """
        self.lcfs = self.equilibrium.get_LCFS()
        o_point, x_point = self.equilibrium.get_OX_points()
        self.points = {
            "x_point": {
                "x": x_point[0][0],
                "z_low": x_point[0][1],
                "z_up": x_point[1][1],
            },
            "o_point": {"x": o_point[0][0], "z": o_point[0][1]},
        }
        if self.points["x_point"]["z_low"] > self.points["x_point"]["z_up"]:
            self.points["x_point"]["z_low"] = x_point[1][1]
            self.points["x_point"]["z_up"] = x_point[0][1]
        self.mid_plane = Plane(
            [self.points["o_point"]["x"], 0, self.points["o_point"]["z"]],
            [0, 1, self.points["o_point"]["z"]],
            [0, 0, self.points["o_point"]["z"]],
        )
        mp_ints = loop_plane_intersect(self.lcfs, self.mid_plane)
        self.x_omp_lcfs = next(
            filter(
                lambda p: p[0] > self.points["o_point"]["x"],
                mp_ints,
            )
        )[0]
        self.x_imp_lcfs = next(
            filter(
                lambda p: p[0] < self.points["o_point"]["x"],
                mp_ints,
            )
        )[0]

        self.lcfs_shift = lcfs_shift
        self.x_point_shift = x_point_shift

        if round(self.points["x_point"]["z_low"], 3) == -round(
            self.points["x_point"]["z_up"], 3
        ):
            self.sep = self.equilibrium.get_separatrix()
            x_point_limit = self.points["x_point"]["z_low"] - self.x_point_shift
            sep_limit = min(self.sep[0].z)

            if sep_limit < x_point_limit:
                self.separatrix = self.sep
            if sep_limit > x_point_limit:
                loops = self.equilibrium.get_flux_surface_through_point(
                    self.x_omp_lcfs + self.lcfs_shift, 0
                )
                loops.reverse()
                self.separatrix = MultiLoop(loops[:2])

        elif round(self.points["x_point"]["z_low"], 3) != -round(
            self.points["x_point"]["z_up"], 3
        ):
            self.separatrix = self.equilibrium.get_separatrix()


class FluxSurface(EqInputs):
    """
    Create a flux surface
    Evaluate all needed attributes which are useful for to the heat flux calculation

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
        """
        Parameters
        ----------
        equilibrium: eqdsk, geqdsk or json
        x: float
            x coordinate of the midplane point from which the fs has to pass
        z: float
            z coordinate of the midplane point from which the fs has to pass
        """
        self.equilibrium = equilibrium
        super().load_equilibrium()

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

    def polar_coordinates(self, x_int, z_int):
        """
        Calculate the polar coordinate theta for a given set of intersection points

        Parameters
        ----------
        x_int : np.array (n intersections)
            x coordinate of intersections
        z_int : np.array (n intersections)
            z coordinate of intersections

        Returns
        -------
        theta_coord : np.array (n intersections)
            The theta coordinates corresponding to the intersections
            measured from the outer mid-plane proceeding anti-clockwise
        """
        theta_coord = []
        for x, z in zip(x_int, z_int):
            theta_abs = np.arctan(
                (z - self.points["o_point"]["z"]) / (x - self.points["x_point"]["x"])
            )
            if (x - self.points["x_point"]["x"]) > 0 and (
                z - self.points["o_point"]["z"]
            ) > 0:
                theta_coord.append((theta_abs) * (180 / np.pi))
            elif (x - self.points["x_point"]["x"]) > 0 and (
                z - self.points["o_point"]["z"]
            ) < 0:
                theta_coord.append(((theta_abs) * (180 / np.pi)) + 360)
            elif (x - self.points["x_point"]["x"]) < 0:
                theta_coord.append(((theta_abs) * (180 / np.pi)) + 180)
        return np.array(theta_coord)

    def assign_lfs_hfs_sn(self, x_int, z_int, theta_coord):
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

        int_points_lfs = np.zeros((3, lfs_ind[0].size))
        int_points_hfs = np.zeros((3, hfs_ind[0].size))

        for no, i in enumerate(lfs_ind[0]):
            int_points_lfs[0, no] = x_int[i]
            int_points_lfs[1, no] = z_int[i]
            int_points_lfs[2, no] = theta_coord[i]

        for no, i in enumerate(hfs_ind[0]):
            int_points_hfs[0, no] = x_int[i]
            int_points_hfs[1, no] = z_int[i]
            int_points_hfs[2, no] = theta_coord[i]

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

    def find_first_intersection_lfs_sn(self, x_int_lfs, z_int_lfs, theta_coord_lfs):
        """
        Find the first intersection point at the low field side for one flux surface
        Applicable to the SN configuration

        Parameters
        ----------
        x_int_lfs : [float]
            x coordinate of intersections located at the lfs
        z_int_lfs : [float]
            z coordinate of intersections located at the lfs
        theta_coord_lfs: [float]
            theta coordinate of intersections located at the lfs

        Returns
        -------
        lfs_first_int_x : float
            x coordinate of first intersection located at the lfs
        lfs_first_int_z : float
            z coordinate of first intersection located at the lfs
        lfs_first_int_theta: float
            theta coordinate of first intersection located at the lfs
        """
        first_int_ind = np.where(theta_coord_lfs == max(theta_coord_lfs))
        return (
            x_int_lfs[first_int_ind][0],
            z_int_lfs[first_int_ind][0],
            theta_coord_lfs[first_int_ind][0],
        )

    def find_first_intersection_hfs_sn(self, x_int_hfs, z_int_hfs, theta_coord_hfs):
        """
        Find the first intersection point at the high field side for one flux surface
        Applicable to the SN configuration

        Parameters
        ----------
        x_int_hfs : [float]
            x coordinate of intersections located at the hfs
        z_int_hfs : [float]
            z coordinate of intersections located at the hfs
        theta_coord_hfs: [float]
            theta coordinate of intersections located at the hfs

        Returns
        -------
        hfs_first_int_x : float
            x coordinate of first intersection located at the hfs
        hfs_first_int_z : float
            z coordinate of first intersection located at the hfs
        hfs_first_int_theta: float
            theta coordinate of first intersection located at the hfs
        """
        first_int_ind = np.where(theta_coord_hfs == min(theta_coord_hfs))
        return (
            x_int_hfs[first_int_ind][0],
            z_int_hfs[first_int_ind][0],
            theta_coord_hfs[first_int_ind][0],
        )

    def snip_flux_surface(
        self,
        loop,
        intersection_points_x,
        intersection_points_z,
    ):
        """
        Shorten the flux surface, removing the part after the intersection

        Parameters
        ----------
        loop : Loop
            flux surface loop
        intersection_points_x : [float]
            x coordinates of all the intersection points of a flux surface
        intersection_points_z: [float]
            z coordinates of all the intersection points of a flux surface

        Returns
        -------
        clips_up : [Loop]
        clips_down : [Loop]
            Each clip is the piece of the flux surface between the midplane
            and a certain point
        """
        clips_up = []
        clips_down = []

        if loop.point_in_poly([self.x_omp, self.z_omp], True) or loop.point_in_poly(
            [self.x_imp, self.z_imp], True
        ):
            clip_up = np.where(loop.z > self.z_omp)
            clipped_loop_up = Loop(loop.x[clip_up], z=loop.z[clip_up])
            clip_down = np.where(loop.z < self.z_omp)
            clipped_loop_down = Loop(loop.x[clip_down], z=loop.z[clip_down])

            z_up_ind = np.where(intersection_points_z > self.z_omp)[0]
            for ind_z in z_up_ind:
                clip = np.where(clipped_loop_up.z < intersection_points_z[ind_z])
                clipped_up = Loop(clipped_loop_up.x[clip], z=clipped_loop_up.z[clip])
                clips_up.append(clipped_up)

            z_down_ind = np.where(intersection_points_z < self.z_omp)[0]
            for ind_z in z_down_ind:
                clip = np.where(clipped_loop_down.z > intersection_points_z[ind_z])
                clipped_down = Loop(
                    clipped_loop_down.x[clip], z=clipped_loop_down.z[clip]
                )
                clips_down.append(clipped_down)

        return (clips_up, clips_down)

    def find_first_intersection_dn(self, list_intersections, clipped_loops):
        """
        Find the first intersection between a flux surface and the first wall
        Applicable to the DN configuration

        Parameters
        ----------
        list_intersections : [float, float]
            The [x, z] points corresponding to the intersections
        clipped_loops : [Loop]
            All the sub-pieces of a flux surface

        Returns
        -------
        first_int_x : float
            x coordinate of first intersection
        first_int_z : float
            z coordinate of first intersection
        flux_surface_snip : Loop
            Portion of flux surface from the midplane to the intersection point
        linear_coordinate : [float]
            Distance between the starting point at the midplane and each point of the
            loop until the intersection point
        poloidal_length : float
            Distance between the staring point at the midplane and the intersection point
        """
        linear_coordinates = []
        poloidal_lengths = []

        for x, z, clip in zip(
            list_intersections[0],
            list_intersections[1],
            clipped_loops,
        ):
            s = clip.distance_to([x, z])
            linear_coordinates.append(s)
            poloidal_lengths.append(max(s))

        ind = np.where(poloidal_lengths == min(poloidal_lengths))[0][0]
        first_int_x = list_intersections[0][ind]
        first_int_z = list_intersections[1][ind]
        flux_surface_snip = clipped_loops[ind]
        linear_coordinate = linear_coordinates[ind]
        poloidal_length = poloidal_lengths[ind]

        return (
            first_int_x,
            first_int_z,
            flux_surface_snip,
            linear_coordinate,
            poloidal_length,
        )

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


class FirstWallSN(EqInputs, ReactorSystem):
    """
    Reactor First Wall (FW) system

    First Wall design for a SN configuration
    The user needs to change the default parameters according to the case
    """

    config: Type[ParameterFrame]
    inputs: dict

    default_params = [
        ["n_TF", "Number of TF coils", 16, "N/A", None, "Input"],
        ["plasma_type", "Type of plasma", "SN", "N/A", None, "Input"],
        ["A", "Plasma aspect ratio", 3.1, "N/A", None, "Input"],
        ["fw_dx", "Initial offset from LCFS", 0.3, "m", None, "Input"],
        ["fw_psi_n_prel", "psi preliminary profile", 1.01, "N/A", None, "Input"],
        ["fw_p_sol_near", "near Scrape off layer power", 50, "MW", None, "Input"],
        ["fw_p_sol_far", "far Scrape off layer power", 50, "MW", None, "Input"],
        ["fw_lambda_q_near", "Lambda q near SOL", 0.05, "m", None, "Input"],
        ["fw_lambda_q_far", "Lambda q far SOL", 0.05, "m", None, "Input"],
        ["f_outer_target", "Power fraction", 0.75, "N/A", None, "Input"],
        ["f_inner_target", "Power fraction", 0.25, "N/A", None, "Input"],
        ["tk_bb_fw", "First wall thickness", 0.052, "m", None, "Input"],
        # To draw the divertor
        ["xpt_outer_gap", "Gap between x-point and outer wall", 2, "m", None, "Input"],
        ["xpt_inner_gap", "Gap between x-point and inner wall", 1, "m", None, "Input"],
        ["outer_strike_h", "Outer strike point height", 2, "m", None, "Input"],
        ["inner_strike_h", "Inner strike point height", 1, "m", None, "Input"],
        ["outer_target_SOL", "Outer target length SOL side", 0.7, "m", None, "Input"],
        ["outer_target_PFR", "Outer target length PFR side", 0.3, "m", None, "Input"],
        ["inner_target_SOL", "Inner target length SOL side", 0.3, "m", None, "Input"],
        ["inner_target_PFR", "Inner target length PFR side", 0.5, "m", None, "Input"],
    ]
    CADConstructor = FirstWallCAD

    def __init__(self, config, inputs):
        self.config = config
        self.inputs = inputs

        self.params = ParameterFrame(self.default_params.to_records())
        self.params.update_kw_parameters(self.config)

        self.equilibrium = inputs["equilibrium"]
        super().load_equilibrium()

        if "profile" in inputs:
            self.profile = inputs["profile"]
        else:
            self.profile = self.make_preliminary_profile()

        self.make_flux_surfaces()

        self.inner_profile = self.make_divertor(self.profile)
        outer_profile = self.inner_profile[0].offset(self.params.tk_bb_fw)
        outer_profile = simplify_loop(outer_profile)

        self.geom["2D profile"] = Shell(inner=self.inner_profile[0], outer=outer_profile)

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
            self.params.fw_psi_n_prel,
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

        return fw_loop

    def make_divertor(self, fw_loop):
        """
        Make a divertor

        Parameters
        ----------
        fw_loop: Loop
            first wall profile

        Returns
        -------
        fw_diverted_loop: Loop
            Here the first wall also has a divertor geometry
        """
        mid_plane_inner_leg_cut = Plane(
            [
                self.points["x_point"]["x"],
                0,
                self.points["x_point"]["z_low"] - self.params.inner_strike_h,
            ],
            [0, 1, self.points["x_point"]["z_low"] - self.params.inner_strike_h],
            [0, 0, self.points["x_point"]["z_low"] - self.params.inner_strike_h],
        )

        mid_plane_outer_leg_cut = Plane(
            [
                self.points["x_point"]["x"],
                0,
                self.points["x_point"]["z_low"] - self.params.outer_strike_h,
            ],
            [0, 1, self.points["x_point"]["z_low"] - self.params.outer_strike_h],
            [0, 0, self.points["x_point"]["z_low"] - self.params.outer_strike_h],
        )

        inner_strike = next(
            filter(
                lambda p: p[0] < self.points["x_point"]["x"],
                loop_plane_intersect(self.separatrix, mid_plane_inner_leg_cut),
            )
        )

        outer_strike = next(
            filter(
                lambda p: p[0] > self.points["x_point"]["x"],
                loop_plane_intersect(self.separatrix, mid_plane_outer_leg_cut),
            )
        )

        # Divertor entrance
        div_left_limit = np.array(
            [
                self.points["x_point"]["x"] - self.params.xpt_inner_gap,
                self.points["x_point"]["z_low"],
            ],
        )
        div_right_limit = np.array(
            [
                self.points["x_point"]["x"] + self.params.xpt_outer_gap,
                self.points["x_point"]["z_low"],
            ],
        )

        x_point_plane = Plane(
            [0, 0, self.points["x_point"]["z_low"]],
            [0, 1, self.points["x_point"]["z_low"]],
            [10, 0, self.points["x_point"]["z_low"]],
        )

        tmp_div_top_right_limit_x = next(
            filter(
                lambda p: p[0] > self.points["x_point"]["x"],
                loop_plane_intersect(fw_loop, x_point_plane),
            )
        )[0]

        if tmp_div_top_right_limit_x > div_right_limit[0]:
            div_right_limit[0] = tmp_div_top_right_limit_x
        else:
            pass

        x_div = [
            div_left_limit[0],
            inner_strike[0] - self.params.inner_target_SOL,
            inner_strike[0] + self.params.inner_target_PFR,
            outer_strike[0] - self.params.outer_target_PFR,
            outer_strike[0] + self.params.outer_target_SOL,
            div_right_limit[0],
        ]
        z_div = [
            div_left_limit[1],
            inner_strike[2],
            inner_strike[2],
            outer_strike[2],
            outer_strike[2],
            div_right_limit[1],
        ]

        divertor_loop = Loop(x=x_div, z=z_div)
        divertor_loop.close()

        fw_loop.close()
        union = boolean_2d_union(fw_loop, divertor_loop)
        fw_diverted_loop = union

        return fw_diverted_loop

    def make_flux_surfaces(self, step_size=0.005):
        """
        Generate a set of flux surfaces placed between lcfs and fw

        Parameters
        ----------
        step_size: float
            Defines the thickness of each flux surface at the midplane

        Attributes
        ----------
        self.flux_surfaces: [Loop]
            Set of flux surfaces to discretise the SOL
        self.flux_surface_width_omp: [float]
            Thickness of flux sirfaces
        """
        self.flux_surfaces = []
        x_omp = self.x_omp_lcfs + self.lcfs_shift
        double_step = 2 * step_size

        profile_x_omp = next(
            filter(
                lambda p: p[0] > self.points["o_point"]["x"],
                loop_plane_intersect(self.profile, self.mid_plane),
            )
        )[0]

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
                x_int, z_int = fs.find_intersections(fs.loop_lfs, self.profile)
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

    def calculate_parameters_for_heat_flux(self, qpar_omp):
        """
        Calculate the parameters for the heat flux calculation
        The parameters are collected by flux surface
        len(flux_surface_list) == len(parameter_list)

        Parameters
        ----------
        qpar_omp: [float]
            Parallel contribution of the power carried by all the fs at the omp

        Returns
        -------
        lfs_first_int: [float, float, float]
            x, z and theta coordinate of the first intersection for each fs at the lfs
        hfs_first_int: [float, float, float]
            x, z and theta coordinate of the first intersection for each fs at the hfs
        qpar_omp: [float]
            Parallel contribution of the power carried by all the fs at the omp
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
        lfs_first_int = []
        hfs_first_int = []

        qpar_local_lfs = []
        qpar_local_hfs = []

        incindent_angle_lfs = []
        incindent_angle_hfs = []

        f_lfs_list = []  # target flux expansion at the lfs
        f_hfs_list = []  # target flux expansion at the hfs

        for fs, q in zip(self.flux_surfaces, qpar_omp):
            x_int, z_int = fs.find_intersections(fs.loop_lfs, self.inner_profile[0])
            theta = fs.polar_coordinates(x_int, z_int)

            # assign to points to lfs/hfs
            lfs_points, hfs_points = fs.assign_lfs_hfs_sn(x_int, z_int, theta)
            lfs_points_x, lfs_points_z, lfs_points_theta = lfs_points
            hfs_points_x, hfs_points_z, hfs_points_theta = hfs_points

            first_int_lfs = fs.find_first_intersection_lfs_sn(
                lfs_points_x,
                lfs_points_z,
                lfs_points_theta,
            )

            lfs_first_int.append(first_int_lfs)

            # q parallel local at lfs
            q_local_lfs = fs.calculate_q_par_local(
                first_int_lfs[0],
                first_int_lfs[1],
                q / self.power_correction_factor_omp,
            )
            qpar_local_lfs.append(q_local_lfs)

            angle = fs.calculate_incindent_angle(
                fs.loop_lfs,
                first_int_lfs[0],
                first_int_lfs[1],
                self.inner_profile[0],
            )
            incindent_angle_lfs.append(angle)
            # flux expansion
            f_lfs = fs.f
            f_lfs_list.append(f_lfs)

            if len(hfs_points_x) > 0:

                first_int_hfs = fs.find_first_intersection_hfs_sn(
                    hfs_points_x,
                    hfs_points_z,
                    hfs_points_theta,
                )

                hfs_first_int.append(first_int_hfs)

                # q parallel local at lfs
                q_local_hfs = fs.calculate_q_par_local(
                    first_int_hfs[0],
                    first_int_hfs[1],
                    q / self.power_correction_factor_omp,
                )
                qpar_local_hfs.append(q_local_hfs)

                angle = fs.calculate_incindent_angle(
                    fs.loop_lfs,
                    first_int_hfs[0],
                    first_int_hfs[1],
                    self.inner_profile[0],
                )
                incindent_angle_hfs.append(angle)
                # flux expansion
                f_hfs = fs.f
                f_hfs_list.append(f_hfs)

        return (
            lfs_first_int,
            hfs_first_int,
            qpar_omp,
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
        Collect al the final parameters under single lists

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
        th_int_hf: [float]
            List of all the theta coordinates at the intersections
        heat_flux: [float]
            List of all the heat fluxes
        """
        x_int_hf = []
        z_int_hf = []
        th_int_hf = []
        heat_flux = []

        for list_xz, hf in zip(lfs_first_int, heat_flux_lfs):
            if list_xz is not None:
                x_int_hf.append(list_xz[0])
                z_int_hf.append(list_xz[1])
                th_int_hf.append(list_xz[2])
                heat_flux.append(hf)
        for list_xz, hf in zip(hfs_first_int, heat_flux_hfs):
            if list_xz is not None:
                x_int_hf.append(list_xz[0])
                z_int_hf.append(list_xz[1])
                th_int_hf.append(list_xz[2])
                heat_flux.append(hf)
        return (x_int_hf, z_int_hf, th_int_hf, heat_flux)


class FirstWallDN(EqInputs, ReactorSystem):
    """
    Reactor First Wall (FW) system

    First Wall design for a DN configuration
    The user needs to change the default parameters according to the case
    """

    config: Type[ParameterFrame]
    inputs: dict

    # fmt: off
    default_params = [
        ["n_TF", "Number of TF coils", 16, "N/A", None, "Input"],
        ["A", "Plasma aspect ratio", 3.1, "N/A", None, "Input"],
        ["fw_psi_init", "Initial psi norm value", 1, "N/A", None, "Input"],
        ["fw_dpsi_n_near", "Step size of psi in near SOL", 0.1, "N/A", None, "Input"],
        ["fw_dpsi_n_far", "Step size of psi in far SOL", 0.1, "N/A", None, "Input"],
        ["fw_dx_omp", "Initial offset from LCFS omp", 0.2, "m", None, "Input"],
        ["fw_dx_imp", "Initial offset from LCFS imp", 0.05, "m", None, "Input"],
        ["fw_psi_n_prel", "psi for preliminary profile", 1, "N/A", None, "Input"],
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
        ["tk_bb_fw", "First wall thickness", 0.052, "m", None, "Input"],
        # External inputs to draw the divertor
        ["xpt_outer_gap", "Gap between x-point and outer wall", 0.5, "m", None, "Input"],
        ["xpt_inner_gap", "Gap between x-point and inner wall", 0.4, "m", None, "Input"],
        ["outer_strike_r", "Outer strike point major radius", 10.3, "m", None, "Input"],
        ["inner_strike_r", "Inner strike point major radius", 8, "m", None, "Input"],
        ["outer_target_SOL", "Outer target length SOL side", 0.4, "m", None, "Input"],
        ["outer_target_PFR", "Outer target length PFR side", 0.4, "m", None, "Input"],
        ["theta_outer_target", "Target-Separatrix angle", 20, "deg", None, "Input"],
        ["inner_target_SOL", "Inner target length SOL side", 0.2, "m", None, "Input"],
        ["inner_target_PFR", "Inner target length PFR side", 0.2, "m", None, "Input"],
        ["xpt_height", "x-point vertical_gap", 0.35, "m", None, "Input"],
    ]
    # fmt: on
    CADConstructor = FirstWallCAD

    def __init__(self, config, inputs):
        self.config = config
        self.inputs = inputs

        self.params = ParameterFrame(self.default_params.to_records())
        self.params.update_kw_parameters(self.config)

        self.fw_p_sol_near_omp = self.params.fw_p_sol_near * self.params.p_rate_omp
        self.fw_p_sol_far_omp = self.params.fw_p_sol_far * self.params.p_rate_omp
        self.fw_p_sol_near_imp = self.params.fw_p_sol_near * self.params.p_rate_imp
        self.fw_p_sol_far_imp = self.params.fw_p_sol_far * self.params.p_rate_imp

        self.equilibrium = inputs["equilibrium"]
        super().load_equilibrium()

        if "profile" in inputs:
            self.profile = inputs["profile"]
        else:
            self.profile = self.make_preliminary_profile()

        self.make_flux_surfaces()

        self.inner_profile = self.make_divertor(self.profile)
        outer_profile = self.inner_profile.offset(self.params.tk_bb_fw)
        outer_profile = simplify_loop(outer_profile)

        self.geom["2D profile"] = Shell(inner=self.inner_profile, outer=outer_profile)

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
        psi_n_loop = self.equilibrium.get_flux_surface(self.params.fw_psi_n_prel)

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

    def make_divertor_outer_target(self):
        """
        Make a divertor outer target

        Returns
        -------
        outer_target_internal_point: [float, float]
            x and z coordinates of the internal point.
            Meaning private flux region side
        outer_target_external_point: [float, float]
            x and z coordinates of the internal point.
            Meaning SOL side

        The divertor target is a straight line
        """
        # Get strike point
        mid_plane_outer_leg_limit = Plane(
            [self.params.outer_strike_r, 0, self.points["x_point"]["z_low"]],
            [self.params.outer_strike_r, 1, self.points["x_point"]["z_low"]],
            [
                self.params.outer_strike_r,
                0,
                self.points["x_point"]["z_low"] - self.x_point_shift,
            ],
        )

        outer_strike = next(
            filter(
                lambda p: p[0] > self.points["x_point"]["x"],
                loop_plane_intersect(self.separatrix[0], mid_plane_outer_leg_limit),
            )
        )

        sep_p_ind = np.roll(np.arange(self.separatrix[0].x.size), -1)
        arr = np.array(
            [
                [self.separatrix[0].x, self.separatrix[0].z],
                [self.separatrix[0].x[sep_p_ind], self.separatrix[0].z[sep_p_ind]],
            ]
        )
        int_ind = np.where(
            [
                check_linesegment(
                    arr[0, :, i], arr[1, :, i], [outer_strike[0], outer_strike[2]]
                )
                for i in range(self.separatrix[0].x.size)
            ]
        )
        p_before_strike = (
            self.separatrix[0].x[int_ind][0],
            self.separatrix[0].z[int_ind][0],
        )

        v1 = p_before_strike - np.array([outer_strike[0], outer_strike[2]])
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = rotate_vector_2d(v1_u, np.radians(self.params.theta_outer_target))

        # Find endpoints
        target_int = v2_u * self.params.outer_target_PFR
        target_ext = v2_u * self.params.outer_target_SOL
        target = np.array([-target_int, target_ext])
        target = target + [outer_strike[0], outer_strike[2]]
        if target[0, 0] < target[1, 0]:
            outer_target_internal_point = np.array([target[0, 0], target[0, 1]])
            outer_target_external_point = np.array([target[1, 0], target[1, 1]])
        elif target[0, 0] > target[1, 0]:
            outer_target_internal_point = np.array([target[1, 0], target[1, 1]])
            outer_target_external_point = np.array([target[0, 0], target[0, 1]])

        return (outer_target_internal_point, outer_target_external_point)

    def make_divertor_inner_target(self):
        """
        Make a divertor inner target

        Returns
        -------
        inner_target_internal_point: [float, float]
            x and z coordinates of the internal point.
            Meaning SOL side
        inner_target_external_point: [float, float]
            x and z coordinates of the internal point.
            Meaning private flux region side
        """
        mid_plane_inner_leg_limit = Plane(
            [self.params.inner_strike_r, 0, self.points["x_point"]["z_low"]],
            [self.params.inner_strike_r, 1, self.points["x_point"]["z_low"]],
            [
                self.params.inner_strike_r,
                0,
                self.points["x_point"]["z_low"] - self.x_point_shift,
            ],
        )

        inner_strike = next(
            filter(
                lambda p: p[0] < self.points["x_point"]["x"],
                loop_plane_intersect(self.separatrix[1], mid_plane_inner_leg_limit),
            )
        )

        inner_target_internal_point = [
            self.points["x_point"]["x"] - self.params.xpt_inner_gap,
            self.points["x_point"]["z_low"],
        ]

        m = lineq(inner_target_internal_point, [inner_strike[0], inner_strike[2]])[0]

        x_target_external_point = (
            inner_strike[0] - (self.params.inner_target_PFR ** 2) + (m * inner_strike[0])
        ) / (1 + m)

        z_target_external_point = (
            m * (x_target_external_point - inner_strike[0]) + inner_strike[2]
        )

        inner_target_external_point = [x_target_external_point, z_target_external_point]

        return (inner_target_internal_point, inner_target_external_point)

    def make_divertor(self, fw_loop):
        """
        Make a divertor: attaches a divertor to the first wall

        Parameters
        ----------
        fw_loop: Loop
            first wall profile

        Returns
        -------
        fw_diverted_loop: Loop
            Complete first wall profile (with divertor)
        """
        (
            outer_target_internal_point,
            outer_target_external_point,
        ) = self.make_divertor_outer_target()
        (
            inner_target_internal_point,
            inner_target_external_point,
        ) = self.make_divertor_inner_target()

        middle_point = [
            self.points["x_point"]["x"],
            self.points["x_point"]["z_low"] - self.params.xpt_height,
        ]

        outer_leg_central_guide_line = self.separatrix[0]
        top_clip_outer_leg_central_guide_line = np.where(
            outer_leg_central_guide_line.z < self.points["x_point"]["z_low"],
        )

        outer_leg_central_guide_line = Loop(
            outer_leg_central_guide_line.x[top_clip_outer_leg_central_guide_line],
            z=outer_leg_central_guide_line.z[top_clip_outer_leg_central_guide_line],
        )
        bottom_clip_outer_leg_central_guide_line = np.where(
            outer_leg_central_guide_line.z > outer_target_internal_point[1],
        )
        outer_leg_central_guide_line = Loop(
            outer_leg_central_guide_line.x[bottom_clip_outer_leg_central_guide_line],
            z=outer_leg_central_guide_line.z[bottom_clip_outer_leg_central_guide_line],
        )

        outer_target = np.array(
            [
                [
                    outer_target_internal_point[0],
                    outer_target_external_point[0],
                ],
                [
                    outer_target_internal_point[1],
                    outer_target_external_point[1],
                ],
            ]
        )

        inner_target = np.array(
            [
                [
                    inner_target_internal_point[0],
                    inner_target_external_point[0],
                ],
                [
                    inner_target_internal_point[1],
                    inner_target_external_point[1],
                ],
            ]
        )

        # Divertor entrance
        # div_top_left_limit = np.array(
        #     [
        #         self.points["x_point"]["x"] - self.params.xpt_inner_gap,
        #         self.points["x_point"]["z_low"],
        #     ],
        # )
        div_top_right_limit = np.array(
            [
                self.points["x_point"]["x"] + self.params.xpt_outer_gap,
                self.points["x_point"]["z_low"],
            ],
        )

        # To modify the inputs for the divertor entrance if needed to lower the hf
        x_point_plane = Plane(
            [0, 0, self.points["x_point"]["z_low"]],
            [0, 1, self.points["x_point"]["z_low"]],
            [10, 0, self.points["x_point"]["z_low"]],
        )

        tmp_div_top_right_limit_x = next(
            filter(
                lambda p: p[0] > self.points["x_point"]["x"],
                loop_plane_intersect(fw_loop, x_point_plane),
            )
        )[0]

        if tmp_div_top_right_limit_x > div_top_right_limit[0]:
            div_top_right_limit[0] = tmp_div_top_right_limit_x
        else:
            pass

        (outer_leg_external_line_x, outer_leg_external_line_z,) = self.reshape_curve(
            outer_leg_central_guide_line.x,
            outer_leg_central_guide_line.z,
            [div_top_right_limit[0], div_top_right_limit[1]],
            outer_target[:, 1],
            2,
        )

        (outer_leg_internal_line_x, outer_leg_internal_line_z,) = self.reshape_curve(
            outer_leg_central_guide_line.x,
            outer_leg_central_guide_line.z,
            [middle_point[0], middle_point[1]],
            outer_target[:, 0],
            3,
        )

        outer_leg_x = np.append(
            outer_leg_internal_line_x, outer_leg_external_line_x[::-1]
        )
        outer_leg_z = np.append(
            outer_leg_internal_line_z, outer_leg_external_line_z[::-1]
        )

        inner_leg_x = np.append(inner_target[0], middle_point[0])
        inner_leg_z = np.append(inner_target[1], middle_point[1])

        x_div_bottom = np.append(inner_leg_x, outer_leg_x)
        z_div_bottom = np.append(inner_leg_z, outer_leg_z)
        x_div_bottom = [round(elem, 5) for elem in x_div_bottom]
        z_div_bottom = [round(elem, 5) for elem in z_div_bottom]

        x_div_top = x_div_bottom
        z_div_top = [z * -1 for z in z_div_bottom]

        bottom_divertor_loop = Loop(x=x_div_bottom, z=z_div_bottom)
        top_divertor_loop = Loop(x=x_div_top, z=z_div_top)

        bottom_divertor_loop.close()
        top_divertor_loop.close()

        union = boolean_2d_union(fw_loop, bottom_divertor_loop)
        union = boolean_2d_union(top_divertor_loop, union)
        fw_diverted_loop = union[0]
        return fw_diverted_loop

    def make_flux_surfaces(self):
        """
        Generate a set of flux surfaces placed between lcfs and fw

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
        self.flux_surfaces = []
        x_omp = self.x_omp_lcfs + self.params.dr_near_omp

        profile_x_omp = next(
            filter(
                lambda p: p[0] > self.points["o_point"]["x"],
                loop_plane_intersect(self.profile, self.mid_plane),
            )
        )[0]

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
                > next(
                    filter(
                        lambda p: p[0] < self.points["o_point"]["x"],
                        loop_plane_intersect(self.profile, self.mid_plane),
                    ),
                )[0]
            ):
                self.flux_surface_hfs.append(fs.loop_hfs)

        for fs in self.flux_surfaces:
            if hasattr(fs, "loop_lfs") and fs.find_intersections(
                fs.loop_lfs, self.profile
            ):
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

    def find_intersections(self):
        """
        Find the intersections between all the flux surfaces and the first wall

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
            x_int, z_int = fs.find_intersections(loop, self.inner_profile)
            hfs_intersections_x.append(x_int)
            hfs_intersections_z.append(z_int)

        for loop, fs in zip(self.flux_surface_lfs, self.flux_surfaces):
            x_int, z_int = fs.find_intersections(loop, self.inner_profile)
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
        Find the first intersections between all the flux surfaces and the first wall

        Parameters
        ----------
        lfs_intersections_x: [[float]]
            x coordinate of all the intersections of all the fs at lfs
        lfs_intersections_z: [[float]]
            z coordinate of all the intersections of all the fs at lfs
        hfs_intersections_x: [[float]]
            x coordinate of all the intersections of all the fs at hfs
        hfs_intersections_z: [[float]]
            z coordinate of all the intersections of all the fs at hfs

        Returns
        -------
        lfs_down_first_intersections: [float, float]
            x, z coordinates of first intersections at lfs bottom
        lfs_up_first_intersections: [float, float]
            x, z coordinates of first intersections at lfs top
        hfs_down_first_intersections: [float, float]
            x, z coordinates of first intersections at hfs bottom
        hfs_up_first_intersections: [float, float]
            x, z coordinates of first intersections at hfs top
        """
        lfs_down_first_intersections = []
        lfs_up_first_intersections = []
        hfs_down_first_intersections = []
        hfs_up_first_intersections = []

        for x, z, loop, fs in zip(
            lfs_intersections_x,
            lfs_intersections_z,
            self.flux_surface_lfs,
            self.flux_surfaces,
        ):
            clips_lfs_up, clips_lfs_down = fs.snip_flux_surface(loop, x, z)
            lfs_top_intersections, lfs_bottom_intersections = fs.assign_top_bottom(x, z)

            lfs_down = np.where(
                len(lfs_bottom_intersections[0]) != 0,
                fs.find_first_intersection_dn(lfs_bottom_intersections, clips_lfs_down),
                None,
            )
            lfs_down_first_intersections.append(lfs_down)

            lfs_up = np.where(
                len(lfs_top_intersections[0]) != 0,
                fs.find_first_intersection_dn(lfs_top_intersections, clips_lfs_up),
                None,
            )
            lfs_up_first_intersections.append(lfs_up)

        for x, z, loop, fs in zip(
            hfs_intersections_x,
            hfs_intersections_z,
            self.flux_surface_hfs,
            self.flux_surfaces,
        ):
            clips_hfs_up, clips_hfs_down = fs.snip_flux_surface(loop, x, z)
            hfs_top_intersections, hfs_bottom_intersections = fs.assign_top_bottom(x, z)

            hfs_down = np.where(
                len(hfs_bottom_intersections[0]) != 0,
                fs.find_first_intersection_dn(hfs_bottom_intersections, clips_hfs_down),
                None,
            )
            hfs_down_first_intersections.append(hfs_down)

            hfs_up = np.where(
                len(hfs_top_intersections[0]) != 0,
                fs.find_first_intersection_dn(hfs_top_intersections, clips_hfs_up),
                None,
            )
            hfs_up_first_intersections.append(hfs_up)

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
                    None,
                )
                to_append.append(test)

        return (list_qpar_target, list_incident_angle, list_flux_expansion)

    def define_flux_surfaces_parameters_to_calculate_heat_flux(
        self,
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
                self.inner_profile,
            )
            qpar_local_lfs.append(q)
            incindent_angle_lfs.append(angle)
            f_lfs_list.append(f)

        for intersection in [hfs_down_first_intersections, hfs_up_first_intersections]:
            (q, angle, f) = self.calculate_parameters_for_heat_flux(
                qpar_imp,
                intersection,
                self.flux_surface_hfs,
                self.inner_profile,
            )
            qpar_local_hfs.append(q)
            incindent_angle_hfs.append(angle)
            f_hfs_list.append(f)

        return (
            np.array([qpar_local_lfs, qpar_local_hfs]),
            np.array([incindent_angle_lfs, incindent_angle_hfs]),
            np.array([f_lfs_list, f_hfs_list]),
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
                None,
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
            List of all the x coordinates at the inetrsections
        z_int_hf: [float]
            List of all the z coordinates at the intersections
        heat_flux: [float]
            List of all the heat fluxes
        """
        x_int_hf = []
        z_int_hf = []
        heat_flux = []

        for list_xz, hf in zip(list_first_intersections, list_heat_flux):
            attributes = [lambda: list_xz[0], lambda: list_xz[1], lambda: hf]
            lists = [x_int_hf, z_int_hf, heat_flux]
            for attribute, to_append in zip(attributes, lists):
                test = np.where(
                    len(list_xz) != 0,
                    attribute(),
                    None,
                )
                to_append.append(test)

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

    def clipper(self, loop, clip_vertical):
        """
        Loop clipper

        Parameters
        ----------
        loop: Loop
            Loop to cut
        clip_vertical: [float, float]
            Reference axis against which to cut

        Returns
        -------
        Loop: Loop
            New modified loop
        """
        new_loop = Loop(loop.x[clip_vertical], z=loop.z[clip_vertical])

        clip_bottom = np.where(
            new_loop.z > self.points["x_point"]["z_low"] - self.x_point_shift
        )

        new_loop = Loop(new_loop.x[clip_bottom], z=new_loop.z[clip_bottom])

        clip_top = np.where(
            new_loop.z < self.points["x_point"]["z_up"] + self.x_point_shift
        )

        return Loop(new_loop.x[clip_top], z=new_loop.z[clip_top])

    def modify_fw_profile(self, profile, x_int_hf, z_int_hf, heat_flux):
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
        if (
            z_int_hf > self.points["x_point"]["z_low"]
            and z_int_hf < self.points["x_point"]["z_up"]
            and heat_flux > self.params.hf_limit
        ):
            self.loops = self.equilibrium.get_flux_surface_through_point(
                x_int_hf, z_int_hf
            )
            for loop in self.loops:
                if loop_plane_intersect(loop, self.mid_plane) is not None:

                    if (
                        loop_plane_intersect(loop, self.mid_plane)[0][0]
                        > self.points["o_point"]["x"]
                    ):
                        clip_vertical = np.where(loop.x > self.points["x_point"]["x"])

                        clipped_loops.append(self.clipper(loop, clip_vertical))

                    elif loop_plane_intersect(loop, self.mid_plane)[0][0] < self.points[
                        "o_point"
                    ]["x"] and loop_plane_intersect(loop, self.mid_plane)[0][0] > (
                        self.x_imp_lcfs - self.params.fw_dx_imp
                    ):
                        clip_vertical = np.where(loop.x < self.points["x_point"]["x"])

                        clipped_loops.append(self.clipper(loop, clip_vertical))

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
