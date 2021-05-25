# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.

"""
Flux surface attributes and first wall profile based on heat flux calculation
"""
import numpy as np
from typing import Type
from BLUEPRINT.base import ReactorSystem, ParameterFrame
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.boolean import (
    convex_hull,
    boolean_2d_difference,
)
from BLUEPRINT.geometry.geomtools import (
    get_intersect,
    check_linesegment,
    loop_plane_intersect,
)
from BLUEPRINT.geometry.geombase import Plane
from BLUEPRINT.nova.firstwall import Paneller


class EqInputs:
    """
    Load a equilibrium file
    """

    def load_equilibrium(self):
        """
        Extract basic information
        """
        self.lcfs = self.equilibrium.get_LCFS()
        self.separatrix = self.equilibrium.get_separatrix()
        o_point = self.equilibrium.get_OX_points()[0]  # Find O point coordinates
        x_point = self.equilibrium.get_OX_points()[1]
        self.x_o_point = o_point[0][0]
        self.z_o_point = o_point[0][1]
        self.x_x_point = x_point[0][0]
        self.z_x_point = x_point[0][1]
        mid_plane = Plane(
            [self.x_o_point, 0, self.z_o_point],
            [0, 1, self.z_o_point],
            [0, 0, self.z_o_point],
        )
        self.x_omp_lcfs = next(
            filter(
                lambda p: p[0] > self.x_o_point,
                loop_plane_intersect(self.lcfs, mid_plane),
            )
        )[0]


class FluxSurface(EqInputs):
    """
    Create a flux surface
    Evaluate all needed attributes which lead to the heat flux calculation
    """

    def __init__(self, equilibrium, psi_norm):
        self.equilibrium = equilibrium
        super().load_equilibrium()

        self.loop = self.equilibrium.get_flux_surface(psi_norm)
        mid_plane = Plane(
            [self.x_o_point, 0, self.z_o_point],
            [0, 1, self.z_o_point],
            [0, 0, self.z_o_point],
        )
        self.x_omp, y, z_omp = next(
            filter(
                lambda p: p[0] > self.x_o_point,
                loop_plane_intersect(self.loop, mid_plane),
            )
        )
        self.Bt_omp = self.equilibrium.Bt(self.x_omp)
        self.Bp_omp = self.equilibrium.Bp(self.x_omp, z_omp)
        self.B_omp = np.sqrt(self.Bt_omp ** 2 + self.Bp_omp ** 2)
        self.dr_omp = self.x_omp - self.x_omp_lcfs

    def find_intersections(self, fw_profile):
        """
        Find intersection between a flux surface and a given first wall profile
        """
        x_int, z_int = get_intersect(fw_profile, self.loop)
        return (x_int, z_int)

    def polar_coordinates(self, x_int, z_int):
        """
        Calculate the polar coordinate theta for intersection points
        """
        theta_coord = []
        for x, z in zip(x_int, z_int):
            if (x - self.x_x_point) > 0 and (z - self.z_o_point) > 0:
                theta_coord.append(
                    (np.arctan((z - self.z_o_point) / (x - self.x_x_point)))
                    * (180 / np.pi)
                )
            elif (x - self.x_x_point) > 0 and (z - self.z_o_point) < 0:
                theta_coord.append(
                    (
                        (np.arctan((z - self.z_o_point) / (x - self.x_x_point)))
                        * (180 / np.pi)
                    )
                    + 360
                )
            elif (x - self.x_x_point) < 0:
                theta_coord.append(
                    (
                        (np.arctan((z - self.z_o_point) / (x - self.x_x_point)))
                        * (180 / np.pi)
                    )
                    + 180
                )
        return theta_coord

    def assign_lfs_hfs(self, x_int, z_int, theta_coord):
        """
        Assign intersection points between low field side and high field side
        """
        lfs_int_points = [], [], []
        hfs_int_points = [], [], []
        for x, z, th in zip(x_int, z_int, theta_coord):
            if x > self.x_x_point and z < self.z_o_point:
                lfs_int_points[0].append(x)
                lfs_int_points[1].append(z)
                lfs_int_points[2].append(th)
            else:
                hfs_int_points[0].append(x)
                hfs_int_points[1].append(z)
                hfs_int_points[2].append(th)
        return (lfs_int_points, hfs_int_points)

    def find_first_intersection_lfs(self, x_int_lfs, z_int_lfs, theta_coord_lfs):
        """
        Find the first intersection point at the low field side
        """
        for x, z, th in zip(x_int_lfs, z_int_lfs, theta_coord_lfs):
            if th == max(theta_coord_lfs):
                lfs_first_int_x, lfs_first_int_z, lfs_first_int_theta = x, z, th
                return (lfs_first_int_x, lfs_first_int_z, lfs_first_int_theta)

    def find_first_intersection_hfs(self, x_int_hfs, z_int_hfs, theta_coord_hfs):
        """
        Find the first intersection point at the high field side
        """
        for x, z, th in zip(x_int_hfs, z_int_hfs, theta_coord_hfs):
            if th == min(theta_coord_hfs):
                hfs_first_int_x, hfs_first_int_z, hfs_first_int_theta = x, z, th
                return (hfs_first_int_x, hfs_first_int_z, hfs_first_int_theta)

    def calculate_q_par_omp(self, Psol_near, Psol_far, lambdaq_near, lambdaq_far):
        """
        Calculate the parallel power density at the separatrix at outer midplane
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

    def calculate_q_par_local(self, x_int, z_int, qpar_omp):
        """
        Calculate the parallel power density associated to a given point
        """
        Bp_local = self.equilibrium.Bp(x_int, z_int)
        self.f = (self.x_omp * self.Bp_omp) / (x_int * Bp_local)
        qpar_local = (
            qpar_omp * (self.Bp_omp / self.B_omp) * (self.x_omp / x_int) * (1 / self.f)
        )
        return qpar_local

    def calculate_glancing_angle(self, x_int, z_int, fw_profile):
        """
        Calculate the glancing angle at a given intersection with fw
        """
        for i, j in zip(range(len(self.loop.x)), range(len(self.loop.z))):
            if check_linesegment(
                [self.loop.x[i], self.loop.z[j]],
                [
                    self.loop.x[
                        (i + 1) % len(self.loop.x),
                    ],
                    self.loop.z[(j + 1) % len(self.loop.z)],
                ],
                [x_int, z_int],
            ):
                p_int = (x_int, z_int)
                p_before_int = (self.loop.x[i], self.loop.z[j])

        for i_wall, j_wall in zip(range(len(fw_profile.x)), range(len(fw_profile.z))):
            if check_linesegment(
                [fw_profile.x[i_wall], fw_profile.z[j_wall]],
                [
                    fw_profile.x[
                        (i_wall + 1) % len(fw_profile.x),
                    ],
                    fw_profile.z[(j_wall + 1) % len(fw_profile.z)],
                ],
                [x_int, z_int],
            ):
                p_int_wall = (x_int, z_int)
                p_before_int_wall = (fw_profile.x[i_wall], fw_profile.z[j_wall])
                break

        v0 = np.array(p_before_int) - np.array(p_int)
        v1 = np.array(p_before_int_wall) - np.array(p_int_wall)

        v0_u = v0 / np.linalg.norm(v0)
        v1_u = v1 / np.linalg.norm(v1)
        glancing_angle = (np.arccos(np.clip(np.dot(v0_u, v1_u), -1.0, 1.0))) * (
            180 / np.pi
        )
        glancing_angle_rad = (
            ((np.arccos(np.clip(np.dot(v0_u, v1_u), -1.0, 1.0))) * (180 / np.pi))
        ) * (np.pi / 180)
        return (glancing_angle, glancing_angle_rad)

    def calculate_heat_flux_onto_fw_surface(self, qpar, glancing_angle):
        """
        Project the power density carried by a flux surface onto the fw surface
        """
        heat_flux = qpar * np.sin(glancing_angle)  # q perp to target
        return heat_flux


class FirstWall(EqInputs, ReactorSystem):
    """
    Reactor First Wall (FW) system
    """

    config: Type[ParameterFrame]
    inputs: dict

    default_params = [
        ["plasma_type", "Type of plasma", "SN", "N/A", None, "Input"],
        ["A", "Plasma aspect ratio", 3.1, "N/A", None, "Input"],
        ["R_0", "Major radius", 9, "m", None, "Input"],
        ["fw_psi_init", "Initial psi norm value", 1, "N/A", None, "Input"],
        [
            "fw_dpsi_n_near",
            "Step size for psi norm near SOL",
            0.001,
            "N/A",
            None,
            "Input",
        ],
        ["fw_dpsi_n_far", "Step size for psi norm far SOL", 0.001, "N/A", None, "Input"],
        ["fw_dx", "Initial offset from LCFS", 0.3, "m", None, "Input"],
        [
            "fw_psi_n_preliminary",
            "psi norm value for preliminary profile",
            1.01,
            "N/A",
            None,
            "Input",
        ],
        ["fw_p_sol_near", "near Scrape off layer power", 50, "MW", None, "Input"],
        ["fw_p_sol_far", "far Scrape off layer power", 50, "MW", None, "Input"],
        ["fw_lambda_q_near", "Lambda q near SOL", 0.05, "m", None, "Input"],
        ["fw_lambda_q_far", "Lambda q far SOL", 0.05, "m", None, "Input"],
        ["f_outer_target", "Power fraction", 0.75, "N/A", None, "Input"],
        ["f_inner_target", "Power fraction", 0.25, "N/A", None, "Input"],
    ]

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

    def make_preliminary_profile(self):
        """
        Generate a preliminary first wall profile shape
        """
        dx_loop = self.lcfs.offset(self.params.fw_dx)
        psi_n_loop = self.equilibrium.get_flux_surface(self.params.fw_psi_n_preliminary)

        (equilibrium_opoint, equilibrium_xpoints) = self.equilibrium.get_OX_points()
        self.equilibrium_xpoint = equilibrium_xpoints[0]
        self.equilibrium_z_xpoint = self.equilibrium_xpoint.z

        clipped_loops = []
        clip1 = np.where(dx_loop.z > self.equilibrium_z_xpoint)
        clip2 = np.where(psi_n_loop.z > self.equilibrium_z_xpoint)
        new_loop1 = Loop(dx_loop.x[clip1], z=dx_loop.z[clip1])
        new_loop2 = Loop(psi_n_loop.x[clip2], z=psi_n_loop.z[clip2])
        clipped_loops.append(new_loop1)
        clipped_loops.append(new_loop2)
        hull = convex_hull(clipped_loops)

        # excluding divertor region
        hull.interpolate(200)
        z_min = min(hull.z)
        div_box = Loop(x=[5, 12, 12, 0, 5], z=[-10, -10, z_min + 1.5, z_min + 1.5, -10])
        count = 0
        for i, point in enumerate(hull.d2.T):
            if div_box.point_in_poly(point):
                if count > 2:
                    hull.reorder(i, 0)
                    hull.open_()
                    break
                count += 1
        hull = boolean_2d_difference(hull, div_box)[0]

        # Panelling the smooth silhouette
        paneller = Paneller(hull.x, hull.z, angle=20, dx_min=0.5, dx_max=1)
        paneller.optimise()
        x, z = paneller.d2
        fw_loop = Loop(x=x, z=z)
        x_div = [6, 7, 7.5, 8, 8.5, 10.2]
        z_div = [-6.5, -6.5, -6.5, -6.5, -6.5, -6.5]
        x = np.append(fw_loop.x, x_div)
        z = np.append(fw_loop.z, z_div)
        fw_loop = Loop(x=x, z=z)
        fw_loop.close()
        fw_loop.to_json("first_wall.json")
        return fw_loop

    def make_flux_surfaces(self):
        """
        Generate a set of flux surfaces placed between lcfs and fw
        """
        self.flux_surfaces = []
        psi_norm = self.params.fw_psi_init
        x_omp = 0.0
        profile_x_omp = self.profile.intersect(
            [self.x_o_point, self.z_o_point], [16, self.z_o_point]
        ).max()

        while x_omp < profile_x_omp:
            flux_surface = FluxSurface(self.equilibrium, psi_norm)
            x_omp = flux_surface.x_omp
            self.flux_surfaces.append(flux_surface)
            if (
                flux_surface.x_omp - flux_surface.x_omp_lcfs
            ) < self.params.fw_lambda_q_near:
                psi_norm += self.params.fw_dpsi_n_near
            else:
                psi_norm += self.params.fw_dpsi_n_far

        if len(self.flux_surfaces) == 0:
            raise ValueError(f"fs for initial psi = {self.params.fw_psi_n} outside fw")

        for fs in self.flux_surfaces:  # exclude empty flux surfaces
            x_int, z_int = fs.find_intersections(self.profile)
            if len(x_int) == 0:
                self.flux_surfaces.remove(fs)

        if len(self.flux_surfaces) == 0:
            raise ValueError(
                "No intersections found between Flux Surfaces and First Wall."
            )
        self.flux_surfaces = self.flux_surfaces[:-1]  # the last one is outside the wall

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

    def define_flux_surfaces_parameters(self):
        """
        Intersect the set of flux surfaces with fw
        Define parameters of intersection points
        """
        lfs_first_intersection = []
        hfs_first_intersection = []

        qpar_omp = []
        qpar_local_lfs = []
        qpar_local_hfs = []

        glancing_angle_lfs = []
        glancing_angle_hfs = []

        f_lfs_list = []  # target flux expansion at the lfs
        f_hfs_list = []  # target flux expansion at the hfs

        for fs in self.flux_surfaces:
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
        power_correction_factor = integrated_power_entering_omp / (
            self.params.fw_p_sol_near + self.params.fw_p_sol_far
        )

        # find intersections
        for fs, q in zip(self.flux_surfaces, qpar_omp):
            x_int, z_int = fs.find_intersections(self.profile)
            theta = fs.polar_coordinates(x_int, z_int)

            # assign to points to lfs/hfs
            lfs_points, hfs_points = fs.assign_lfs_hfs(x_int, z_int, theta)

            lfs_points_x, lfs_points_z, lfs_points_theta = lfs_points

            if len(lfs_points_x) > 0:
                first_int_lfs = fs.find_first_intersection_lfs(
                    lfs_points_x, lfs_points_z, lfs_points_theta
                )
                (
                    first_int_lfs_x,
                    first_int_lfs_z,
                    first_int_lfs_theta,
                ) = fs.find_first_intersection_lfs(
                    lfs_points_x, lfs_points_z, lfs_points_theta
                )
                lfs_first_intersection.append(first_int_lfs)

                if first_int_lfs is not None:
                    # q parallel local at lfs
                    q_local_lfs = fs.calculate_q_par_local(
                        first_int_lfs_x, first_int_lfs_z, q / power_correction_factor
                    )
                    qpar_local_lfs.append(q_local_lfs)

                    angle = fs.calculate_glancing_angle(
                        first_int_lfs_x, first_int_lfs_z, self.profile
                    )
                    glancing_angle_lfs.append(angle)
                    # flux expansion
                    f_lfs = fs.f
                    f_lfs_list.append(f_lfs)

            hfs_points_x, hfs_points_z, hfs_points_theta = hfs_points

            if len(hfs_points_x) > 0:
                first_int_hfs = fs.find_first_intersection_hfs(
                    hfs_points_x, hfs_points_z, hfs_points_theta
                )
                (
                    first_int_hfs_x,
                    first_int_hfs_z,
                    first_int_hfs_theta,
                ) = fs.find_first_intersection_hfs(
                    hfs_points_x, hfs_points_z, hfs_points_theta
                )
                hfs_first_intersection.append(first_int_hfs)

                if first_int_hfs is not None:
                    # q parallel local at hfs
                    q_local_hfs = fs.calculate_q_par_local(
                        first_int_hfs_x, first_int_hfs_z, q / power_correction_factor
                    )
                    qpar_local_hfs.append(q_local_hfs)

                    angle = fs.calculate_glancing_angle(
                        first_int_hfs_x, first_int_hfs_z, self.profile
                    )
                    glancing_angle_hfs.append(angle)
                    # flux expansion
                    f_hfs = fs.f
                    f_hfs_list.append(f_hfs)
        return (
            lfs_first_intersection,
            hfs_first_intersection,
            qpar_omp,
            qpar_local_lfs,
            qpar_local_hfs,
            glancing_angle_lfs,
            glancing_angle_hfs,
            f_lfs_list,
            f_hfs_list,
        )

    def calculate_heat_flux_lfs_hfs(
        self,
        lfs_first_intersection,
        hfs_first_intersection,
        qpar_omp,
        qpar_local_lfs,
        qpar_local_hfs,
        glancing_angle_lfs,
        glancing_angle_hfs,
    ):
        """
        Heat flux calculation
        """
        heat_flux_lfs = []
        for q, angle_rad, fs in zip(
            qpar_local_lfs, glancing_angle_lfs, self.flux_surfaces
        ):
            hf = (
                fs.calculate_heat_flux_onto_fw_surface(q, angle_rad[1])
                * self.params.f_outer_target
            )
            heat_flux_lfs.append(hf)

        heat_flux_hfs = []
        for q, angle_rad, fs in zip(
            qpar_local_hfs, glancing_angle_hfs, self.flux_surfaces
        ):
            hf = (
                fs.calculate_heat_flux_onto_fw_surface(q, angle_rad[1])
                * self.params.f_inner_target
            )
            heat_flux_hfs.append(hf)

        # Collecting intersection point coordinates and heat fluxes
        x_int_hf = []
        z_int_hf = []
        th_int_hf = []
        heat_flux = []
        for list_xz, hf in zip(lfs_first_intersection, heat_flux_lfs):
            if list_xz is not None:
                x_int_hf.append(list_xz[0])
                z_int_hf.append(list_xz[1])
                th_int_hf.append(list_xz[2])
                heat_flux.append(hf)
        for list_xz, hf in zip(hfs_first_intersection, heat_flux_hfs):
            if list_xz is not None:
                x_int_hf.append(list_xz[0])
                z_int_hf.append(list_xz[1])
                th_int_hf.append(list_xz[2])
                heat_flux.append(hf)
        return (x_int_hf, z_int_hf, heat_flux, heat_flux_lfs, heat_flux_hfs, th_int_hf)

    def calculate_power_balance_in_out(
        self,
        f_lfs_list,
        glancing_angle_lfs,
        f_hfs_list,
        glancing_angle_hfs,
        lfs_first_intersection,
        hfs_first_intersection,
        heat_flux_lfs,
        heat_flux_hfs,
    ):
        """
        Verification of power balance
        """
        d_lfs_analytic = []
        for f, dr, angle_rad in zip(
            f_lfs_list, self.flux_surface_width_omp, glancing_angle_lfs
        ):
            d = f * dr / np.sin(angle_rad[1])
            d_lfs_analytic.append(d)

        d_hfs_analytic = []
        for f, dr, angle_rad in zip(
            f_hfs_list, self.flux_surface_width_omp, glancing_angle_hfs
        ):
            d = f * dr / np.sin(angle_rad[1])
            d_hfs_analytic.append(d)

        # distance between separatrix and first open flux surface at the divertor target
        x_int_sep, z_int_sep = get_intersect(self.profile, self.separatrix)

        # distance between two flux surfaces at the target_lfs
        d_lfs = []
        d0_lfs = np.sqrt(
            ((lfs_first_intersection[0][0] - x_int_sep[1]) ** 2)
            + ((lfs_first_intersection[0][1] - z_int_sep[1]) ** 2)
        )
        d_lfs.append(d0_lfs)
        for i in range(len(lfs_first_intersection)):
            d = np.sqrt(
                (
                    (
                        lfs_first_intersection[i][0]
                        - lfs_first_intersection[(i + 1) % len(lfs_first_intersection)][
                            0
                        ]
                    )
                    ** 2
                )
                + (
                    (
                        lfs_first_intersection[i][1]
                        - lfs_first_intersection[(i + 1) % len(lfs_first_intersection)][
                            1
                        ]
                    )
                    ** 2
                )
            )
            d_lfs.append(d)
        d_lfs = d_lfs[:-1]

        # distance between two flux surfaces at the target_lfs
        d_hfs = []
        d0_hfs = np.sqrt(
            ((hfs_first_intersection[0][0] - x_int_sep[0]) ** 2)
            + ((hfs_first_intersection[0][1] - z_int_sep[0]) ** 2)
        )
        d_hfs.append(d0_hfs)
        for i in range(len(hfs_first_intersection)):
            d = np.sqrt(
                (
                    (
                        hfs_first_intersection[i][0]
                        - hfs_first_intersection[(i + 1) % len(hfs_first_intersection)][
                            0
                        ]
                    )
                    ** 2
                )
                + (
                    (
                        hfs_first_intersection[i][1]
                        - hfs_first_intersection[(i + 1) % len(hfs_first_intersection)][
                            1
                        ]
                    )
                    ** 2
                )
            )
            d_hfs.append(d)
        d_hfs = d_hfs[:-1]

        power_exiting_target_lfs_analytic = []
        for q, dx, list in zip(heat_flux_lfs, d_lfs_analytic, lfs_first_intersection):
            p = q * dx * list[0]
            power_exiting_target_lfs_analytic.append(p)
        integrated_power_lfs_analytic = (
            2 * np.pi * (sum(power_exiting_target_lfs_analytic))
        )

        power_exiting_target_lfs_geometric = []
        for q, dx, list in zip(heat_flux_lfs, d_lfs, lfs_first_intersection):
            p = q * dx * list[0]
            power_exiting_target_lfs_geometric.append(p)
        integrated_power_lfs_geometric = (
            2 * np.pi * (sum(power_exiting_target_lfs_geometric))
        )

        power_exiting_target_hfs_analytic = []
        for q, dx, list in zip(heat_flux_hfs, d_hfs_analytic, hfs_first_intersection):
            p = q * dx * list[0]
            power_exiting_target_hfs_analytic.append(p)
        integrated_power_hfs_analytic = (
            2 * np.pi * (sum(power_exiting_target_hfs_analytic))
        )

        power_exiting_target_hfs_geometric = []
        for q, dx, list in zip(heat_flux_hfs, d_hfs, hfs_first_intersection):
            p = q * dx * list[0]
            power_exiting_target_hfs_geometric.append(p)
        integrated_power_hfs_geometric = (
            2 * np.pi * (sum(power_exiting_target_hfs_geometric))
        )
        power_balance_error_analytic = (
            (
                (self.params.fw_p_sol_near + self.params.fw_p_sol_far)
                - (integrated_power_lfs_analytic + integrated_power_hfs_analytic)
            )
            / (self.params.fw_p_sol_near + self.params.fw_p_sol_far)
        ) * 100
        power_balance_error_geometric = (
            (
                (self.params.fw_p_sol_near + self.params.fw_p_sol_far)
                - (integrated_power_lfs_geometric + integrated_power_hfs_geometric)
            )
            / (self.params.fw_p_sol_near + self.params.fw_p_sol_far)
        ) * 100
        return (
            integrated_power_lfs_analytic,
            integrated_power_lfs_geometric,
            integrated_power_hfs_analytic,
            integrated_power_hfs_geometric,
            power_balance_error_analytic,
            power_balance_error_geometric,
        )
