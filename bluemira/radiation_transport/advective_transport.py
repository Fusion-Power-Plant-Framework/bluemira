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
A simplified 2-D solver for calculating charged particle heat loads.
"""

from os import sep
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from BLUEPRINT.base.parameter import ParameterFrame
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.geometry._deprecated_base import Plane
from bluemira.geometry._deprecated_tools import (
    loop_plane_intersect,
    join_intersect,
    get_angle_between_points,
    check_linesegment,
)
from bluemira.geometry._deprecated_loop import Loop
from bluemira.equilibria.find import find_flux_surfs
from bluemira.equilibria.physics import calc_psi_norm
from bluemira.radiation_transport.error import AdvectionTransportError


__all__ = ["ChargedParticleSolver"]


class OpenFluxSurface:
    """
    Utility class for handling flux surface geometries.
    """

    __slots__ = [
        "loop",
        "x_omp",
        "z_omp",
        "x_lfs_inter",
        "z_lfs_inter",
        "x_hfs_inter",
        "z_hfs_inter",
        "alpha_lfs",
        "alpha_hfs",
        "lfs_loop",
        "hfs_loop",
    ]

    def __init__(self, loop):
        self.loop = loop

        if loop.closed:
            raise AdvectionTransportError(
                "OpenFluxSurface cannot be made from a closed geometry."
            )

        # Constructors
        self.x_omp = None
        self.z_omp = None
        self.x_lfs_inter = None
        self.z_lfs_inter = None
        self.x_hfs_inter = None
        self.z_hfs_inter = None
        self.alpha_lfs = None
        self.alpha_hfs = None
        self.lfs_loop = None
        self.hfs_loop = None

    def split_LFS_HFS(self, plane, o_point):
        """
        Split the FluxSurface into low-field side and high-field side surfaces.

        Parameters
        ----------
        plane: Plane
            The x-y cutting plane
        o_point: O-point
            The magnetic centre of the plasma
        """
        intersections = loop_plane_intersect(self.loop, plane)
        x_inter = intersections.T[0]

        # Pick the first intersection, travelling from the o_point outwards
        deltas = x_inter - o_point.x
        arg_inter = np.argmax(deltas > 0)
        self.x_omp = x_inter[arg_inter]
        self.z_omp = o_point.z

        # Split the flux surface geometry into LFS and HFS geometries
        loop = self.loop
        delta = 1e-1 if o_point.x < self.x_omp else -1e-1
        radial_line = Loop(x=[o_point.x, self.x_omp + delta], z=[self.z_omp, self.z_omp])
        # Add the intersection point to the loop
        arg_inter = join_intersect(loop, radial_line, get_arg=True)[0]

        # Split the flux surface geometry
        loop1 = Loop.from_array(loop[: arg_inter + 1])
        loop2 = Loop.from_array(loop[arg_inter:])

        loop1 = self._reset_direction(loop1)
        loop2 = self._reset_direction(loop2)

        # Sort the segments into LFS (outboard) and HFS (inboard) geometries
        if loop1.z[1] > self.z_omp:
            self.lfs_loop = loop2
            self.hfs_loop = loop1
        else:
            self.lfs_loop = loop1
            self.hfs_loop = loop2

    def _reset_direction(self, loop):
        if loop.argmin([self.x_omp, self.z_omp]) != 0:
            loop.reverse()
        return loop

    def clip(self, first_wall):
        """
        Clip the LFS and HFS geometries to a first wall.
        """
        first_wall = first_wall.copy()
        (
            self.lfs_loop,
            self.x_lfs_inter,
            self.z_lfs_inter,
            self.alpha_lfs,
        ) = self._clipper(self.lfs_loop, first_wall)
        (
            self.hfs_loop,
            self.x_hfs_inter,
            self.z_hfs_inter,
            self.alpha_hfs,
        ) = self._clipper(self.hfs_loop, first_wall)

    def _clipper(self, loop, first_wall):
        args = join_intersect(loop, first_wall, get_arg=True)

        # Because we oriented the loop the "right" way, the first intersection
        # is at the smallest argument
        loop = Loop.from_array(loop[: min(args) + 1])
        loop = self._reset_direction(loop)
        x_inter = loop.x[-1]
        z_inter = loop.z[-1]

        fw_arg = first_wall.argmin([x_inter, z_inter])
        if check_linesegment(
            first_wall.d2.T[fw_arg],
            first_wall.d2.T[fw_arg + 1],
            np.array([x_inter, z_inter]),
        ):
            fw_arg = int(fw_arg + 1)
        else:
            fw_arg = int(fw_arg)

        # Relying on the fact that first wall is ccw, get the intersection angle
        alpha = get_angle_between_points(loop[-2], loop[-1], first_wall[fw_arg])

        return loop, x_inter, z_inter, alpha

    def plot(self, ax=None, **kwargs):
        if ax is None:
            ax = plt.gca()

        if "linewidth" not in kwargs:
            kwargs["linewidth"] = 0.01

        self.lfs_loop.plot(ax, color="b", **kwargs)
        self.hfs_loop.plot(ax, color="r", **kwargs)

    def copy(self):
        return deepcopy(self)


class ChargedParticleSolver:
    """
    A simplified charged particle transport model along open field lines.
    """

    # fmt: off
    default_params = [
        ["fw_p_sol_near", "near scrape-off layer power", 50, "MW", None, "Input"],
        ["fw_p_sol_far", "far scrape-off layer power", 50, "MW", None, "Input"],
        ["fw_lambda_q_near", "Lambda q near SOL", 0.05, "m", None, "Input"],
        ["fw_lambda_q_far", "Lambda q far SOL", 0.05, "m", None, "Input"],
        ["f_outer_target", "Power fraction", 0.75, "N/A", None, "Input"],
        ["f_inner_target", "Power fraction", 0.25, "N/A", None, "Input"],
        ["f_upper_target", "Power fraction", 0.5, "N/A", None, "Input"],
        ["f_lower_target", "Power fraction", 0.5, "N/A", None, "Input"],
    ]
    # fmt: on

    def __init__(self, config, equilibrium, **kwargs):
        self.params = ParameterFrame(self.default_params)
        self.params.update_kw_parameters(config)
        self._check_params()
        self.dpsi_near = kwargs.get("dpsi_near", 0.001)
        self.dpsi_far = kwargs.get("dpsi_far", 0.001)

        self.eq = equilibrium

        # Constructors
        self.first_wall = None
        self.flux_surfaces = None
        self.x_sep_omp = None
        self.x_sep_imp = None
        self.result = None

    def _check_params(self):
        if self.params.f_outer_target + self.params.f_inner_target != 1.0:
            raise AdvectionTransportError(
                "Inner / outer fractions should sum to 1.0:\n"
                f"{self.params.f_outer_target} + {self.params.f_inner_target} != 1.0:"
            )
        if self.params.f_upper_target + self.params.f_lower_target != 1.0:
            raise AdvectionTransportError(
                "Upper / lower fractions should sum to 1.0:\n"
                f"{self.params.f_upper_target} + {self.params.f_lower_target} != 1.0:"
            )

    @staticmethod
    def _process_first_wall(first_wall):
        """
        Force working first wall geometry to be closed and counter-clockwise.
        """
        first_wall = first_wall.copy()
        if not first_wall.closed:
            bluemira_warn("First wall should be a closed geometry. Closing it.")
            first_wall.close()

        if not first_wall.ccw:
            bluemira_warn(
                "First wall should be oriented counter-clockwise. Reversing it."
            )
            first_wall.reverse()
        return first_wall

    @staticmethod
    def _get_arrays(flux_surfaces):
        x_omp = np.array([fs.x_omp for fs in flux_surfaces])
        z_omp = np.array([fs.z_omp for fs in flux_surfaces])
        x_lfs_inter = np.array([fs.x_lfs_inter for fs in flux_surfaces])
        z_lfs_inter = np.array([fs.z_lfs_inter for fs in flux_surfaces])
        x_hfs_inter = np.array([fs.x_hfs_inter for fs in flux_surfaces])
        z_hfs_inter = np.array([fs.z_hfs_inter for fs in flux_surfaces])
        alpha_lfs = np.array([fs.alpha_lfs for fs in flux_surfaces])
        alpha_hfs = np.array([fs.alpha_hfs for fs in flux_surfaces])
        return (
            x_omp,
            z_omp,
            x_lfs_inter,
            z_lfs_inter,
            x_hfs_inter,
            z_hfs_inter,
            alpha_lfs,
            alpha_hfs,
        )

    def _get_xpoint_psi(self, x_points):
        if self.eq.is_double_null:
            return 0.5 * (x_points[0].psi + x_points[1].psi)
        else:
            return x_points[0].psi

    def _get_sep_out_intersection(self, o_point, outboard=True):
        """
        Find the middle and maximum outboard mid-plane psi norm values
        """
        yz_plane = Plane([0, 0, o_point.z], [1, 0, o_point.z], [1, 1, o_point.z])
        separatrix = self.eq.get_separatrix()

        if not isinstance(separatrix, Loop):
            sep1_intersections = loop_plane_intersect(separatrix[0], yz_plane)
            sep2_intersections = loop_plane_intersect(separatrix[1], yz_plane)
            sep1_arg = np.argmin(np.abs(sep1_intersections.T[0] - o_point.x))
            sep2_arg = np.argmin(np.abs(sep2_intersections.T[0] - o_point.x))
            x_sep1_mp = sep1_intersections.T[0][sep1_arg]
            x_sep2_mp = sep2_intersections.T[0][sep2_arg]
            if outboard:
                x_sep_mp = x_sep1_mp if x_sep1_mp > x_sep2_mp else x_sep2_mp
            else:
                x_sep_mp = x_sep1_mp if x_sep1_mp < x_sep2_mp else x_sep2_mp
        else:
            sep_intersections = loop_plane_intersect(separatrix, yz_plane)
            sep_arg = np.argmin(np.abs(sep_intersections.T[0] - o_point.x))
            x_sep_mp = sep_intersections.T[0][sep_arg]

        out_intersections = loop_plane_intersect(self.first_wall, yz_plane)
        if outboard:
            x_out_mp = np.max(out_intersections.T[0])
        else:
            x_out_mp = np.min(out_intersections.T[0])

        return x_sep_mp, x_out_mp

    @staticmethod
    def _sort_flux_surfaces(loop, x_mp, z_mp):
        return min(loop.distance_to([x_mp, z_mp]))

    def _get_flux_surface(self, psi_norm, x, z, o_points, x_points):
        """
        Get the flux surface at specified normalised psi, as close as possible to a point.
        """
        loops = find_flux_surfs(
            self.eq.x,
            self.eq.z,
            self.eq.psi(),
            psi_norm,
            o_points=o_points,
            x_points=x_points,
        )
        loops = [Loop(x=loop.T[0], z=loop.T[1]) for loop in loops]

        loop = sorted(loops, key=lambda loop: self._sort_flux_surfaces(loop, x, z))[0]
        return OpenFluxSurface(loop)

    def make_flux_surfaces_ob(self):
        """
        Make the flux surfaces along which the charged particle power is to be
        transported from the outboard
        """
        o_points, x_points = self.eq.get_OX_points()
        o_point = o_points[0]
        x_point_psi = self._get_xpoint_psi(x_points)

        self.x_sep_omp, x_out_omp = self._get_sep_out_intersection(
            o_point, outboard=True
        )

        yz_plane = Plane([0, 0, o_point.z], [1, 0, o_point.z], [1, 1, o_point.z])

        psi_out_omp = self.eq.psi(x_out_omp, o_point.z)
        psi_norm_out = float(calc_psi_norm(psi_out_omp, o_point.psi, x_point_psi))

        self.flux_surfaces = []

        if self.eq.is_double_null:
            psi_sep = self.eq.psi(self.x_sep_omp, o_point.z)
            psi_norm = calc_psi_norm(psi_sep, o_point.psi, x_point_psi) + self.dpsi_near
        else:
            psi_norm = 1.0

        while psi_norm < psi_norm_out:
            f_s = self._get_flux_surface(
                psi_norm, self.x_sep_omp, o_point.z, o_points, x_points
            )

            # Split the flux surface and set OMP values
            f_s.split_LFS_HFS(yz_plane, o_point)
            self.flux_surfaces.append(f_s)

            if f_s.x_omp - self.x_sep_omp < self.params.fw_lambda_q_near:
                psi_norm += self.dpsi_near
            else:
                psi_norm += self.dpsi_far

    def make_flux_surfaces_ib(self):
        """
        Make the flux surfaces along which the charged particle power is to be
        transported from the inboard
        """
        o_points, x_points = self.eq.get_OX_points()
        o_point = o_points[0]
        x_point_psi = self._get_xpoint_psi(x_points)

        # Find the middle and maximum outboard mid-plane psi norm values
        self.x_sep_imp, x_out_imp = self._get_sep_out_intersection(
            o_point, outboard=False
        )

        yz_plane = Plane([0, 0, o_point.z], [1, 0, o_point.z], [1, 1, o_point.z])

        psi_out_imp = self.eq.psi(x_out_imp, o_point.z)
        psi_norm_out_imp = float(calc_psi_norm(psi_out_imp, o_point.psi, x_point_psi))

        ib_flux_surfaces = []

        psi_sep = self.eq.psi(self.x_sep_imp, o_point.z)
        psi_norm = calc_psi_norm(psi_sep, o_point.psi, x_point_psi) + self.dpsi_near
        while psi_norm < psi_norm_out_imp:
            f_s = self._get_flux_surface(
                psi_norm, self.x_sep_imp, o_point.z, o_points, x_points
            )
            f_s.split_LFS_HFS(yz_plane, o_point)
            ib_flux_surfaces.append(f_s)

            if abs(f_s.x_omp - self.x_sep_imp) < self.params.fw_lambda_q_near:
                psi_norm += self.dpsi_near
            else:
                psi_norm += self.dpsi_far

        # Separate outboard and inboard flux surfaces
        self.flux_surfaces = [self.flux_surfaces, ib_flux_surfaces]

    def analyse(self, first_wall):
        """
        Perform the calculation to obtain charged particle heat fluxes on the
        the specified first_wall

        Parameters
        ----------
        first_wall: Loop
            The closed first wall geometry on which to calculate the heat flux

        Returns
        -------
        x: np.array
            The x coordinates of the flux surface intersections
        z: np.array
            The z coordinates of the flux surface intersections
        heat_flux: np.array
            The perpendicular heat fluxes at the intersection points [MW/m^2]
        """
        self.first_wall = self._process_first_wall(first_wall)

        if self.eq.is_double_null:
            x, z, hf = self._analyse_DN(first_wall)
        else:
            x, z, hf = self._analyse_SN(first_wall)

        self.result = x, z, hf
        return x, z, hf

    def _analyse_SN(self, first_wall):

        self.make_flux_surfaces_ob()

        # Find the intersections of the flux surfaces with the first wall
        for flux_surface in self.flux_surfaces:
            flux_surface.clip(first_wall)

        (
            x_omp,
            z_omp,
            x_lfs_inter,
            z_lfs_inter,
            x_hfs_inter,
            z_hfs_inter,
            alpha_lfs,
            alpha_hfs,
        ) = self._get_arrays(self.flux_surfaces)

        # Calculate values at OMP
        dx_omp = x_omp - self.x_sep_omp
        Bp_omp = self.eq.Bp(x_omp, z_omp)
        Bt_omp = self.eq.Bt(x_omp)
        B_omp = np.hypot(Bp_omp, Bt_omp)

        # Parallel heat flux at the outboard midplane
        p_sol_near = self.params.fw_p_sol_near
        p_sol_far = self.params.fw_p_sol_far
        lq_near = self.params.fw_lambda_q_near
        lq_far = self.params.fw_lambda_q_far
        q_par_omp = (
            (
                p_sol_near * np.exp(-dx_omp / lq_near) / lq_near
                + p_sol_far * np.exp(-dx_omp / lq_far) / lq_far
            )
            * B_omp
            / (Bp_omp * 2 * np.pi * x_omp)
        )

        # Calculate values at intersections
        Bp_lfs = self.eq.Bp(x_lfs_inter, z_lfs_inter)
        Bp_hfs = self.eq.Bp(x_hfs_inter, z_hfs_inter)

        # Flux expansion
        fx_lfs = x_omp * Bp_omp / (x_lfs_inter * Bp_lfs)
        fx_hfs = x_omp * Bp_omp / (x_hfs_inter * Bp_hfs)

        # Calculate parallel heat fluxes
        factor = q_par_omp * x_omp * Bp_omp / B_omp
        q_par_lfs = factor / (x_lfs_inter * fx_lfs)
        q_par_hfs = factor / (x_hfs_inter * fx_hfs)

        # Calculate perpendicular heat fluxes
        heat_flux_lfs = self.params.f_outer_target * q_par_lfs * np.sin(alpha_lfs)
        heat_flux_hfs = self.params.f_inner_target * q_par_hfs * np.sin(alpha_hfs)

        # Correct power (energy conservation)
        fs_widths = (x_omp - np.roll(x_omp, 1))[1:]
        # Add the first flux surface width (to the LCFS)
        fs_widths = np.append(x_omp[0] - self.x_sep_omp, fs_widths)
        q_omp_int = 2 * np.pi * np.sum(q_par_omp / (B_omp / Bp_omp) * fs_widths * x_omp)
        f_correct_power = (
            self.params.fw_p_sol_near + self.params.fw_p_sol_far
        ) / q_omp_int
        return (
            np.append(x_lfs_inter, x_hfs_inter),
            np.append(z_lfs_inter, z_hfs_inter),
            f_correct_power * np.append(heat_flux_lfs, heat_flux_hfs),
        )

    def _analyse_DN(self, first_wall):
        self.make_flux_surfaces_ob()
        self.make_flux_surfaces_ib()

        ob_flux_surfaces, ib_flux_surfaces = self.flux_surfaces

        # Find the intersections of the flux surfaces with the first wall
        for flux_surface in ib_flux_surfaces:
            flux_surface.clip(first_wall)
        for flux_surface in ob_flux_surfaces:
            flux_surface.clip(first_wall)

        (
            x_imp,
            z_imp,
            x_hfs_down_inter,
            z_hfs_down_inter,
            x_hfs_up_inter,
            z_hfs_up_inter,
            alpha_hfs_up,
            alpha_hfs_down,
        ) = self._get_arrays(ib_flux_surfaces)
        (
            x_omp,
            z_omp,
            x_lfs_down_inter,
            z_lfs_down_inter,
            x_lfs_up_inter,
            z_lfs_up_inter,
            alpha_lfs_up,
            alpha_lfs_down,
        ) = self._get_arrays(ob_flux_surfaces)

        # Calculate values at OMP
        dx_omp = x_omp - self.x_sep_omp
        Bp_omp = self.eq.Bp(x_omp, z_omp)
        Bt_omp = self.eq.Bt(x_omp)
        B_omp = np.hypot(Bp_omp, Bt_omp)

        # Calculate values at IMP
        dx_imp = abs(x_imp - self.x_sep_imp)
        Bp_imp = self.eq.Bp(x_imp, z_imp)
        Bt_imp = self.eq.Bt(x_imp)
        B_imp = np.hypot(Bp_imp, Bt_imp)

        # Parallel heat flux at the outboard and inboard midplane
        q_par_omp = self._q_par(x_omp, dx_omp, B_omp, Bp_omp)
        q_par_imp = self._q_par(x_imp, dx_imp, B_imp, Bp_imp)

        # Calculate poloidal field at intersections
        Bp_lfs_down = self.eq.Bp(x_lfs_down_inter, z_lfs_down_inter)
        Bp_lfs_up = self.eq.Bp(x_lfs_up_inter, z_lfs_up_inter)
        Bp_hfs_down = self.eq.Bp(x_hfs_down_inter, z_hfs_down_inter)
        Bp_hfs_up = self.eq.Bp(x_hfs_up_inter, z_hfs_up_inter)

        # Calculate parallel heat fluxes at the intersections
        # Note that flux expansion terms cancelate down to this
        q_par_lfs_down = q_par_omp * Bp_lfs_down / B_omp
        q_par_lfs_up = q_par_omp * Bp_lfs_up / B_omp
        q_par_hfs_down = q_par_imp * Bp_hfs_down / B_imp
        q_par_hfs_up = q_par_imp * Bp_hfs_up / B_imp

        # Calculate perpendicular heat fluxes
        heat_flux_lfs_down = (
            self.params.f_outer_target
            * self.params.f_lower_target
            * q_par_lfs_down
            * np.sin(alpha_lfs_down)
        )
        heat_flux_lfs_up = (
            self.params.f_outer_target
            * self.params.f_upper_target
            * q_par_lfs_up
            * np.sin(alpha_lfs_up)
        )
        heat_flux_hfs_down = (
            self.params.f_inner_target
            * self.params.f_lower_target
            * q_par_hfs_down
            * np.sin(alpha_hfs_down)
        )
        heat_flux_hfs_up = (
            self.params.f_inner_target
            * self.params.f_upper_target
            * q_par_hfs_up
            * np.sin(alpha_hfs_up)
        )

        # Correct power (energy conservation)
        fs_widths = (x_omp - np.roll(x_omp, 1))[1:]
        # Add the first flux surface width (to the LCFS)
        fs_widths = np.append(x_omp[0] - self.x_sep_omp, fs_widths)
        q_omp_int = 2 * np.pi * np.sum(q_par_omp / (B_omp / Bp_omp) * fs_widths * x_omp)

        # Correct power (energy conservation)
        fs_widths = np.abs((x_imp - np.roll(x_imp, 1))[1:])
        # Add the first flux surface width (to the LCFS)
        fs_widths = np.append(abs(x_imp[0] - self.x_sep_imp), fs_widths)
        q_imp_int = 2 * np.pi * np.sum(q_par_imp / (B_imp / Bp_imp) * fs_widths * x_imp)

        total_power = self.params.fw_p_sol_near + self.params.fw_p_sol_far
        f_correct_power_ob = (self.params.f_outer_target * total_power) / q_omp_int

        f_correct_power_ib = (self.params.f_inner_target * total_power) / q_imp_int
        print(q_omp_int, total_power * self.params.f_outer_target)
        print(q_imp_int, total_power * self.params.f_inner_target)

        return (
            np.concatenate(
                [x_lfs_down_inter, x_lfs_up_inter, x_hfs_down_inter, x_hfs_up_inter]
            ),
            np.concatenate(
                [z_lfs_down_inter, z_lfs_up_inter, z_hfs_down_inter, z_hfs_up_inter]
            ),
            np.concatenate(
                [
                    f_correct_power_ob * self.params.f_lower_target * heat_flux_lfs_down,
                    f_correct_power_ob * self.params.f_upper_target * heat_flux_lfs_up,
                    f_correct_power_ib * self.params.f_lower_target * heat_flux_hfs_down,
                    f_correct_power_ib * self.params.f_upper_target * heat_flux_hfs_up,
                ]
            ),
        )

    def _q_par(self, x, dx, B, Bp):
        p_sol_near = self.params.fw_p_sol_near
        p_sol_far = self.params.fw_p_sol_far
        lq_near = self.params.fw_lambda_q_near
        lq_far = self.params.fw_lambda_q_far
        return (
            (
                p_sol_near * np.exp(-dx / lq_near) / lq_near
                + p_sol_far * np.exp(-dx / lq_far) / lq_far
            )
            * B
            / (Bp * 2 * np.pi * x)
        )

    def plot(self, ax=None):
        """
        Plot the ChargedParticleSolver results.
        """

        if ax is None:
            ax = plt.gca()

        self.first_wall.plot(ax, linewidth=0.1, fill=False)
        separatrix = self.eq.get_separatrix()

        if isinstance(separatrix, Loop):
            separatrix = [separatrix]

        for sep in separatrix:
            sep.plot(ax, linewidth=0.12)

        if isinstance(self.flux_surfaces[0], list):
            flux_surfaces = self.flux_surfaces[0]
            flux_surfaces.extend(self.flux_surfaces[1])
        else:
            flux_surfaces = self.flux_surfaces

        for f_s in flux_surfaces:
            f_s.plot(ax, linewidth=0.01)

        cm = ax.scatter(self.result[0], self.result[1], c=self.result[2], s=3, zorder=40)
        f = plt.gcf()
        f.colorbar(cm, label="MW/m^2")
