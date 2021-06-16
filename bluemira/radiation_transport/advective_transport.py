# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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

import numpy as np
from copy import deepcopy
from bluemira.base.parameter import ParameterFrame
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.geometry.base import Plane
from bluemira.geometry.tools import (
    loop_plane_intersect,
    join_intersect,
    get_angle_between_points,
    check_linesegment,
)
from bluemira.geometry.loop import Loop
from BLUEPRINT.equilibria.find import get_psi_norm


__all__ = ["ChargedParticleSolver"]


class FluxSurface:
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

        radial_line = Loop(x=[o_point.x, self.x_omp + 1e-3], z=[self.z_omp, self.z_omp])
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
        alpha = np.deg2rad(
            get_angle_between_points(loop[-2], loop[-1], first_wall[fw_arg])
        )

        return loop, x_inter, z_inter, alpha

    def copy(self):
        return deepcopy(self)


class ChargedParticleSolver:
    """
    A simplified charged particle transport model along open field lines.
    """

    # fmt: off
    default_params = [
        ["plasma_type", "Type of plasma", "SN", "N/A", None, "Input"],
        ["fw_p_sol_near", "near scrape-off layer power", 50, "MW", None, "Input"],
        ["fw_p_sol_far", "far scrape-off layer power", 50, "MW", None, "Input"],
        ["fw_lambda_q_near", "Lambda q near SOL", 0.05, "m", None, "Input"],
        ["fw_lambda_q_far", "Lambda q far SOL", 0.05, "m", None, "Input"],
        ["f_outer_target", "Power fraction", 0.75, "N/A", None, "Input"],
        ["f_inner_target", "Power fraction", 0.25, "N/A", None, "Input"],
    ]
    # fmt: on

    def __init__(self, config, equilibrium, **kwargs):
        self.params = ParameterFrame(self.default_params)
        self.params.update_kw_parameters(config)

        self.eq = equilibrium

        self.dpsi_near = kwargs.get("dpsi_near", 0.001)
        self.dpsi_far = kwargs.get("dpsi_far", 0.001)

        # Constructors
        self.first_wall = None
        self.flux_surfaces = None
        self.x_sep_omp = None

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

        self.make_flux_surfaces()

        # Find the intersections of the flux surfaces with the first wall
        for flux_surface in self.flux_surfaces:
            flux_surface.clip(first_wall)

        x_omp = np.array([fs.x_omp for fs in self.flux_surfaces])
        z_omp = np.array([fs.z_omp for fs in self.flux_surfaces])
        x_lfs_inter = np.array([fs.x_lfs_inter for fs in self.flux_surfaces])
        z_lfs_inter = np.array([fs.z_lfs_inter for fs in self.flux_surfaces])
        x_hfs_inter = np.array([fs.x_hfs_inter for fs in self.flux_surfaces])
        z_hfs_inter = np.array([fs.z_hfs_inter for fs in self.flux_surfaces])
        alpha_lfs = np.array([fs.alpha_lfs for fs in self.flux_surfaces])
        alpha_hfs = np.array([fs.alpha_hfs for fs in self.flux_surfaces])

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

    def make_flux_surfaces(self):
        """
        Make the flux surfaces along which the charged particle power is to be
        transported.
        """
        o_points, x_points = self.eq.get_OX_points()
        o_point, x_point = o_points[0], x_points[0]

        # Find the middle and maximum outboard mid-plane psi norm values
        yz_plane = Plane([0, 0, o_point.z], [1, 0, o_point.z], [1, 1, o_point.z])
        sep_intersections = loop_plane_intersect(self.eq.get_LCFS(), yz_plane)
        out_intersections = loop_plane_intersect(self.first_wall, yz_plane)
        self.x_sep_omp = np.max(sep_intersections.T[0])
        x_out_omp = np.max(out_intersections.T[0])

        psi_out_omp = self.eq.psi(x_out_omp, 0)
        psi_norm_out = float(get_psi_norm(psi_out_omp, o_point.psi, x_point.psi))

        self.flux_surfaces = []
        psi_norm = 1.0
        while psi_norm < psi_norm_out:
            f_s = FluxSurface(
                self.eq.get_flux_surface(psi_norm, o_points=o_points, x_points=x_points)
            )
            # Split the flux surface and set OMP values
            f_s.split_LFS_HFS(yz_plane, o_points[0])
            self.flux_surfaces.append(f_s)

            if f_s.x_omp - self.x_sep_omp < self.params.fw_lambda_q_near:
                psi_norm += self.dpsi_near
            else:
                psi_norm += self.dpsi_far
