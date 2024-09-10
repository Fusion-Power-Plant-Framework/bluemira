# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
A simplified 2-D solver for calculating charged particle heat loads.
"""

from copy import deepcopy
from dataclasses import dataclass, fields

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

import bluemira.radiation_transport.flux_surfaces_maker as fsm
from bluemira.base.constants import EPS
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.display.plotter import Zorder, plot_coordinates
from bluemira.geometry.coordinates import Coordinates, coords_plane_intersect
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import make_polygon
from bluemira.radiation_transport.error import AdvectionTransportError
from bluemira.radiation_transport.flux_surfaces_maker import _clip_flux_surfaces

__all__ = ["ChargedParticleSolver"]


class ChargedParticleSolver:
    """
    A simplified charged particle transport model along open field lines.

    Parameters
    ----------
    config: Dict[str, float]
        The parameters for running the transport model. See
        :py:class:`~bluemira.radiation_transport.advective_transport.ChargedParticleSolverParams`
        for available parameters and their defaults.
    equilibrium: Equilibrium
        The equilibrium defining flux surfaces.
    dx_mp: float (optional)
        The midplane spatial resolution between flux surfaces [m]
        (default: 0.001).
    """

    def __init__(
        self,
        config: dict[str, float],
        equilibrium,
        dx_mp: float = 0.001,
        psi_n_tol: float = 1e-6,
    ):
        self.eq = equilibrium
        self.params = self._make_params(config)
        self._check_params()
        self.dx_mp = dx_mp

        # Constructors
        self.first_wall = None
        self.flux_surfaces_ob_down = None
        self.flux_surfaces_ob_up = None
        self.flux_surfaces_ib_down = None
        self.flux_surfaces_ib_up = None
        self.x_sep_omp = None
        self.x_sep_imp = None
        self.result = None

        # Pre-processing
        self.psi_n_tol = psi_n_tol
        o_points, _ = self.eq.get_OX_points()
        self._o_point = o_points[0]
        z = self._o_point.z
        self._yz_plane = BluemiraPlane.from_3_points([0, 0, z], [1, 0, z], [1, 1, z])

    @property
    def flux_surfaces(self):
        """
        All flux surfaces in the ChargedParticleSolver.

        Returns
        -------
        flux_surfaces: List[PartialOpenFluxSurface]
        """
        flux_surfaces = []
        for group in [
            self.flux_surfaces_ob_down,
            self.flux_surfaces_ob_up,
            self.flux_surfaces_ib_down,
            self.flux_surfaces_ib_up,
        ]:
            if group:
                flux_surfaces.extend(group)
        return flux_surfaces

    def _check_params(self):
        """
        Check input fractions for validity.

        Raises
        ------
        AdvectionTransportError
            Sum of total power fractions is not ~= 1
        """
        # Check lower power fractions
        lower_power = self.params.f_lfs_lower_target + self.params.f_hfs_lower_target
        upper_power = self.params.f_lfs_upper_target + self.params.f_hfs_upper_target
        power_sum = lower_power + upper_power

        if not np.isclose(power_sum, 1.0, rtol=EPS, atol=1e-9):
            raise AdvectionTransportError(
                f"Total power fractions should sum to 1, not : {power_sum}"
            )

        zero_in_one_direction = np.any(
            np.isclose([lower_power, upper_power], 0, rtol=EPS, atol=1e-9)
        )
        if self.eq.is_double_null and zero_in_one_direction:
            bluemira_warn(
                "A DN equilibrium was detected but your power distribution"
                " is 0 in either the lower or upper directions."
            )
        elif not self.eq.is_double_null and not zero_in_one_direction:
            bluemira_warn(
                "A SN equilibrium was detected but you power distribution is not 0"
                " in either the lower or upper directions."
            )

    def _process_first_wall(self, first_wall):
        """
        Force working first wall geometry to be closed and counter-clockwise.
        """
        first_wall = deepcopy(first_wall)

        if not first_wall.check_ccw(axis=[0, 1, 0]):
            bluemira_warn(
                "First wall should be oriented counter-clockwise. Reversing it."
            )
            first_wall.reverse()

        if not first_wall.closed:
            bluemira_warn("First wall should be a closed geometry. Closing it.")
            first_wall.close()

        int_intersection = coords_plane_intersect(first_wall, self._yz_plane)[0]
        out_intersection = coords_plane_intersect(first_wall, self._yz_plane)[1]

        return first_wall, int_intersection, out_intersection

    @staticmethod
    def _get_arrays(flux_surfaces):
        """
        Get arrays of flux surface values.
        """
        x_mp = np.array([fs.x_start for fs in flux_surfaces])
        z_mp = np.array([fs.z_start for fs in flux_surfaces])
        x_fw = np.array([fs.x_end for fs in flux_surfaces])
        z_fw = np.array([fs.z_end for fs in flux_surfaces])
        alpha = np.array([fs.alpha for fs in flux_surfaces])
        return x_mp, z_mp, x_fw, z_fw, alpha

    def _make_flux_surfaces_ob(self):
        """
        Make the flux surfaces on the outboard.
        """
        self.x_sep_omp, x_out_omp = fsm._get_sep_out_intersection(
            self.eq,
            self.first_wall,
            self._yz_plane,
            outboard=True,
        )

        self.flux_surfaces_ob_down, self.flux_surfaces_ob_up = (
            fsm._make_flux_surfaces_ibob(
                self.dx_mp,
                self.eq,
                self._o_point,
                self._yz_plane,
                self.x_sep_omp,
                x_out_omp,
                outboard=True,
            )
        )

    def _make_flux_surfaces_ib(self):
        """
        Make the flux surfaces on the inboard.
        """
        self.x_sep_imp, x_out_imp = fsm._get_sep_out_intersection(
            self.eq,
            self.first_wall,
            self._yz_plane,
            outboard=False,
        )

        self.flux_surfaces_ib_down, self.flux_surfaces_ib_up = (
            fsm._make_flux_surfaces_ibob(
                self.dx_mp,
                self.eq,
                self._o_point,
                self._yz_plane,
                self.x_sep_imp,
                x_out_imp,
                outboard=False,
            )
        )

    def _clip_flux_surfaces(self, first_wall):
        """
        Clip the flux surfaces to a first wall. Catch the cases where no intersections
        are found.
        """
        _clip_flux_surfaces(
            first_wall,
            [
                self.flux_surfaces_ob_down,
                self.flux_surfaces_ob_up,
                self.flux_surfaces_ib_down,
                self.flux_surfaces_ib_up,
            ],
        )

    def _no_wall_intersection_region(
        self, x_up_inter, z_up_inter, x_down_inter, z_down_inter, *, lfs=True
    ):
        """
        Get first wall mid-plane region with no flux line inetrsections.
        """
        up_end_i = self.first_wall.argmin(np.array([x_up_inter[-1], 0, z_up_inter[-1]]))
        down_end_i = self.first_wall.argmin(
            np.array([x_down_inter[-1], 0, z_down_inter[-1]])
        )

        reg_i = np.nonzero(
            (self.first_wall.z < self.first_wall.z[up_end_i])
            & (self.first_wall.z >= self.first_wall.z[down_end_i])
            & (
                (self.first_wall.x > self._o_point.x)
                if lfs
                else (self.first_wall.x < self._o_point.x)
            )
        )[0]

        x_reg_inter = self.first_wall.x[reg_i]
        z_reg_inter = self.first_wall.z[reg_i]

        reg_wire = make_polygon(self.first_wall.T[reg_i])
        wire_length = reg_wire.length

        return x_reg_inter, z_reg_inter, wire_length

    def analyse(self, first_wall: Coordinates):
        """
        Perform the calculation to obtain charged particle heat fluxes on the
        the specified first_wall

        Parameters
        ----------
        first_wall: Coordinates
            The closed first wall geometry on which to calculate the heat flux

        Returns
        -------
        x: np.array
            The x coordinates of the flux surface intersections
        z: np.array
            The z coordinates of the flux surface intersections
        heat_flux: np.array
            The perpendicular heat fluxes at the intersection points [MW/m^2]

        Notes
        -----
        The heat flux model assumes pure parallel transport and fudges
        the perpendicular transport via the power decay length, lambda.
        This approach, while is widely used, leads to no power deposited
        on the wall at the mid-plane.
        _analyse_SN and _analyse_DN assume, in the area in proximity of
        the mid-plane, where the outermost flux tube is open, that the
        remaining power of from the exponential decay is deposited on
        the wall perpendicularly.
        """
        (
            self.first_wall,
            self.imp_int,
            self.omp_int,
        ) = self._process_first_wall(first_wall)

        if self.eq.is_double_null:
            x, z, hf = self._analyse_DN()
        else:
            x, z, hf = self._analyse_SN()

        self.result = x, z, hf
        return x, z, hf

    def _analyse_SN(self):
        """
        Calculation for the case of single nulls.

        """
        self._make_flux_surfaces_ob()

        # Find the intersections of the flux surfaces with the first wall
        self._clip_flux_surfaces(self.first_wall)

        x_omp, z_omp, x_lfs_inter, z_lfs_inter, alpha_lfs = self._get_arrays(
            self.flux_surfaces_ob_down
        )
        _, _, x_hfs_inter, z_hfs_inter, alpha_hfs = self._get_arrays(
            self.flux_surfaces_ob_up
        )

        # Calculate values at OMP
        dx_omp = x_omp - self.x_sep_omp
        Bp_omp = self.eq.Bp(x_omp, z_omp)
        Bt_omp = self.eq.Bt(x_omp)
        B_omp = np.hypot(Bp_omp, Bt_omp)

        # Parallel power at the outboard midplane
        q_par_omp = self._q_par(x_omp, dx_omp, B_omp, Bp_omp)

        # Calculate values at intersections
        Bp_lfs = self.eq.Bp(x_lfs_inter, z_lfs_inter)
        Bp_hfs = self.eq.Bp(x_hfs_inter, z_hfs_inter)

        # Calculate parallel power at the intersections
        # Note that flux expansion terms cancel down to this
        q_par_lfs = q_par_omp * Bp_lfs / B_omp
        q_par_hfs = q_par_omp * Bp_hfs / B_omp

        # Calculate perpendicular heat fluxes
        heat_flux_lfs = self.params.f_lfs_lower_target * q_par_lfs * np.sin(alpha_lfs)
        heat_flux_hfs = self.params.f_hfs_lower_target * q_par_hfs * np.sin(alpha_hfs)

        # Find FW portion for perpendicular power
        if self.first_wall.argmin(
            np.array([x_hfs_inter[-1], 0, z_hfs_inter[-1]])
        ) != self.first_wall.argmin(np.array([x_lfs_inter[-1], 0, z_lfs_inter[-1]])):
            x_out_inter, z_out_inter, outb_length = self._no_wall_intersection_region(
                x_hfs_inter, z_hfs_inter, x_lfs_inter, z_lfs_inter, lfs=True
            )
        else:
            mid_i = self.first_wall.argmin(
                np.array([x_lfs_inter[-1], 0, z_lfs_inter[-1]])
            )
            x_out_inter = self.first_wall.x[mid_i]
            z_out_inter = self.first_wall.z[mid_i]
            outb_length = 1

        # Calculating missing power from parallel transport
        q_omp_int = 2 * np.pi * np.sum(q_par_omp * Bp_omp / B_omp * self.dx_mp * x_omp)
        miss_omp = self.params.P_sep_particle - q_omp_int
        outb_surf = outb_length * 2 * np.pi * self.omp_int[0]

        # Calculating mid-outboard and mid-inboard heat flux
        heat_flux_x_outb = miss_omp / outb_surf
        if outb_length != 1:
            heat_flux_x_outb = [heat_flux_x_outb] * len(x_out_inter)

        return (
            np.concatenate([np.atleast_1d(x_out_inter), x_lfs_inter, x_hfs_inter]),
            np.concatenate([np.atleast_1d(z_out_inter), z_lfs_inter, z_hfs_inter]),
            np.concatenate([
                np.atleast_1d(heat_flux_x_outb),
                heat_flux_lfs,
                heat_flux_hfs,
            ]),
        )

    def _analyse_DN(self):  # noqa: PLR0914
        """
        Calculation for the case of double nulls.
        """
        self._make_flux_surfaces_ob()
        self._make_flux_surfaces_ib()

        # Find the intersections of the flux surfaces with the first wall
        self._clip_flux_surfaces(self.first_wall)

        (
            x_omp,
            z_omp,
            x_lfs_down_inter,
            z_lfs_down_inter,
            alpha_lfs_down,
        ) = self._get_arrays(self.flux_surfaces_ob_down)
        _, _, x_lfs_up_inter, z_lfs_up_inter, alpha_lfs_up = self._get_arrays(
            self.flux_surfaces_ob_up
        )
        (
            x_imp,
            z_imp,
            x_hfs_down_inter,
            z_hfs_down_inter,
            alpha_hfs_down,
        ) = self._get_arrays(self.flux_surfaces_ib_down)
        _, _, x_hfs_up_inter, z_hfs_up_inter, alpha_hfs_up = self._get_arrays(
            self.flux_surfaces_ib_up
        )

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

        # Parallel power set-up at the outboard and inboard midplane
        # Note that the power is not split yet into hfs and lfs rates
        q_par_omp = self._q_par(x_omp, dx_omp, B_omp, Bp_omp)
        q_par_imp = self._q_par(x_imp, dx_imp, B_imp, Bp_imp, outboard=False)

        # Calculate poloidal field at intersections
        Bp_lfs_down = self.eq.Bp(x_lfs_down_inter, z_lfs_down_inter)
        Bp_lfs_up = self.eq.Bp(x_lfs_up_inter, z_lfs_up_inter)
        Bp_hfs_down = self.eq.Bp(x_hfs_down_inter, z_hfs_down_inter)
        Bp_hfs_up = self.eq.Bp(x_hfs_up_inter, z_hfs_up_inter)

        # Calculate parallel power at the intersections
        # Each q_par_* stores full P_sep_particle
        # Note that flux expansion terms cancel down to this
        q_par_lfs_down = q_par_omp * Bp_lfs_down / B_omp
        q_par_lfs_up = q_par_omp * Bp_lfs_up / B_omp
        q_par_hfs_down = q_par_imp * Bp_hfs_down / B_imp
        q_par_hfs_up = q_par_imp * Bp_hfs_up / B_imp

        # Calculate perpendicular heat fluxes
        # Here P_sep_particle actually gets distributed over the four targets
        heat_flux_lfs_down = (
            self.params.f_lfs_lower_target * q_par_lfs_down * np.sin(alpha_lfs_down)
        )
        heat_flux_lfs_up = (
            self.params.f_lfs_upper_target * q_par_lfs_up * np.sin(alpha_lfs_up)
        )
        heat_flux_hfs_down = (
            self.params.f_hfs_lower_target * q_par_hfs_down * np.sin(alpha_hfs_down)
        )
        heat_flux_hfs_up = (
            self.params.f_hfs_upper_target * q_par_hfs_up * np.sin(alpha_hfs_up)
        )

        # Find FW portion for perpendicular power
        x_out_inter, z_out_inter, outb_length = self._no_wall_intersection_region(
            x_lfs_up_inter, z_lfs_up_inter, x_lfs_down_inter, z_lfs_down_inter, lfs=True
        )
        x_in_inter, z_in_inter, inb_length = self._no_wall_intersection_region(
            x_hfs_up_inter, z_hfs_up_inter, x_hfs_down_inter, z_hfs_down_inter, lfs=False
        )
        # Calculating missing power from parallel transport
        q_omp_int = (
            2
            * np.pi
            * np.sum(q_par_omp * Bp_omp / B_omp * self.dx_mp * x_omp)
            * (self.params.f_lfs_lower_target + self.params.f_lfs_upper_target)
        )
        q_imp_int = (
            2
            * np.pi
            * np.sum(q_par_imp * Bp_imp / B_imp * self.dx_mp * x_imp)
            * (self.params.f_hfs_lower_target + self.params.f_hfs_upper_target)
        )
        miss_omp = (
            self.params.P_sep_particle
            * (self.params.f_lfs_lower_target + self.params.f_lfs_upper_target)
        ) - q_omp_int
        miss_imp = (
            self.params.P_sep_particle
            * (self.params.f_hfs_lower_target + self.params.f_hfs_upper_target)
        ) - q_imp_int
        outb_surf = outb_length * 2 * np.pi * self.omp_int[0]
        inb_surf = inb_length * 2 * np.pi * self.imp_int[0]

        # Calculating mid-outboard and mid-inboard heat flux
        heat_flux_x_outb = [miss_omp / outb_surf] * len(x_out_inter)
        heat_flux_x_inb = [miss_imp / inb_surf] * len(x_in_inter)

        return (
            np.concatenate([
                x_out_inter,
                x_lfs_down_inter,
                x_lfs_up_inter,
                x_in_inter,
                x_hfs_down_inter,
                x_hfs_up_inter,
            ]),
            np.concatenate([
                z_out_inter,
                z_lfs_down_inter,
                z_lfs_up_inter,
                z_in_inter,
                z_hfs_down_inter,
                z_hfs_up_inter,
            ]),
            np.concatenate([
                heat_flux_x_outb,
                heat_flux_lfs_down,
                heat_flux_lfs_up,
                heat_flux_x_inb,
                heat_flux_hfs_down,
                heat_flux_hfs_up,
            ]),
        )

    def _q_par(self, x, dx, B, Bp, *, outboard=True):
        """
        Calculate the parallel power at the midplane.
        """
        p_sol_near = self.params.P_sep_particle * self.params.f_p_sol_near
        p_sol_far = self.params.P_sep_particle * (1 - self.params.f_p_sol_near)
        if outboard:
            lq_near = self.params.fw_lambda_q_near_omp
            lq_far = self.params.fw_lambda_q_far_omp
        else:
            lq_near = self.params.fw_lambda_q_near_imp
            lq_far = self.params.fw_lambda_q_far_imp
        return (
            (
                p_sol_near * np.exp(-dx / lq_near) / lq_near
                + p_sol_far * np.exp(-dx / lq_far) / lq_far
            )
            * B
            / (Bp * 2 * np.pi * x)
        )

    def plot(self, ax: Axes = None, *, show=False) -> Axes:
        """
        Plot the ChargedParticleSolver results.
        """
        if ax is None:
            _, ax = plt.subplots()

        plot_coordinates(self.first_wall, ax=ax, linewidth=0.5, fill=False)
        separatrix = self.eq.get_separatrix()

        if isinstance(separatrix, Coordinates):
            separatrix = [separatrix]

        for sep in separatrix:
            plot_coordinates(sep, ax=ax, linewidth=0.2)

        for f_s in self.flux_surfaces:
            plot_coordinates(f_s.coords, ax=ax, linewidth=0.01)

        cm = ax.scatter(
            self.result[0],
            self.result[1],
            c=self.result[2],
            s=10,
            zorder=Zorder.RADIATION.value,
            cmap="plasma",
        )
        f = ax.figure
        f.colorbar(cm, label="MW/m²")
        if show:
            plt.show()
        return ax

    @staticmethod
    def _make_params(config):
        """Convert the given params to ``ChargedParticleSolverParams``

        Raises
        ------
        TypeError
            Unsupported config type
        ValueError
            Unknown configuration parameters
        """
        if isinstance(config, dict):
            try:
                return ChargedParticleSolverParams(**config)
            except TypeError:
                unknown = [
                    k for k in config if k not in fields(ChargedParticleSolverParams)
                ]
                raise ValueError(
                    f"Unknown config parameter(s) {str(unknown)[1:-1]}"
                ) from None
        elif isinstance(config, ChargedParticleSolverParams):
            return config
        else:
            raise TypeError(
                "Unsupported type: 'config' must be a 'dict', or "
                "'ChargedParticleSolverParams' instance; found "
                f"'{type(config).__name__}'."
            )


@dataclass
class ChargedParticleSolverParams:
    P_sep_particle: float = 150
    """Separatrix power [MW]."""

    f_p_sol_near: float = 0.5
    """Near scrape-off layer power rate [dimensionless]."""

    fw_lambda_q_near_omp: float = 0.003
    """Lambda q near SOL at the outboard [m]."""

    fw_lambda_q_far_omp: float = 0.05
    """Lambda q far SOL at the outboard [m]."""

    fw_lambda_q_near_imp: float = 0.003
    """Lambda q near SOL at the inboard [m]."""

    fw_lambda_q_far_imp: float = 0.05
    """Lambda q far SOL at the inboard [m]."""

    f_lfs_lower_target: float = 0.9
    """Fraction of SOL power deposited on the LFS lower target [dimensionless]."""

    f_hfs_lower_target: float = 0.1
    """Fraction of SOL power deposited on the HFS lower target [dimensionless]."""

    f_lfs_upper_target: float = 0
    """
    Fraction of SOL power deposited on the LFS upper target (DN only)
    [dimensionless].
    """

    f_hfs_upper_target: float = 0
    """
    Fraction of SOL power deposited on the HFS upper target (DN only)
    [dimensionless].
    """
