# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
TF coil case class
"""

import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

from bluemira.base.logs import logger_setup
from bluemira.base.look_and_feel import bluemira_debug, bluemira_print, bluemira_warn
from bluemira.magnets.conductor import Conductor
from bluemira.magnets.registry import RegistrableMeta
from bluemira.magnets.utils import parall_k, serie_k
from bluemira.magnets.winding_pack import WindingPack
from bluemira.materials.material import Material

logger = logger_setup()


# ------------------------------------------------------------------------------
# Global Registries
# ------------------------------------------------------------------------------
CABLE_REGISTRY = {}

# ------------------------------------------------------------------------------
# CaseTF Class
# ------------------------------------------------------------------------------


class CaseTF(metaclass=RegistrableMeta):
    """TF case class"""

    _registry_ = CABLE_REGISTRY
    _name_in_registry_ = "CaseTF"

    def __init__(
        self,
        Ri: float,  # noqa: N803
        dy_ps: float,
        dy_vault: float,
        theta_TF: float,
        mat_case: Material,
        WPs: list[WindingPack],  # noqa: N803
        name: str = "CaseTF",
    ):
        """
        Case structure for TF coils

        Parameters
        ----------
        Ri:
            external radius of the coil
        dy_ps:
            radial thickness of the case cap
        dy_vault:
            radial thickness of the vault
        theta_TF:
            toroidal angle of a TF coil
        mat_case:
            material of the case
        WPs:
            list of winding packs associated with the case
        name:
            string identifier
        """
        self.name = name
        self.dy_ps = dy_ps
        self.dy_vault = dy_vault
        self.theta_TF = theta_TF
        self._rad_theta_TF = np.radians(theta_TF)
        self.Ri = Ri
        self.mat_case = mat_case
        self.WPs = WPs

    @property
    def dx_i(self):
        """Toroidal length of the coil case at its maximum radial position [m]"""
        return 2 * self.Ri * np.tan(self._rad_theta_TF / 2)

    @property
    def dx_ps(self):
        """Average toroidal length of the ps plate [m]"""
        return (self.Ri + (self.Ri - self.dy_ps)) * np.tan(self._rad_theta_TF / 2)

    @property
    def n_conductors(self):
        """Total number of conductors in the winding pack."""
        return sum(w.n_conductors for w in self.WPs)

    def max_Iop(self, B, T, T_margin):  # noqa: N803, N802
        """
        Compute the maximum operational current of the TF coil.

        Parameters
        ----------
        B : float
            Magnetic field intensity [T].
        T : float
            Operating temperature [K].
        T_margin : float
            Temperature margin [K].

        Returns
        -------
        float
            Maximum operational current [A], equal to the critical current of the
            superconducting strand.
        """
        return self.WPs[0].conductor.cable.sc_strand.Ic(B=B, T=T, T_margin=T_margin)

    @property
    def R_wp_i(self):  # noqa: N802
        """
        Compute the maximum radial positions for each winding pack.

        Returns
        -------
        np.ndarray
            Array of radial positions [m] corresponding to the inner edge of
            each winding pack.
        """
        dy_wp_cumsum = np.cumsum(np.array([0] + [w.dy for w in self.WPs]))
        return np.array([self.Ri - self.dy_ps - y for y in dy_wp_cumsum[0:-1]])

    @property
    def R_wp_k(self):  # noqa: N802
        """
        Compute the minimum radial positions for each winding pack.

        Returns
        -------
        np.ndarray
            Array of radial positions [m] corresponding to the outer edge of
            each winding pack.
        """
        return self.R_wp_i - np.array([w.dy for w in self.WPs])

    @property
    def Rk(self):  # noqa: N802
        """
        Minimum radial position of the TF case structure.

        Returns
        -------
        float
            Minimum radial position [m], located at the bottom of the vault.
        """
        return self.R_wp_k[-1] - self.dy_vault

    @property
    def dx_k(self):
        """
        Toroidal length of the case at its minimum radial position.

        Returns
        -------
        float
            Length in the toroidal direction at `Rk` [m].
        """
        return 2 * self.Rk * np.tan(self._rad_theta_TF / 2)

    @property
    def dx_vault(self):
        """
        Average toroidal length of the vault.

        Returns
        -------
        float
            Average length of the vault in the toroidal direction [m].
        """
        return (self.R_wp_k[-1] + self.Rk) * np.tan(self._rad_theta_TF / 2)

    @property
    def area(self):
        """
        Total cross-sectional area of the case including winding packs.

        Returns
        -------
        float
            Total area of the case [m²].
        """
        return (self.dx_i + self.dx_k) * (self.Ri - self.Rk) / 2

    @property
    def area_case_jacket(self):
        """
        Area of the case jacket (excluding winding pack regions).

        Returns
        -------
        float
            Case jacket area [m²], computed as total area minus total WP area.
        """
        total_wp_area = np.sum([w.conductor.area * w.nx * w.ny for w in self.WPs])
        return self.area - total_wp_area

    @property
    def area_wps(self):
        """
        Total area occupied by all winding packs.

        Returns
        -------
        float
            Combined area of the winding packs [m²].
        """
        return np.sum([w.area for w in self.WPs])

    @property
    def area_wps_jacket(self):
        """
        Total jacket area of all winding packs.

        Returns
        -------
        float
            Combined area of conductor jackets in all WPs [m²].
        """
        return np.sum([w.conductor.area_jacket * w.nx * w.ny for w in self.WPs])

    def Kx_ps(self, **kwargs):  # noqa: N802
        """
        Compute the equivalent radial stiffness of the poloidal support (PS) region.

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments passed to the material's Young's modulus
            function.

        Returns
        -------
        float
            Equivalent radial stiffness of the poloidal support [Pa].
        """
        return self.mat_case.E(**kwargs) * self.dy_ps / self.dx_ps

    def Kx_lat(self, **kwargs):  # noqa: N802
        """
        Compute the equivalent radial stiffness of the lateral case sections.

        These are the mechanical links between each winding pack and the outer case.
        Each lateral segment is approximated as a rectangular element.

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments passed to the material's Young's modulus function.

        Returns
        -------
        np.ndarray
            Array of radial stiffness values for each lateral segment [Pa].
        """
        dx_lat = np.array([
            (self.R_wp_i[i] + self.R_wp_k[i]) / 2 * np.tan(self._rad_theta_TF / 2)
            - w.dx / 2
            for i, w in enumerate(self.WPs)
        ])
        dy_lat = np.array([w.dy for w in self.WPs])
        return self.mat_case.E(**kwargs) * dy_lat / dx_lat

    def Kx_vault(self, **kwargs):  # noqa: N802
        """
        Compute the equivalent radial stiffness of the vault region.

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments passed to the material's Young's modulus function.

        Returns
        -------
        float
            Equivalent radial stiffness of the vault [Pa].
        """
        return self.mat_case.E(**kwargs) * self.dy_vault / self.dx_vault

    def Kx(self, **kwargs):  # noqa: N802
        """
        Compute the total equivalent radial stiffness of the entire case structure.

        Combines:
        - Two lateral case elements (left and right) per winding pack in series
            with the WP
        - All such combined units in parallel with the PS and vault regions

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments passed to subcomponent stiffness evaluations.

        Returns
        -------
        float
            Total equivalent radial stiffness of the TF case [Pa].
        """
        temp = [
            serie_k([
                self.Kx_lat(**kwargs)[i],
                w.Kx(**kwargs),
                self.Kx_lat(**kwargs)[i],
            ])
            for i, w in enumerate(self.WPs)
        ]
        return parall_k([self.Kx_ps(**kwargs), self.Kx_vault(**kwargs), *temp])

    def Ky_ps(self, **kwargs):  # noqa: N802
        """
        Compute the equivalent toroidal stiffness of the poloidal support (PS) region.

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments passed to the material's Young's modulus function.

        Returns
        -------
        float
            Equivalent toroidal stiffness of the PS region [Pa].
        """
        return self.mat_case.E(**kwargs) * self.dx_ps / self.dy_ps

    def Ky_lat(self, **kwargs):  # noqa: N802
        """
        Compute the equivalent toroidal stiffness of lateral case sections
        per winding pack.

        Each lateral piece is treated as a rectangular beam in the toroidal direction.

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments passed to the material's Young's modulus function.

        Returns
        -------
        np.ndarray
            Array of toroidal stiffness values for each lateral segment [Pa].
        """
        dx_lat = np.array([
            (self.R_wp_i[i] + self.R_wp_k[i]) / 2 * np.tan(self._rad_theta_TF / 2)
            - w.dx / 2
            for i, w in enumerate(self.WPs)
        ])
        dy_lat = np.array([w.dy for w in self.WPs])
        return self.mat_case.E(**kwargs) * dx_lat / dy_lat

    def Ky_vault(self, **kwargs):  # noqa: N802
        """
        Compute the equivalent toroidal stiffness of the vault region.

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments passed to the material's Young's modulus function.

        Returns
        -------
        float
            Equivalent toroidal stiffness of the vault [Pa].
        """
        return self.mat_case.E(**kwargs) * self.dx_vault / self.dy_vault

    def Ky(self, **kwargs):  # noqa: N802
        """
        Compute the total equivalent toroidal stiffness of the entire case structure.

        Combines:
        - Each winding pack and its adjacent lateral case sections in parallel
        - These parallel combinations are arranged in series with the PS and
            vault regions

        Parameters
        ----------
        **kwargs :
            Optional keyword arguments passed to subcomponent stiffness evaluations.

        Returns
        -------
        float
            Total equivalent toroidal stiffness of the TF case [Pa].
        """
        temp = [
            parall_k([
                self.Ky_lat(**kwargs)[i],
                w.Ky(**kwargs),
                self.Ky_lat(**kwargs)[i],
            ])
            for i, w in enumerate(self.WPs)
        ]
        return serie_k([self.Ky_ps(**kwargs), self.Ky_vault(**kwargs), *temp])

    def _tresca_stress(self, pm: float, fz: float, **kwargs):
        """
        Estimate the maximum principal (Tresca) stress on the inner case of the TF coil.

        The stress is computed as the sum of:
        - The hoop (circumferential) compressive stress due to radial magnetic pressure.
        - The vertical tensile stress due to axial electromagnetic forces on the inner
            leg.

        The hoop stress is derived using classical shell theory for isotropic continuous
        cylindrical shells, corrected by the stiffness distribution in the structure.

        Parameters
        ----------
        pm : float
            Radial magnetic pressure acting on the case [Pa].
        fz : float
            Vertical force acting on the inner leg of the case [N].
        **kwargs :
            Additional keyword arguments forwarded to the stiffness calculations (e.g.,
            temperature, material model switches).

        Returns
        -------
        float
            Estimated maximum stress [Pa] acting on the case nose (hoop + vertical
            contribution).
        """
        # The maximum principal stress acting on the case nose is the compressive
        # hoop stress generated in the equivalent shell from the magnetic pressure. From
        # the Shell theory, for an isotropic continuous shell with a thickness ratio:
        beta = self.Rk / (self.Rk + self.dy_vault)
        # the maximum hoop stress, corrected to account for the presence of the WP, is
        # placed at the innermost radius of the case as:
        sigma_theta = (
            2.0 / (1 - beta**2) * pm * self.Kx_vault(**kwargs) / self.Kx(**kwargs)
        )

        # In addition to the radial centripetal force, the second in-plane component
        # to be accounted is the vertical force acting on the TFC inner-leg.
        # t_z = 0.5*np.log(self.Ri / Re) * MU_0_4PI * (360. / self.theta_TF) * I ** 2

        # As conservative approximation, the vertical force is considered to act only
        # on jackets and vault
        sigma_z = fz / (self.area_case_jacket + self.area_wps_jacket)
        return sigma_theta + sigma_z

    def optimize_vault_radial_thickness(
        self,
        pm: float,
        fz: float,
        T: float,  # noqa: N803
        B: float,
        allowable_sigma: float,
        bounds: np.array = None,
    ):
        """
        Optimize the vault radial thickness of the case

        Parameters
        ----------
        pm :
            The magnetic pressure applied along the radial direction (Pa).
        f_z :
            The force applied in the z direction, perpendicular to the case
            cross-section (N).
        T :
            The operating temperature (K).
        B :
            The operating magnetic field (T).
        allowable_sigma :
            The allowable stress (Pa) for the jacket material.
        bounds :
            Optional bounds for the jacket thickness optimization (default is None).

        Returns
        -------
            The result of the optimization process containing information about the
            optimal vault thickness.

        Raises
        ------
        ValueError
            If the optimization process did not converge.
        """
        bluemira_print(
            f"pm: {pm}, fz: {fz}, T: {T}, B: {B}, allowable_sigma: {allowable_sigma}"
        )
        bluemira_print(f"bounds: {bounds}")

        method = None
        if bounds is not None:
            method = "bounded"

        result = minimize_scalar(
            fun=self._sigma_difference,
            args=(pm, fz, T, B, allowable_sigma),
            bounds=bounds,
            method=method,
            options={"xatol": 1e-4},
        )

        if not result.success:
            raise ValueError("dy_vault optimization did not converge.")
        self.dy_vault = result.x
        # print(f"Optimal dy_vault: {self.dy_vault}")
        # print(f"Tresca sigma: {self._tresca_stress(pm, fz, T=T, B=B) / 1e6} MPa")

        return result

    def _sigma_difference(
        self,
        dy_vault: float,
        pm: float,
        fz: float,
        temperature: float,
        B: float,
        allowable_sigma: float,
    ):
        """
        Fitness function for the optimization problem. It calculates the absolute
        difference between
        the Tresca stress and the allowable stress.

        Parameters
        ----------
        dy_vault :
            The thickness of the vault in the direction perpendicular to the
            applied pressure(m).
        pm :
            The magnetic pressure applied along the radial direction (Pa).
        fz :
            The force applied in the z direction, perpendicular to the case
            cross-section (N).
        temperature :
            The temperature (K) at which the conductor operates.
        B :
            The magnetic field (T) at which the conductor operates.
        allowable_sigma :
            The allowable stress (Pa) for the vault material.

        Returns
        -------
            The absolute difference between the calculated Tresca stress and the
            allowable stress (Pa).

        Notes
        -----
            This function modifies the case's vault thickness
            using the value provided in jacket_thickness.
        """
        self.dy_vault = dy_vault
        sigma = self._tresca_stress(pm, fz, temperature=temperature, B=B)
        # bluemira_print(f"sigma: {sigma}, allowable_sigma: {allowable_sigma},
        # diff: {sigma - allowable_sigma}")
        return abs(sigma - allowable_sigma)

    def optimize_jacket_and_vault(
        self,
        pm: float,
        fz: float,
        temperature: float,
        B: float,
        allowable_sigma: float,
        bounds_cond_jacket: np.array = None,
        bounds_dy_vault: np.array = None,
        layout: str = "auto",
        wp_reduction_factor: float = 0.8,
        min_gap_x: float = 0.05,
        n_layers_reduction: int = 4,
        max_niter: int = 10,
        eps: float = 1e-8,
        n_conds: int | None = None,
    ):
        """
        Jointly optimize the conductor jacket and case vault thickness
        under electromagnetic loading constraints.

        This method performs an iterative optimization of:
        - The cross-sectional area of the conductor jacket.
        - The vault radial thickness of the TF coil casing.

        The optimization loop continues until the relative change in
        jacket area and vault thickness drops below the specified
        convergence threshold `eps`, or `max_niter` is reached.

        Parameters
        ----------
        pm : float
            Radial magnetic pressure on the conductor [Pa].
        fz : float
            Axial electromagnetic force on the winding pack [N].
        temperature : float
            Operating temperature [K].
        B : float
            Operating magnetic field [T].
        allowable_sigma : float
            Maximum allowable stress for structural material [Pa].
        bounds_cond_jacket : np.ndarray, optional
            Min/max bounds for conductor jacket area optimization [m²].
        bounds_dy_vault : np.ndarray, optional
            Min/max bounds for the case vault thickness optimization [m].
        layout : str, optional
            Cable layout strategy; "auto" or predefined layout name.
        wp_reduction_factor : float, optional
            Reduction factor applied to WP footprint during conductor rearrangement.
        min_gap_x : float, optional
            Minimum spacing between adjacent conductors [m].
        n_layers_reduction : int, optional
            Number of conductor layers to remove when reducing WP height.
        max_niter : int, optional
            Maximum number of optimization iterations.
        eps : float, optional
            Convergence threshold for the combined optimization loop.
        n_conds : int, optional
            Target total number of conductors in the winding pack. If None, the self
            number of conductors is used.

        Notes
        -----
            The function modifies the internal state of `conductor` and `self.dy_vault`.
        """
        if n_conds is None:
            n_conds = self.n_conductors
        conductor = self.WPs[0].conductor
        tot_err = 100 * eps
        i = 0
        while i < max_niter and tot_err > eps:
            i += 1
            bluemira_debug(f"Internal optimazion - iteration {i}")
            bluemira_debug(
                f"before optimization: conductor jacket area = {conductor.area_jacket}"
            )
            cond_area_jacket0 = conductor.area_jacket
            t_z_cable_jacket = (
                fz
                * self.area_wps_jacket
                / (self.area_case_jacket + self.area_wps_jacket)
                / self.n_conductors
            )
            conductor.optimize_jacket_conductor(
                pm, t_z_cable_jacket, temperature, B, allowable_sigma, bounds_cond_jacket
            )
            bluemira_debug(f"t_z_cable_jacket: {t_z_cable_jacket}")
            bluemira_debug(
                f"after optimization: conductor jacket area = {conductor.area_jacket}"
            )
            err_conductor_area_jacket = (
                abs(conductor.area_jacket - cond_area_jacket0) / cond_area_jacket0
            )

            self.rearrange_conductors_in_wp(
                n_conds,
                conductor,
                self.R_wp_i[0],
                wp_reduction_factor,
                min_gap_x,
                n_layers_reduction,
                layout=layout,
            )
            case_dy_vault0 = self.dy_vault
            (f"before optimization: case dy_vault = {self.dy_vault}")
            self.optimize_vault_radial_thickness(
                pm=pm,
                fz=fz,
                T=temperature,
                B=B,
                allowable_sigma=allowable_sigma,
                bounds=bounds_dy_vault,
            )

            delta_case_dy_vault = abs(self.dy_vault - case_dy_vault0)
            err_dy_vault = delta_case_dy_vault / self.dy_vault
            tot_err = err_dy_vault + err_conductor_area_jacket

            bluemira_debug(
                f"after optimization: case dy_vault = {self.dy_vault}\n"
                f"err_dy_jacket = {err_conductor_area_jacket}\n "
                f"err_dy_vault = {err_dy_vault}\n "
                f"tot_err = {tot_err}"
            )

    def plot(self, ax=None, *, show: bool = False, homogenized: bool = False):
        """
        Schematic plot of the case cross-section.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axis on which to draw the figure. If `None`, a new figure and axis
            will be created automatically.
        show : bool, optional
            If `True`, displays the plot immediately using `plt.show()`.
            Default is `False`.
        homogenized : bool, optional
            If `True`, the winding packs are drawn as homogenized blocks.
            If `False`, they are drawn using their actual geometry.
            Default is `False`.

        Returns
        -------
        matplotlib.axes.Axes
            The axis object containing the plot, useful for further customization.
        """
        if ax is None:
            _, ax = plt.subplots()
            ax.set_aspect("equal", adjustable="box")

        p0 = np.array([-self.dx_i / 2, self.Ri])
        p1 = np.array([self.dx_i / 2, self.Ri])
        p2 = np.array([self.dx_k / 2, self.Rk])
        p3 = np.array([-self.dx_k / 2, self.Rk])

        points_ext = np.vstack((p0, p1, p2, p3, p0))

        ax.plot(points_ext[:, 0], points_ext[:, 1], "r")
        for i, w in enumerate(self.WPs):
            xc_w = 0
            yc_w = self.R_wp_i[i] - w.dy / 2
            ax = w.plot(xc=xc_w, yc=yc_w, ax=ax, homogenized=homogenized)

        if show:
            plt.show()

        return ax

    def rearrange_conductors_in_wp(
        self,
        n_conductors: int,
        cond: Conductor,
        R_wp_i: float,  # noqa: N803
        wp_reduction_factor: float,
        min_gap_x: float,
        n_layers_reduction: int,
        layout: str = "auto",
    ):
        """
        Rearrange the total number of conductors into winding packs (WPs)
        within the TF coil case geometry.

        This method attempts to distribute the provided number of conductors
        (`n_conductors`) into multiple winding packs based on geometric constraints
        and layout strategy. It dynamically computes feasible values for number
        of layers (`nx`) and turns (`ny`) per winding pack while satisfying
        radial and toroidal space requirements.

        Parameters
        ----------
        n_conductors : int
            Total number of conductors to be allocated.
        cond : Conductor
            The conductor object with known dimensions (`dx`, `dy`).
        R_wp_i : float
            Radial position of the inner edge of the first winding pack [m].
        wp_reduction_factor : float
            Fractional reduction of the total available toroidal space for WPs.
        min_gap_x : float
            Minimum allowable gap in toroidal direction between winding pack and coil
            boundary [m].
        n_layers_reduction : int
            Number of layers to remove in subsequent winding packs to control growth
            in `dx`.
        layout : str, optional
            Layout type for conductor placement. Supported options:
            - `"auto"`: no constraints on nx or ny
            - `"layer"`: enforces even number of vertical turns (ny % 2 == 0)
            - `"pancake"`: enforces even number of horizontal layers (nx % 2 == 0)

        Raises
        ------
        ValueError
            Raised when the available space is insufficient to place even one layer
            or one turn of conductors based on the current layout and geometry.

        Notes
        -----
        - The number of conductors allocated may differ slightly from `n_conductors`
          due to rounding needed to close final layers or turns.
        - The winding packs are stored in `self.WPs`.

        """
        dx_WP = self.dx_i * wp_reduction_factor  # noqa: N806
        bluemira_debug(f"dx_WP = {dx_WP}")
        bluemira_debug(f"self.dx_i = {self.dx_i}")
        bluemira_debug(f"wp_reduction_factor = {wp_reduction_factor}")

        WPs = []  # noqa: N806
        # number of conductors to be allocated
        remaining_conductors = n_conductors
        # maximum number of internal iterations
        i_max = 50
        i = 0
        while i < i_max and remaining_conductors > 0:
            i += 1
            # bluemira_debug(f"Rearrange conductors in WP - iteration: {i}")
            # bluemira_debug(f"remaining_conductors: {remaining_conductors}")

            # maximum toroidal dimension of the WP most outer pancake
            # dx_WP = 2 * (R_wp_i * np.tan(self._rad_theta_TF / 2) - dx0_wp)

            # maximum number of turns on the considered WP
            if i == 1:
                n_layers_max = math.floor(dx_WP / cond.dx)
                if layout == "pancake":
                    n_layers_max = math.floor(dx_WP / cond.dx / 2.0) * 2
                    if n_layers_max == 0:
                        n_layers_max = 2
            else:
                n_layers_max -= n_layers_reduction

            if n_layers_max < 1:
                raise ValueError(
                    f"n_layers_max: {n_layers_max} < 1. There is not enough space to "
                    f"allocate all the conductors"
                )

            if n_layers_max >= remaining_conductors:
                WPs.append(WindingPack(conductor=cond, nx=remaining_conductors, ny=1))
                remaining_conductors = 0
            else:
                dx_WP = n_layers_max * cond.dx  # noqa: N806

                gap_0 = R_wp_i * np.tan(self._rad_theta_TF / 2) - dx_WP / 2
                gap_1 = min_gap_x

                max_dy = (gap_0 - gap_1) / np.tan(self._rad_theta_TF / 2)
                n_turns_max = min(
                    int(np.floor(max_dy / cond.dy)),
                    int(np.ceil(remaining_conductors / n_layers_max)),
                )
                if layout == "layer":
                    n_turns_max = min(
                        int(np.floor(max_dy / cond.dy / 2.0) * 2),
                        int(np.ceil(remaining_conductors / n_layers_max / 2.0) * 2),
                    )
                    if n_turns_max == 0:
                        n_turns_max = 2

                if n_turns_max < 1:
                    raise ValueError(
                        f"n_turns_max: {n_turns_max} < 1. There is not enough space to "
                        f"allocate all the conductors"
                    )

                if n_layers_max * n_turns_max > remaining_conductors:
                    n_turns_max -= 1
                    WPs.append(
                        WindingPack(conductor=cond, nx=n_layers_max, ny=n_turns_max)
                    )
                    remaining_conductors -= n_layers_max * n_turns_max
                    WPs.append(
                        WindingPack(conductor=cond, nx=remaining_conductors, ny=1)
                    )
                    remaining_conductors = 0
                else:
                    WPs.append(
                        WindingPack(conductor=cond, nx=n_layers_max, ny=n_turns_max)
                    )
                    remaining_conductors -= n_layers_max * n_turns_max

                if remaining_conductors < 0:
                    bluemira_warn(
                        f"{abs(remaining_conductors)}/{n_layers_max * n_turns_max}"
                        f"have been added to complete the last winding pack (nx"
                        f"={n_layers_max}, ny={n_turns_max})."
                    )

                R_wp_i -= n_turns_max * cond.dy  # noqa: N806
                # dx_WP = dx_WP - n_layers_reduction * cond.dx
                # print(f"remaining_conductors: {remaining_conductors}")

        self.WPs = WPs
