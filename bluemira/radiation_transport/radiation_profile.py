# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
1-D radiation model inspired by the PROCESS function "plot_radprofile" in plot_proc.py.
"""

from __future__ import annotations

import functools
import operator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from bluemira.base import constants
from bluemira.base.parameter_frame import Parameter, ParameterFrame, make_parameter_frame
from bluemira.display.plotter import Zorder, plot_coordinates
from bluemira.equilibria.physics import calc_psi_norm
from bluemira.geometry.coordinates import Coordinates
from bluemira.radiation_transport.flux_surfaces_maker import (
    analyse_first_wall_flux_surfaces,
)
from bluemira.radiation_transport.radiation_tools import (
    calculate_line_radiation_loss,
    calculate_total_radiated_power,
    electron_density_and_temperature_sol_decay,
    exponential_decay,
    gaussian_decay,
    get_impurity_data,
    interpolated_field_values,
    ion_front_distance,
    linear_interpolator,
    radiative_loss_function_plot,
    radiative_loss_function_values,
    specific_point_temperature,
    target_temperature,
    upstream_temperature,
)

if TYPE_CHECKING:
    import numpy.typing as npt

    from bluemira.base.parameter_frame.typed import ParameterFrameLike
    from bluemira.equilibria.equilibrium import Equilibrium
    from bluemira.equilibria.flux_surfaces import PartialOpenFluxSurface
    from bluemira.equilibria.grid import Grid
    from bluemira.geometry.wire import BluemiraWire
    from bluemira.radiation_transport.midplane_temperature_density import (
        MidplaneProfiles,
    )


@dataclass
class RadiationSourceParams(ParameterFrame):
    """Radiaition source parameter frame"""

    sep_corrector_omp: Parameter[float]
    """Separation correction for double and single null plasma"""
    sep_corrector_imp: Parameter[float]
    """Separation correction for double and single null plasma"""
    alpha_n: Parameter[float]
    """Density profile factor"""
    alpha_t: Parameter[float]
    """Temperature profile index"""
    det_t: Parameter[float]
    """Detachment target temperature"""
    eps_cool: Parameter[float]
    """electron energy loss"""
    f_ion_t: Parameter[float]
    """Hydrogen first ionization"""
    main_ext: Parameter[float]
    """radiation region extention in main chamber"""
    rec_ext_out_leg: Parameter[float]
    """recyccling region extetion in outer leg"""
    rec_ext_in_leg: Parameter[float]
    """recyccling region extetion in inner leg"""
    fw_lambda_q_far_imp: Parameter[float]
    """Lambda_q far SOL imp"""
    fw_lambda_q_far_omp: Parameter[float]
    """Lambda_q far SOL omp"""
    fw_lambda_q_near_imp: Parameter[float]
    """Lambda_q near SOL imp"""
    fw_lambda_q_near_omp: Parameter[float]
    """Lambda_q near SOL omp"""
    lambda_t_factor: Parameter[float]
    """Lambda_t factor for non conduction-limited regime"""
    lambda_n_factor: Parameter[float]
    """Lambda_n factor for non conduction-limited regime"""
    gamma_sheath: Parameter[float]
    """sheath heat transmission coefficient"""
    k_0: Parameter[float]
    """material's conductivity"""
    lfs_p_fraction: Parameter[float]
    """lfs fraction of SoL power"""
    n_e_0: Parameter[float]
    """Electron density on axis"""
    n_e_ped: Parameter[float]
    """Electron density pedestal height"""
    n_e_sep: Parameter[float]
    """Electron density at separatrix"""
    P_sep: Parameter[float]
    """Radiation power"""
    rho_ped_n: Parameter[float]
    """Density pedestal r/a location"""
    rho_ped_t: Parameter[float]
    """Temperature pedestal r/a location"""
    n_points_core_95: Parameter[float]
    """rho discretization to 95% of core"""
    n_points_core_99: Parameter[float]
    """rho discretization to 99% of core"""
    n_points_mantle: Parameter[float]
    """rho discretization to separatrix"""
    t_beta: Parameter[float]
    """Temperature profile index beta"""
    T_e_0: Parameter[float]
    """Electron temperature on axis"""
    T_e_ped: Parameter[float]
    """Electron temperature pedestal height"""
    T_e_sep: Parameter[float]
    """Electron temperature at separatrix"""
    theta_inner_target: Parameter[float]
    """Inner divertor poloidal angle with the separatrix flux line"""
    theta_outer_target: Parameter[float]
    """Outer divertor poloidal angle with the separatrix flux line"""


class Radiation:
    """
    Initial and generic class (no distinction between core and SOL)
    to calculate radiation source within the flux tubes.
    """

    def __init__(self, eq: Equilibrium, params: dict | ParameterFrame):
        self.params = params
        self.eq = eq

    def collect_flux_tubes(self, psi_n: np.ndarray) -> list[Coordinates]:
        """
        Collect flux tubes according to the normalised psi.
        For now only used for the core as for the SoL the
        flux surfaces to calculate the heat flux from charged
        particles are used.

        Parameters
        ----------
        psi_n:
            normalised psi

        Returns
        -------
        flux tubes:
            list of flux tubes
        """
        return [self.eq.get_flux_surface(psi) for psi in psi_n]

    @staticmethod
    def flux_tube_pol_t(
        flux_tube: Coordinates,
        te_mp: float,
        t_rad_in: float = 0,
        t_rad_out: float = 0,
        rad_i: np.ndarray | None = None,
        rec_i: np.ndarray | None = None,
        t_tar: float = 0,
        *,
        core: bool = False,
        x_point_rad: bool = False,
        main_chamber_rad: bool = False,
    ) -> np.ndarray:
        """
        Along a single flux tube, it assigns different temperature values.

        Parameters
        ----------
        flux_tube:
            flux tube geometry
        te_mp:
            electron temperature at the midplane [keV]
        t_rad_in:
            temperature value at the entrance of the radiation region [keV]
        t_rad_out:
            temperature value at the exit of the radiation region [keV]
        rad_i:
            indexes of points, belonging to the flux tube, which fall
            into the radiation region
        rec_i:
            indexes of points, belonging to the flux tube, which fall
            into the recycling region
        t_tar:
            electron temperature at the target [keV]
        core:
            if True, t is constant along the flux tube. If false,it varies
        x_point_rad:
            if True, it assumes there is no radiation at all
            in the recycling region.
        main_chamber_rad:
            if True, the temperature from the midplane to
            the radiation region entrance is not constant

        Returns
        -------
        te [keV]:
            poloidal distribution of electron temperature

        Raises
        ------
        ValueError
            Required inputs not provided
        """
        te = np.array([te_mp] * len(flux_tube))

        if core is True:
            return te

        if rad_i is not None:
            if len(rad_i) == 1:
                te[rad_i] = t_rad_in
            elif len(rad_i) > 1:
                te[rad_i] = gaussian_decay(t_rad_in, t_rad_out, len(rad_i), decay=True)

        if rec_i is not None:
            if x_point_rad:
                te[rec_i] = exponential_decay(
                    t_rad_out * 0.95, t_tar, len(rec_i), decay=True
                )
            else:
                te[rec_i] = t_tar

        if main_chamber_rad:
            if rad_i is None or rec_i is None:
                raise ValueError("'rad_i' and 'rec_i' must be specified")
            mask = np.ones_like(te, dtype=bool)
            main_rad = np.concatenate((rad_i, rec_i))
            mask[main_rad] = False
            te[mask] = np.linspace(te_mp, t_rad_in, len(te[mask]))

        return te

    @staticmethod
    def flux_tube_pol_n(
        flux_tube: Coordinates,
        ne_mp: float,
        n_rad_in: float | None = None,
        n_rad_out: float | None = None,
        rad_i: np.ndarray | None = None,
        rec_i: np.ndarray | None = None,
        n_tar: float | None = None,
        *,
        core: bool = False,
        main_chamber_rad: bool = False,
    ) -> np.ndarray:
        """
        Along a single flux tube, it assigns different density values.

        Parameters
        ----------
        flux_tube:
            flux tube geometry
        ne_mp:
            electron density at the midplane [1/m^3]
        n_rad_in:
            density value at the entrance of the radiation region [1/m^3]
        n_rad_out:
            density value at the exit of the radiation region [1/m^3]
        rad_i:
            indexes of points, belonging to the flux tube, which fall
            into the radiation region
        rec_i:
            indexes of points, belonging to the flux tube, which fall
            into the recycling region
        n_tar:
            electron density at the target [1/m^3]
        core:
            if True, n is constant along the flux tube. If false,it varies
        main_chamber_rad:
            if True, the temperature from the midplane to
            the radiation region entrance is not constant

        Returns
        -------
        ne:
            poloidal distribution of electron density

        Raises
        ------
        ValueError
            Required inputs not provided
        """
        # initializing ne with same values all along the flux tube
        ne = np.array([ne_mp] * len(flux_tube))
        if core is True:
            return ne

        if rad_i is not None and len(rad_i) == 1:
            ne[rad_i] = n_rad_in
        elif rad_i is not None and len(rad_i) > 1:
            ne[rad_i] = gaussian_decay(n_rad_out, n_rad_in, len(rad_i), decay=False)

        if rec_i is not None and len(rad_i) == 1:
            ne[rec_i] = n_tar
        elif rec_i is not None and len(rec_i) > 0:
            ne[rec_i] = gaussian_decay(n_rad_out, n_tar, len(rec_i))
        if rec_i is not None and len(rec_i) > 0:
            if rad_i is None:
                raise ValueError("'rad_i' not specified with 'rec_i'")
            if len(rad_i) > 1:
                gap = ne[rad_i[-1]] - ne[rad_i[-2]]
                ne[rec_i] = gaussian_decay(n_rad_out - gap, n_tar, len(rec_i))

        if main_chamber_rad:
            mask = np.ones_like(ne, dtype=bool)
            main_rad = np.concatenate((rad_i, rec_i))
            mask[main_rad] = False
            ne[mask] = np.linspace(ne_mp, n_rad_in, len(ne[mask]))

        return ne

    @staticmethod
    def mp_profile_plot(
        rho: np.ndarray, rad_power: np.ndarray, imp_name: str | list[str], ax=None
    ) -> plt.Axes:
        """
        1D plot of the radiation power distribution along the midplane.

        Parameters
        ----------
        rho:
            dimensionless radius. Values between 0 and 1 for the plasma core.
            Values > 1 for the scrape-off layer
        rad_power:
            radiated power at each mid-plane location corresponding to rho [Mw/m^3]
        imp_name:
            impurity names

        Returns
        -------
        ax:
            axes on which the mid-plane radiation power distribution profile is plotted.

        Notes
        -----
        if rad_power.ndim > 1 the plot shows as many line as the number
        of impurities, plus the total radiated power
        """
        if ax is None:
            _, ax = plt.subplots()
            plt.xlabel(r"$\rho$")
            plt.ylabel(r"$[MW.m^{-3}]$")

        if len(rad_power) == 1:
            ax.plot(
                rho, rad_power, imp_name if isinstance(imp_name, str) else imp_name[0]
            )
        else:
            for rad_part, name in zip(rad_power, imp_name, strict=False):
                ax.plot(rho, rad_part, label=name)
            plt.title("Core radiated power density")
            ax.plot(
                rho,
                np.sum(np.array(rad_power, dtype=object), axis=0).tolist(),
                label="total radiation",
            )
            ax.legend(loc="best", borderaxespad=0, fontsize=12)

        return ax


class CoreRadiation(Radiation):
    """
    Specific class to calculate the core radiation source.

    Temperature and density are assumed to be constant along a single flux tube.
    In addition to `Radiation`, this class also includes the impurity data of all
    gases except Argon.

    Parameters
    ----------
    eq:
        The equilibrium defining flux surfaces.
    midplane_profiles:
        Electron density and electron temperature profile at the mid-plane.
    impurity_content:
        The dictionary of impurities (e.g. 'H') and their fractions (e.g. 1E-2)
        in the core.
    impurity_data:
        The dictionary of impurities in the core at a defined time, sorted by
        species, then sorted by "T_ref" v.s. "L_ref", where
        T_ref = reference ion temperature [eV],
        L_ref = the loss function value $L_z(n_e, T_e)$ [W m^3].
    """

    def __init__(
        self,
        eq: Equilibrium,
        params: ParameterFrame,
        midplane_profiles: MidplaneProfiles,
        impurity_content: dict[str, float],
        impurity_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
    ):
        super().__init__(eq, params)

        # Picking impurity species
        included_species = list(impurity_data)

        # Using the selected list to build other lists
        self.impurities_content = [impurity_content[key] for key in included_species]

        self.imp_data_t_ref = [
            constants.raw_uc(impurity_data[key]["T_ref"], "eV", "keV")
            for key in included_species
        ]

        self.imp_data_l_ref = [impurity_data[key]["L_ref"] for key in included_species]

        self.imp_data_z_ref = [impurity_data[key]["z_ref"] for key in included_species]

        # Store impurity symbols
        self.impurity_symbols = impurity_content.keys()

        # Store the midplane profiles
        self.profiles = midplane_profiles

    def calculate_mp_radiation_profile(self):
        """
        1D profile of the line radiation loss at the mid-plane
        from the magnetic axis to the separatrix
        """
        # Radiative loss function values for each impurity species
        loss_f = [
            radiative_loss_function_values(self.profiles.te, t_ref, l_ref)
            for t_ref, l_ref in zip(
                self.imp_data_t_ref, self.imp_data_l_ref, strict=False
            )
        ]

        # Line radiation loss. Mid-plane distribution through the SoL
        self.rad_mp = [
            calculate_line_radiation_loss(self.profiles.ne, loss, fi)
            for loss, fi in zip(loss_f, self.impurities_content, strict=False)
        ]

    def plot_mp_radiation_profile(self):
        """
        Plot one dimensional behaviour of line radiation
        against the adimensional radius
        """
        self.mp_profile_plot(self.profiles.psi_n, self.rad_mp, self.impurity_symbols)

    def calculate_core_distribution(self) -> list[list[np.ndarray]]:
        """
        Build poloidal distribution (distribution along the field lines) of
        line radiation loss in the plasma core.

        Returns
        -------
        rad :
            Line core radiation for each impurity species
            and for each closed flux line in the core.
        """
        # Collect closed flux tubes within the separatrix
        self.flux_tubes = self.collect_flux_tubes(self.profiles.psi_n)

        # Calculate poloidal density profile for each flux tube
        self.ne_pol = [
            self.flux_tube_pol_n(ft, n, core=True)
            for ft, n in zip(self.flux_tubes, self.profiles.ne, strict=False)
        ]
        # Calculate poloidal temperature profile for each flux tube
        self.te_pol = [
            self.flux_tube_pol_t(ft, t, core=True)
            for ft, t in zip(self.flux_tubes, self.profiles.te, strict=False)
        ]

        # Calculate the radiative power loss function for each impurity
        # species and for each flux tube
        self.loss_f = [
            [radiative_loss_function_values(t, t_ref, l_ref) for t in self.te_pol]
            for t_ref, l_ref in zip(
                self.imp_data_t_ref, self.imp_data_l_ref, strict=False
            )
        ]

        # Calculate the line radiation loss for each impurity species
        # and for each flux tube
        self.rad = [
            [
                calculate_line_radiation_loss(n, l_f, fi)
                for n, l_f in zip(self.ne_pol, loss_per_species, strict=False)
            ]
            for loss_per_species, fi in zip(
                self.loss_f, self.impurities_content, strict=False
            )
        ]

        return self.rad

    def calculate_core_radiation_map(self):
        """
        Build core radiation map.

        Returns
        -------
        :
            the core radiation map
        """
        rad = self.calculate_core_distribution()
        self.total_rad = np.sum(np.array(rad, dtype=object), axis=0).tolist()

        self.x_tot = np.concatenate([flux_tube.x for flux_tube in self.flux_tubes])
        self.z_tot = np.concatenate([flux_tube.z for flux_tube in self.flux_tubes])
        self.rad_tot = np.concatenate(self.total_rad)

        # Calculate the total radiated power
        return calculate_total_radiated_power(self.x_tot, self.z_tot, self.rad_tot)

    def radiation_distribution_plot(
        self, flux_tubes: np.ndarray, power_density: np.ndarray, ax=None
    ) -> plt.Axes:
        """
        2D plot of the core radiation power distribution.

        Parameters
        ----------
        flux_tubes:
            array of the closed flux tubes within the separatrix.
        power_density:
            arrays containing the power radiation density of the
            points lying on each flux tube [MW/m^3]

        Returns
        -------
        :
            the axes
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        p_min = min(np.amin(p) for p in power_density)
        p_max = max(np.amax(p) for p in power_density)

        separatrix = self.eq.get_separatrix()
        if isinstance(separatrix, Coordinates):
            separatrix = [separatrix]

        for sep in separatrix:
            plot_coordinates(sep, ax=ax, linewidth=0.2)
        for flux_tube, p in zip(flux_tubes, power_density, strict=False):
            cm = ax.scatter(
                flux_tube.x,
                flux_tube.z,
                c=p,
                s=10,
                cmap="plasma",
                vmin=p_min,
                vmax=p_max,
                zorder=Zorder.RADIATION.value,
            )

        fig.colorbar(cm, label=r"$[MW.m^{-3}]$")

        return ax

    def plot_radiation_distribution(self):
        """
        Plot poloiadal radiation distribution
        within the plasma core
        """
        self.radiation_distribution_plot(self.flux_tubes, self.total_rad)

    def plot_lz_vs_tref(self):
        """
        Plot radiative loss function for a set of given impurities
        against the reference temperature.
        """
        radiative_loss_function_plot(
            self.imp_data_t_ref[0], self.imp_data_l_ref, self.impurity_symbols
        )


class ScrapeOffLayerRadiation(Radiation):
    """
    Specific class to calculate the SOL radiation source.
    In the SOL is assumed a conduction dominated regime until the
    x-point, with no heat sinks, and a convection dominated regime
    between x-point and target.

    Parameters
    ----------
    eq:
        The equilibrium defining flux surfaces.
    x_sep_omp:
        outboard mid-plane separatrix x-coordinates [m]
    x_sep_imp
        inboard mid-plane separatrix x-coordinates [m]
    dx_omp:
        The midplane spatial resolution between flux surfaces at the outboard [m]
    dx_imp
        The midplane spatial resolution between flux surfaces at the inboard [m]
    """

    def __init__(
        self,
        eq: Equilibrium,
        params: ParameterFrame,
        x_sep_omp: float | None = None,
        x_sep_imp: float | None = None,
        dx_omp: float | None = None,
        dx_imp: float | None = None,
    ):
        super().__init__(eq, params)

        # Constructors - The mid-plane radii
        self.x_sep_omp = x_sep_omp
        self.x_sep_imp = x_sep_imp
        self.dx_omp = dx_omp
        self.dx_imp = dx_imp

        self.collect_x_and_o_point_coordinates()

        # Separatrix parameters
        self.collect_separatrix_parameters()

    def collect_x_and_o_point_coordinates(self):
        """
        Magnetic axis coordinates and x-point(s) coordinates.
        """
        o_point, x_point = self.eq.get_OX_points()
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

    def collect_separatrix_parameters(self):
        """
        Radiation source relevant parameters at the separatrix
        """
        self.separatrix = self.eq.get_separatrix()
        self.z_mp = self.points["o_point"]["z"]
        if self.eq.is_double_null:
            # The two halves
            self.sep_lfs = self.separatrix[0]
            self.sep_hfs = self.separatrix[1]
        else:
            ob_ind = np.nonzero(self.separatrix.x > self.points["x_point"]["x"])
            ib_ind = np.nonzero(self.separatrix.x < self.points["x_point"]["x"])
            self.sep_ob = Coordinates({
                "x": self.separatrix.x[ob_ind],
                "z": self.separatrix.z[ob_ind],
            })
            self.sep_ib = Coordinates({
                "x": self.separatrix.x[ib_ind],
                "z": self.separatrix.z[ib_ind],
            })
        # To move away from the mathematical separatrix which would
        # give infinite connection length
        self.r_sep_omp = self.x_sep_omp + self.params.sep_corrector_omp.value
        # magnetic field components at the midplane
        self.b_pol_sep_omp = self.eq.Bp(self.x_sep_omp, self.z_mp)
        b_tor_sep_omp = self.eq.Bt(self.x_sep_omp)
        self.b_tot_sep_omp = np.hypot(self.b_pol_sep_omp, b_tor_sep_omp)

        if self.eq.is_double_null:
            self.r_sep_imp = self.x_sep_imp - self.params.sep_corrector_imp.value
            self.b_pol_sep_imp = self.eq.Bp(self.x_sep_imp, self.z_mp)
            b_tor_sep_imp = self.eq.Bt(self.x_sep_imp)
            self.b_tot_sep_imp = np.hypot(self.b_pol_sep_imp, b_tor_sep_imp)

    def x_point_radiation_z_ext(
        self,
        main_ext: float | None = None,
        pfr_ext: float = 0.3,
        *,
        low_div: bool = True,
    ) -> tuple[float, ...]:
        """
        Simple definition of a radiation region around the x-point.
        The region is supposed to extend from an arbitrary z coordinate on the
        main plasma side, to an arbitrary z coordinate on the private flux region side.

        Parameters
        ----------
        main_ext:
            region extension on the main plasma side [m]
        pfr_ext:
            region extension on the private flux region side [m]
        low_div: boolean
            default=True for the lower divertor. If False, upper divertor

        Returns
        -------
        z_main:
            vertical (z coordinate) extension of the radiation region
            toward the main plasma [m]
        z_pfr:
            vertical (z coordinate) extension of the radiation region
            toward the pfr [m]
        """
        if main_ext is None:
            main_ext = abs(self.points["x_point"]["z_low"])

        if low_div is True:
            z_main = self.points["x_point"]["z_low"] + main_ext
            z_pfr = self.points["x_point"]["z_low"] - pfr_ext
        else:
            z_main = self.points["x_point"]["z_up"] - main_ext
            z_pfr = self.points["x_point"]["z_up"] + pfr_ext

        return z_main, z_pfr

    def radiation_region_ends(
        self, z_main: float, z_pfr: float, *, lfs: bool = True
    ) -> tuple[float, ...]:
        """
        Entering and exiting points (x, z) of the radiation region
        detected on the separatrix.
        The separatrix is supposed to be given by relevant half.

        Parameters
        ----------
        z_main:
            vertical (z coordinate) extension of the radiation region
            toward the main plasma [m]
        z_pfr:
            vertical (z coordinate) extension of the radiation region
            toward the pfr [m]
        lfs:
            default=True for the low field side (right half).
            If False, high field side (left half).

        Returns
        -------
        entrance_x, entrance_z:
            x, z coordinates of the radiation region starting point
        exit_x, exit_z:
            x, z coordinates of the radiation region ending point
        """
        sep_loop = (
            (self.sep_lfs if lfs else self.sep_hfs)
            if self.eq.is_double_null
            else (self.sep_ob if lfs else self.sep_ib)
        )
        if z_main > z_pfr:
            reg_i = np.nonzero((sep_loop.z < z_main) & (sep_loop.z >= z_pfr))[0]
            i_in = np.nonzero(sep_loop.z == np.max(sep_loop.z[reg_i]))[0]
            i_out = np.nonzero(sep_loop.z == np.min(sep_loop.z[reg_i]))[0]
        else:
            reg_i = np.nonzero((sep_loop.z > z_main) & (sep_loop.z <= z_pfr))[0]
            i_in = np.nonzero(sep_loop.z == np.min(sep_loop.z[reg_i]))[0]
            i_out = np.nonzero(sep_loop.z == np.max(sep_loop.z[reg_i]))[0]

        entrance_x, entrance_z = sep_loop.x[i_in], sep_loop.z[i_in]
        exit_x, exit_z = sep_loop.x[i_out], sep_loop.z[i_out]

        return entrance_x[0], entrance_z[0], exit_x[0], exit_z[0]

    @staticmethod
    def radiation_region_points(
        flux_tube: Coordinates, z_main: float, z_pfr: float, *, lower: bool = True
    ) -> tuple[np.ndarray, ...]:
        """
        For a given flux tube, indexes of points which fall respectively
        into the radiation and recycling region

        Parameters
        ----------
        flux_tube:
            flux tube geometry
        z_main:
            vertical (z coordinate) extension of the radiation region toward
            the main plasma.
            Taken on the separatrix [m]
        z_pfr:
            vertical (z coordinate) extension of the radiation region
            toward the pfr.
            Taken on the separatrix [m]
        lower:
            default=True for the lower divertor. If False, upper divertor

        Returns
        -------
        rad_i:
            indexes of the points within the radiation region
        rec_i:
            indexes pf the points within the recycling region
        """
        if lower:
            rad_i = np.nonzero((flux_tube.z < z_main) & (flux_tube.z >= z_pfr))[0]
            rec_i = np.nonzero(flux_tube.z < z_pfr)[0]
        else:
            rad_i = np.nonzero((flux_tube.z > z_main) & (flux_tube.z <= z_pfr))[0]
            rec_i = np.nonzero(flux_tube.z > z_pfr)[0]

        return rad_i, rec_i

    def mp_electron_density_temperature_profiles(
        self, te_sep: float | None = None, *, omp: bool = True
    ) -> tuple[np.ndarray, ...]:
        """
        Calculation of electron density and electron temperature profiles
        across the SoL at midplane.
        It uses the customised version for the mid-plane of the exponential
        decay law described in "electron_density_and_temperature_sol_decay".

        Parameters
        ----------
        te_sep:
            electron temperature at the separatrix [keV]
        omp:
            outer mid-plane. Default value True. If False it stands for inner mid-plane

        Returns
        -------
        te_sol:
            radial decayed temperatures through the SoL at the mid-plane. Unit [keV]
        ne_sol:
            radial decayed densities through the SoL at the mid-plane. Unit [1/m^3]
        """
        if omp or not self.eq.is_double_null:
            fw_lambda_q_near = self.params.fw_lambda_q_near_omp.value
            fw_lambda_q_far = self.params.fw_lambda_q_far_omp.value
            dx = self.dx_omp
        else:
            fw_lambda_q_near = self.params.fw_lambda_q_near_imp.value
            fw_lambda_q_far = self.params.fw_lambda_q_far_imp.value
            dx = self.dx_imp

        if te_sep is None:
            te_sep = self.params.T_e_sep.value_as("eV")
        ne_sep = self.params.n_e_sep.value

        te_sol, ne_sol = electron_density_and_temperature_sol_decay(
            te_sep,
            ne_sep,
            fw_lambda_q_near,
            fw_lambda_q_far,
            dx,
        )

        return te_sol, ne_sol

    def any_point_density_temperature_profiles(
        self,
        x_p: float,
        z_p: float,
        t_p: float,
        t_u: float,
        *,
        lfs: bool = True,
    ) -> tuple[np.ndarray, ...]:
        """
        Calculation of electron density and electron temperature profiles
        across the SoL, starting from any point on the separatrix.
        (The z coordinate is the same. While the x coordinate changes)
        Using the equation to calculate T(s||).

        Parameters
        ----------
        x_p:
            x coordinate of the point at the separatrix [m]
        z_p:
            z coordinate of the point at the separatrix [m]
        t_p:
            point temperature [eV]
        t_u:
            upstream temperature [eV]
        lfs:
            low (toroidal) field side (outer wall side). Default value True.
            If False it stands for high field side (hfs).

        Returns
        -------
        te_prof:
            radial decayed temperatures through the SoL. Unit [eV]
        ne_prof:
            radial decayed densities through the SoL. Unit [1/m^3]
        """
        # Distinction between lfs and hfs
        if lfs or not self.eq.is_double_null:
            r_sep_mp = self.r_sep_omp
            b_pol_sep_mp = self.b_pol_sep_omp
            fw_lambda_q_near = self.params.fw_lambda_q_near_omp.value
            fw_lambda_q_far = self.params.fw_lambda_q_far_omp.value
            dx = self.dx_omp
        else:
            r_sep_mp = self.r_sep_imp
            b_pol_sep_mp = self.b_pol_sep_imp
            fw_lambda_q_near = self.params.fw_lambda_q_near_imp.value
            fw_lambda_q_far = self.params.fw_lambda_q_far_imp.value
            dx = self.dx_imp

        # magnetic field components at the local point
        b_pol_p = self.eq.Bp(x_p, z_p)

        # flux expansion
        f_p = (r_sep_mp * b_pol_sep_mp) / (x_p * b_pol_p)

        # Ratio between upstream and local temperature
        f_t = t_u / t_p

        # Local electron density
        n_p = self.params.n_e_sep.value * f_t

        # Temperature and density profiles across the SoL
        te_prof, ne_prof = electron_density_and_temperature_sol_decay(
            t_p,
            n_p,
            fw_lambda_q_near,
            fw_lambda_q_far,
            dx,
            f_exp=f_p,
            t_factor_det=self.params.lambda_t_factor.value,
            n_factor_det=self.params.lambda_n_factor.value,
        )

        return te_prof, ne_prof

    def tar_electron_densitiy_temperature_profiles(
        self,
        ne_div: np.ndarray,
        te_div: np.ndarray,
        f_m: float = 1,
        *,
        detachment: bool = False,
    ) -> tuple[np.ndarray, ...]:
        """
        Calculation of electron density and electron temperature profiles
        across the SoL at the target.
        From the pressure balance, considering friction losses.

        Parameters
        ----------
        ne_div:
            density of the flux tubes at the entrance of the recycling region,
            assumed to be corresponding to the divertor plane [1/m^3]
        te_div:
            temperature of the flux tubes at the entrance of the recycling region,
            assumed to be corresponding to the divertor plane [eV]
        f_m:
            fractional loss of pressure due to friction.
            It can vary between 0 and 1.

        Returns
        -------
        te_t:
            target temperature [eV]
        ne_t:
            target density [1/m^3]
        """
        if detachment:
            te_t = np.full(len(te_div), self.params.det_t.value_as("eV"))
            f_m = 0.1
        else:
            te_t = te_div
        ne_t = (f_m * ne_div) / 2

        return te_t, ne_t

    def calculate_sector_distributions(
        self,
        flux_tubes: np.ndarray,
        x_strike: float,
        z_strike: float,
        main_ext: float,
        firstwall_geom: Coordinates,
        pfr_ext: float | None = None,
        rec_ext: float | None = None,
        *,
        x_point_rad: bool = False,
        detachment: bool = False,
        lfs: bool = True,
        low_div: bool = True,
        main_chamber_rad: bool = False,
    ) -> tuple[np.ndarray, ...]:
        """
        Temperature and density profiles calculation.
        Within the scrape-off layer sector, it gives temperature
        and density profile along each flux tube.

        Parameters
        ----------
        flux_tubes: array
            set of flux tubes
        x_strike: float
            x coordinate of the first open flux surface strike point [m]
        z_strike: float
            z coordinate of the first open flux surface strike point [m]
        main_ext: float
            extention of the radiation region from the x-point
            towards the main plasma [m]
        firstwall_geom:
            first wall geometry
        pfr_ext:
            extention of the radiation region from the x-point
            towards private flux region [m]
        rec_ext:
            extention of the recycling region,
            along the separatrix, from the target [m]
        x_point_rad:
            if True, it assumes there is no radiation at all
            in the recycling region, and pfr_ext MUST be provided.
        detachment:
            if True, it makes the temperature decay through the
            recycling region from the H ionization temperature to
            the assign temperature for detachment at the target.
            Else temperature is constant through the recycling region.
        lfs:
            low field side. Default value True.
            If False it stands for high field side (hfs)
        low_div:
            default=True for the lower divertor.
            If False, upper divertor
        main_chamber_rad:
            if True, the temperature from the midplane to
            the radiation region entrance is not constant

        Returns
        -------
        t_pol:
            temperature poloidal profile along each
            flux tube within the specified set [keV]
        n_pol:
            density poloidal profile along each
            flux tube within the specified set [1/m^3]

        Raises
        ------
        ValueError
            Required inputs not provided
        """
        f_ion_t_eV = self.params.f_ion_t.value_as("eV")

        # Validity condition for not x-point radiative
        if not x_point_rad and rec_ext is None:
            raise ValueError("Required recycling region extention: rec_ext")
        if not x_point_rad and rec_ext is not None and lfs:
            ion_front_z = ion_front_distance(
                x_strike,
                z_strike,
                self.eq,
                self.points["x_point"]["z_low"],
                rec_ext=self.params.rec_ext_out_leg.value,
            )
            pfr_ext = abs(ion_front_z)

        elif not x_point_rad and rec_ext is not None and lfs is False:
            ion_front_z = ion_front_distance(
                x_strike,
                z_strike,
                self.eq,
                self.points["x_point"]["z_low"],
                rec_ext=self.params.rec_ext_in_leg.value,
            )
            pfr_ext = abs(ion_front_z)

        # Validity condition for x-point radiative
        elif x_point_rad and pfr_ext is None:
            raise ValueError("Required extention towards pfr: pfr_ext")

        # setting radiation and recycling regions
        z_main, z_pfr = self.x_point_radiation_z_ext(main_ext, pfr_ext, low_div=low_div)

        in_x, in_z, out_x, out_z = self.radiation_region_ends(z_main, z_pfr, lfs=lfs)

        reg_i = [
            self.radiation_region_points(f.coords, z_main, z_pfr, lower=low_div)
            for f in flux_tubes
        ]

        # mid-plane parameters
        if lfs or not self.eq.is_double_null:
            t_u_kev = self.t_omp
            b_pol_tar = self.b_pol_out_tar
            b_pol_u = self.b_pol_sep_omp
            r_sep_mp = self.r_sep_omp
            # alpha = self.params.theta_outer_target.value
            alpha = self.alpha_lfs
            b_tot_tar = self.b_tot_out_tar
            fw_lambda_q_near = self.params.fw_lambda_q_near_omp.value
            sep_corrector = self.params.sep_corrector_omp.value
        else:
            t_u_kev = self.t_imp
            b_pol_tar = self.b_pol_inn_tar
            b_pol_u = self.b_pol_sep_imp
            r_sep_mp = self.r_sep_imp
            # alpha = self.params.theta_inner_target.value
            alpha = self.alpha_hfs
            b_tot_tar = self.b_tot_inn_tar
            fw_lambda_q_near = self.params.fw_lambda_q_near_imp.value
            sep_corrector = self.params.sep_corrector_imp.value

        # Coverting needed parameter units
        t_u_ev = constants.raw_uc(t_u_kev, "keV", "eV")
        p_sol = self.params.P_sep.value

        if self.eq.is_double_null:
            p_sol *= (
                self.params.lfs_p_fraction.value
                if lfs
                else (1 - self.params.lfs_p_fraction.value)
            )

        t_mp_prof, n_mp_prof = self.mp_electron_density_temperature_profiles(
            t_u_ev, omp=lfs
        )
        # entrance of radiation region
        t_rad_in = specific_point_temperature(
            in_x,
            in_z,
            t_u_ev,
            p_sol,
            fw_lambda_q_near,
            self.eq,
            r_sep_mp,
            self.points["o_point"]["z"],
            self.params.k_0.value,
            sep_corrector,
            firstwall_geom,
            lfs=lfs,
        )

        # exit of radiation region
        t_rad_out = (
            f_ion_t_eV
            if (x_point_rad and pfr_ext is not None) or detachment
            else target_temperature(
                p_sol,
                t_u_ev,
                self.params.n_e_sep.value,
                self.params.gamma_sheath.value,
                self.params.eps_cool.value_as("eV"),
                f_ion_t_eV,
                b_pol_tar,
                b_pol_u,
                alpha,
                r_sep_mp,
                x_strike,
                fw_lambda_q_near,
                b_tot_tar,
            )
        )

        # condition for occurred detachment
        if t_rad_out <= self.params.f_ion_t.value_as("eV"):
            x_point_rad = detachment = True

        # profiles through the SoL
        t_in_prof, n_in_prof = self.any_point_density_temperature_profiles(
            in_x,
            in_z,
            t_rad_in,
            t_u_ev,
            lfs=lfs,
        )

        t_out_prof, n_out_prof = self.any_point_density_temperature_profiles(
            out_x,
            out_z,
            t_rad_out,
            t_u_ev,
            lfs=lfs,
        )

        t_tar_prof, n_tar_prof = self.tar_electron_densitiy_temperature_profiles(
            n_out_prof,
            t_out_prof,
            detachment=detachment,
        )
        # temperature poloidal distribution
        t_pol = [
            self.flux_tube_pol_t(
                f.coords,
                t,
                t_rad_in=t_in,
                t_rad_out=t_out,
                rad_i=reg[0],
                rec_i=reg[1],
                t_tar=t_t,
                x_point_rad=x_point_rad,
                main_chamber_rad=main_chamber_rad,
            )
            for f, t, t_in, t_out, reg, t_t in zip(
                flux_tubes,
                t_mp_prof,
                t_in_prof,
                t_out_prof,
                reg_i,
                t_tar_prof,
                strict=False,
            )
        ]
        # density poloidal distribution
        n_pol = [
            self.flux_tube_pol_n(
                f.coords,
                n,
                n_rad_in=n_in,
                n_rad_out=n_out,
                rad_i=reg[0],
                rec_i=reg[1],
                n_tar=n_t,
                main_chamber_rad=main_chamber_rad,
            )
            for f, n, n_in, n_out, reg, n_t in zip(
                flux_tubes,
                n_mp_prof,
                n_in_prof,
                n_out_prof,
                reg_i,
                n_tar_prof,
                strict=False,
            )
        ]

        return t_pol, n_pol

    def radiation_distribution_plot(self, flux_tubes, power_density, firstwall, ax=None):
        """
        2D plot of the radation power distribution.

        Parameters
        ----------
        flux_tubes: [np.array]
            list of arrays of the flux tubes of different sectors.
            Example: if len(flux_tubes) = 2, it could contain the set
            of flux tubes of the lower that go from the omp to the
            outer lower divertor, and the set of flux tubes that go from
            the imp to the inner lower divertor.
        power_density: [np.array]
            list of arrays containing the power radiation density of the
            points lying on the specified flux tubes.
            expected len(flux_tubes) = len(power_density)
        firstwall: Grid
            first wall geometry

        Returns
        -------
        ax:
            Axes on which the 2D radiation power distribution is plotted.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        tubes = functools.reduce(operator.iadd, flux_tubes, [])
        power = functools.reduce(operator.iadd, power_density, [])

        p_min = min(min(p) for p in power)
        p_max = max(max(p) for p in power)

        plot_coordinates(firstwall, ax=ax, linewidth=0.5, fill=False)
        separatrix = self.eq.get_separatrix()
        if isinstance(separatrix, Coordinates):
            separatrix = [separatrix]

        for sep in separatrix:
            plot_coordinates(sep, ax=ax, linewidth=0.2)
        for flux_tube, p in zip(tubes, power, strict=False):
            cm = ax.scatter(
                flux_tube.coords.x,
                flux_tube.coords.z,
                c=p,
                s=10,
                marker=".",
                cmap="plasma",
                vmin=p_min,
                vmax=p_max,
                zorder=Zorder.RADIATION.value,
            )

        fig.colorbar(cm, label=r"$[MW.m^{-3}]$")

        return ax

    @staticmethod
    def poloidal_distribution_plot(
        flux_tubes: np.ndarray,
        flux_property: np.ndarray,
        *,
        temperature: bool = True,
        ax=None,
    ):
        """
        2D plot of a generic property (density, temperature or radiation)
        as poloidal section the flux tube points.

        Parameters
        ----------
        flux_tubes
            array of the open flux tubes within the SoL.
        flux_property
            arrays containing the property values associated
            to the points of each flux tube.

        Returns
        -------
        ax:
            Axes on which the flux tubes' properties are plotted.
        """
        if ax is None:
            _, ax = plt.subplots()

        plt.xlabel("Mid-Plane to Target")
        if temperature is True:
            plt.title("Temperature along flux surfaces")
            plt.ylabel(r"$T_e~[keV]$")
        else:
            plt.title("Density along flux surfaces")
            plt.ylabel(r"$n_e~[m^{-3}]$")

        for flux_tube, val in zip(flux_tubes, flux_property, strict=False):
            ax.plot(
                np.linspace(0, len(flux_tube.coords.x), len(flux_tube.coords.x)), val
            )

        return ax

    @staticmethod
    def plot_t_vs_n(flux_tube, t_distribution, n_distribution, ax1=None) -> plt.Axes:
        """
        2D plot of temperature and density of a single flux tube within the SoL

        Parameters
        ----------
        flux_tube: flux surface object as described in advective_transport.py
            open flux tube within the SoL.
        t_distribution: np.array([np.array])
            arrays containing the temperature values associated
            to the points of the flux tube.
        n_distribution: np.array([np.array])
            arrays containing the density values associated
            to the points of the flux tube.

        Returns
        -------
        ax1:
            Axes object on which the electron temperature is plotted
        ax2:
            Axes object on which the electron density is plotted
        """
        if ax1 is None:
            _, ax1 = plt.subplots()
        else:
            _ = ax1.figure

        ax2 = ax1.twinx()
        x = np.linspace(0, len(flux_tube.coords.x), len(flux_tube.coords.x))
        y1 = t_distribution
        y2 = n_distribution
        ax1.plot(x, y1, "g-")
        ax2.plot(x, y2, "b-")

        plt.title("Electron Temperature vs Electron Density")
        ax1.set_xlabel("Mid-Plane to Target")
        ax1.set_ylabel(r"$T_e~[keV]$", color="g")
        ax2.set_ylabel(r"$n_e~[m^{-3}]$", color="b")

        return ax1, ax2


class DNScrapeOffLayerRadiation(ScrapeOffLayerRadiation):
    """
    Specific class to make the SOL radiation source for a double null configuration.
    Here the SOL is divided into for regions. From the outer midplane to the
    outer lower target; from the omp to the outer upper target; from the inboard
    midplane to the inner lower target; from the imp to the inner upper target.

    Parameters
    ----------
    eq:
        The equilibrium defining flux surfaces.
    flux_surfaces:
        list of flux surfaces, all of which terminating at the first walls.
    impurity_content:
        The dictionary of impurities in the double-null's scrape-off layer
        (e.g. 'H') and their fractions (e.g. 1E-2).
    impurity_data:
        The dictionary of impurities in the double-null's scrape-off layer at a
        defined time, sorted by species, then sorted by "T_ref" v.s. "L_ref", where
        T_ref = reference ion temperature [eV],
        L_ref = the loss function value $L_z(n_e, T_e)$ [W m^3].
    firstwall_geom:
        The closed first wall geometry
    x_sep_omp:
        outboard mid-plane separatrix x-coordinates [m]
    x_sep_imp
        inboard mid-plane separatrix x-coordinates [m]
    dx_omp:
        The midplane spatial resolution between flux surfaces at the outboard [m]
    dx_imp:
        The midplane spatial resolution between flux surfaces at the inboard [m]
    """

    def __init__(
        self,
        eq: Equilibrium,
        params: ParameterFrame,
        flux_surfaces: list[PartialOpenFluxSurface],
        impurity_content: dict[str, float],
        impurity_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
        firstwall_geom: Coordinates,
        x_sep_omp: float | None = None,
        x_sep_imp: float | None = None,
        dx_omp: float | None = None,
        dx_imp: float | None = None,
    ):
        super().__init__(eq, params, x_sep_omp, x_sep_imp, dx_omp, dx_imp)

        self.impurities_content = [
            frac for key, frac in impurity_content.items() if key != "H"
        ]
        self.imp_data_t_ref = [
            data["T_ref"] for key, data in impurity_data.items() if key != "H"
        ]
        self.imp_data_l_ref = [
            data["L_ref"] for key, data in impurity_data.items() if key != "H"
        ]
        # Flux tubes from the particle solver
        # partial flux tube from the mp to the target at the
        # outboard and inboard - lower divertor
        self.flux_tubes_lfs_low = flux_surfaces[0]
        self.flux_tubes_hfs_low = flux_surfaces[2]

        # partial flux tube from the mp to the target at the
        # outboard and inboard - upper divertor
        self.flux_tubes_lfs_up = flux_surfaces[1]
        self.flux_tubes_hfs_up = flux_surfaces[3]

        # strike points from the first open flux tube
        self.x_strike_lfs = self.flux_tubes_lfs_low[0].coords.x[-1]
        self.z_strike_lfs = self.flux_tubes_lfs_low[0].coords.z[-1]
        self.alpha_lfs = self.flux_tubes_lfs_low[0].alpha

        self.b_pol_out_tar = self.eq.Bp(self.x_strike_lfs, self.z_strike_lfs)
        self.b_tor_out_tar = self.eq.Bt(self.x_strike_lfs)
        self.b_tot_out_tar = np.hypot(self.b_pol_out_tar, self.b_tor_out_tar)

        self.x_strike_hfs = self.flux_tubes_hfs_low[0].coords.x[-1]
        self.z_strike_hfs = self.flux_tubes_hfs_low[0].coords.z[-1]
        self.alpha_hfs = self.flux_tubes_hfs_low[0].alpha

        self.b_pol_inn_tar = self.eq.Bp(self.x_strike_hfs, self.z_strike_hfs)
        self.b_tor_inn_tar = self.eq.Bt(self.x_strike_hfs)
        self.b_tot_inn_tar = np.hypot(self.b_pol_inn_tar, self.b_tor_inn_tar)

        p_sol = self.params.P_sep.value
        p_sol_lfs = p_sol * self.params.lfs_p_fraction.value
        p_sol_hfs = p_sol * (1 - self.params.lfs_p_fraction.value)

        # upstream temperature and power density
        self.t_omp = upstream_temperature(
            b_pol=self.b_pol_sep_omp,
            b_tot=self.b_tot_sep_omp,
            lambda_q_near=self.params.fw_lambda_q_near_omp.value,
            p_sol=p_sol_lfs,
            eq=self.eq,
            r_sep_mp=self.r_sep_omp,
            z_mp=self.z_mp,
            k_0=self.params.k_0.value,
            firstwall_geom=firstwall_geom,
        )

        self.t_imp = upstream_temperature(
            b_pol=self.b_pol_sep_imp,
            b_tot=self.b_tot_sep_imp,
            lambda_q_near=self.params.fw_lambda_q_near_imp.value,
            p_sol=p_sol_hfs,
            eq=self.eq,
            r_sep_mp=self.r_sep_imp,
            z_mp=self.z_mp,
            k_0=self.params.k_0.value,
            firstwall_geom=firstwall_geom,
        )

    def calculate_sol_distribution(self, firstwall_geom: Grid):
        """
        Temperature and density profiles calculation.
        For each scrape-off layer sector, it gives temperature
        and density profile along each flux tube.

        Parameters
        ----------
        firstwall_geom: grid
            first wall geometry

        Returns
        -------
        t_and_n_pol["lfs_low"]:
            temperature and density poloidal profile along each
            flux tube within the lfs lower divertor set
        t_and_n_pol["lfs_up"]:
            temperature and density poloidal profile along each
            flux tube within the lfs upper divertor set
        t_and_n_pol["hfs_low"]:
            temperature and density poloidal profile along each
            flux tube within the hfs lower divertor set
        t_and_n_pol["hfs_up"]:
            temperature and density poloidal profile along each
            flux tube within the hfs upper divertor set
        """
        self.t_and_n_pol = {
            f"{side}_{low_up}": self.calculate_sector_distributions(
                flux_tubes=getattr(self, f"flux_tubes_{side}_{low_up}"),
                x_strike=getattr(self, f"x_strike_{side}"),
                z_strike=getattr(self, f"z_strike_{side}"),
                main_ext=self.params.main_ext.value,
                firstwall_geom=firstwall_geom,
                pfr_ext=None,
                rec_ext=self.params.rec_ext_out_leg.value,
                x_point_rad=False,
                detachment=False,
                lfs=side == "lfs",
                low_div=low_up == "low",
                main_chamber_rad=True,
            )
            for side in ["lfs", "hfs"]
            for low_up in ["low", "up"]
        }

        return self.t_and_n_pol

    def calculate_sol_radiation_distribution(
        self,
        lfs_low,
        lfs_up,
        hfs_low,
        hfs_up,
    ):
        """
        Radiation profiles calculation.
        For each scrape-off layer sector, it gives the
        radiation profile along each flux tube.

        Parameters
        ----------
        lfs_low:
            temperature and density poloidal profile along each
            flux tube within the lfs lower divertor set
        lfs_up:
            temperature and density poloidal profile along each
            flux tube within the lfs upper divertor set
        hfs_low:
            temperature and density poloidal profile along each
            flux tube within the hfs lower divertor set
        hfs_up:
            temperature and density poloidal profile along each
            flux tube within the hfs upper divertor set

        Returns
        -------
        rad["lfs_low"]:
            radiation poloidal profile along each
            flux tube within the lfs lower divertor set
        rad["lfs_up"]:
            radiation poloidal profile along each
            flux tube within the lfs upper divertor set
        rad["hfs_low"]:
            radiation poloidal profile along each
            flux tube within the hfs lower divertor set
        rad["hfs_up"]:
            radiation poloidal profile along each
            flux tube within the hfs upper divertor set
        """
        # For each impurity species and for each flux tube,
        # poloidal distribution of the radiative power loss function.
        # Values along the open flux tubes
        loss_data = {
            "lfs_low": lfs_low[0],
            "lfs_up": lfs_up[0],
            "hfs_low": hfs_low[0],
            "hfs_up": hfs_up[0],
        }

        loss = {
            side: [
                [radiative_loss_function_values(t, t_ref, l_ref) for t in t_pol]
                for t_ref, l_ref in zip(
                    self.imp_data_t_ref, self.imp_data_l_ref, strict=False
                )
            ]
            for side, t_pol in loss_data.items()
        }

        # For each impurity species and for each flux tube,
        # poloidal distribution of the line radiation loss.
        # Values along the open flux tubes
        ft_density = {
            "lfs_low": lfs_low[1],
            "lfs_up": lfs_up[1],
            "hfs_low": hfs_low[1],
            "hfs_up": hfs_up[1],
        }
        self.rad = {
            side: [
                [
                    calculate_line_radiation_loss(n, l_f, fi)
                    for n, l_f in zip(ft_density[side], f, strict=False)
                ]
                for f, fi in zip(loss[side], self.impurities_content, strict=False)
            ]
            for side in ft_density
        }

        return self.rad

    def calculate_sol_radiation_map(self, lfs_low, lfs_up, hfs_low, hfs_up):
        """
        Scrape off layer radiation map calculation.

        Parameters
        ----------
        lfs_low:
            radiation poloidal profile along each
            flux tube within the lfs lower divertor set
        lfs_up:
            radiation poloidal profile along each
            flux tube within the lfs upper divertor set
        hfs_low:
            radiation poloidal profile along each
            flux tube within the hfs lower divertor set
        hfs_up:
            radiation poloidal profile along each
            flux tube within the hfs upper divertor set

        Returns
        -------
        :
            the sol radiation map
        """
        # total line radiation loss along the open flux tubes
        self.total_rad_lfs_low = np.sum(np.array(lfs_low, dtype=object), axis=0).tolist()
        self.total_rad_lfs_up = np.sum(np.array(lfs_up, dtype=object), axis=0).tolist()
        self.total_rad_hfs_low = np.sum(np.array(hfs_low, dtype=object), axis=0).tolist()
        self.total_rad_hfs_up = np.sum(np.array(hfs_up, dtype=object), axis=0).tolist()

        flux_tubes = functools.reduce(
            operator.iadd,
            [
                self.flux_tubes_lfs_low,
                self.flux_tubes_hfs_low,
                self.flux_tubes_lfs_up,
                self.flux_tubes_hfs_up,
            ],
            [],
        )

        self.x_tot = np.concatenate([flux_tube.coords.x for flux_tube in flux_tubes])
        self.z_tot = np.concatenate([flux_tube.coords.z for flux_tube in flux_tubes])
        self.rad_tot = np.concatenate(
            functools.reduce(
                operator.iadd,
                [
                    self.total_rad_lfs_low,
                    self.total_rad_hfs_low,
                    self.total_rad_lfs_up,
                    self.total_rad_hfs_up,
                ],
                [],
            )
        )

        # Calculate the total radiated power
        return calculate_total_radiated_power(self.x_tot, self.z_tot, self.rad_tot)

    def plot_poloidal_radiation_distribution(self, firstwall_geom: Grid):
        """
        Plot poloiadal radiation distribution
        within the scrape-off layer

        Parameters
        ----------
        firstwall_geom:
            first wall geometry
        """
        self.radiation_distribution_plot(
            [
                self.flux_tubes_lfs_low,
                self.flux_tubes_hfs_low,
                self.flux_tubes_lfs_up,
                self.flux_tubes_hfs_up,
            ],
            [
                self.total_rad_lfs_low,
                self.total_rad_hfs_low,
                self.total_rad_lfs_up,
                self.total_rad_hfs_up,
            ],
            firstwall_geom,
        )


class SNScrapeOffLayerRadiation(ScrapeOffLayerRadiation):
    """
    Specific class to make the SOL radiation source for a double null configuration.
    Here the SOL is divided into for regions. From the outer midplane to the
    outer lower target; from the omp to the outer upper target; from the inboard
    midplane to the inner lower target; from the imp to the inner upper target.

    Parameters
    ----------
    eq:
        The equilibrium defining flux surfaces.
    flux_surfaces:
        list of flux surfaces, all of which terminating at the first walls.
    impurity_content:
        The dictionary of impurities in the single-null scrape-off layer
        (e.g. 'H') and their fractions (e.g. 1E-2).
    impurity_data:
        The dictionary of impurities in the single-null scrape-off layer at a defined
        time, sorted by species, then sorted by "T_ref" v.s. "L_ref", where
        T_ref = reference ion temperature [eV],
        L_ref = the loss function value $L_z(n_e, T_e)$ [W m^3].
    firstwall_geom:
        The closed first wall geometry
    x_sep_omp:
        outboard mid-plane separatrix x-coordinates [m]
    dx_omp:
        The midplane spatial resolution between flux surfaces at the outboard [m]
    """

    def __init__(
        self,
        eq: Equilibrium,
        params: ParameterFrame,
        flux_surfaces: list[PartialOpenFluxSurface],
        impurity_content: dict[str, float],
        impurity_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]],
        firstwall_geom: Coordinates,
        x_sep_omp: float,
        dx_omp: float,
    ):
        super().__init__(eq, params, x_sep_omp=x_sep_omp, dx_omp=dx_omp)

        self.impurities_content = [
            frac for key, frac in impurity_content.items() if key != "H"
        ]

        self.imp_data_t_ref = [
            data["T_ref"] for key, data in impurity_data.items() if key != "H"
        ]
        self.imp_data_l_ref = [
            data["L_ref"] for key, data in impurity_data.items() if key != "H"
        ]
        # Flux tubes from the particle solver
        # partial flux tube from the mp to the target at the
        # outboard and inboard - lower divertor
        self.flux_tubes_lfs = flux_surfaces[0]
        self.flux_tubes_hfs = flux_surfaces[1]

        # strike points from the first open flux tube
        self.x_strike_lfs = self.flux_tubes_lfs[0].coords.x[-1]
        self.z_strike_lfs = self.flux_tubes_lfs[0].coords.z[-1]
        self.alpha_lfs = self.flux_tubes_lfs[0].alpha

        self.b_pol_out_tar = self.eq.Bp(self.x_strike_lfs, self.z_strike_lfs)
        self.b_tor_out_tar = self.eq.Bt(self.x_strike_lfs)
        self.b_tot_out_tar = np.hypot(self.b_pol_out_tar, self.b_tor_out_tar)

        self.x_strike_hfs = self.flux_tubes_hfs[0].coords.x[-1]
        self.z_strike_hfs = self.flux_tubes_hfs[0].coords.z[-1]
        self.alpha_hfs = self.flux_tubes_hfs[0].alpha

        self.b_pol_inn_tar = self.eq.Bp(self.x_strike_hfs, self.z_strike_hfs)
        self.b_tor_inn_tar = self.eq.Bt(self.x_strike_hfs)
        self.b_tot_inn_tar = np.hypot(self.b_pol_inn_tar, self.b_tor_inn_tar)

        p_sol = self.params.P_sep.value

        # upstream temperature and power density
        self.t_omp = upstream_temperature(
            b_pol=self.b_pol_sep_omp,
            b_tot=self.b_tot_sep_omp,
            lambda_q_near=self.params.fw_lambda_q_near_omp.value,
            p_sol=p_sol,
            eq=self.eq,
            r_sep_mp=self.r_sep_omp,
            z_mp=self.points["o_point"]["z"],
            k_0=self.params.k_0.value,
            firstwall_geom=firstwall_geom,
        )

    def calculate_sol_distribution(
        self, firstwall_geom: Grid
    ) -> dict[str, npt.NDArray[np.float64]]:
        """
        Temperature and density profiles calculation.
        For each scrape-off layer sector, it gives temperature
        and density profile along each flux tube.

        Parameters
        ----------
        firstwall_geom:
            first wall geometry

        Returns
        -------
        t_and_n_pol["lfs"]:
            temperature and density poloidal profile along each
            flux tube within the lfs divertor set
        t_and_n_pol["hfs"]:
            temperature and density poloidal profile along each
            flux tube within the hfs divertor set
        """
        return {
            side: self.calculate_sector_distributions(
                flux_tubes=getattr(self, f"flux_tubes_{side}"),
                x_strike=getattr(self, f"x_strike_{side}"),
                z_strike=getattr(self, f"z_strike_{side}"),
                main_ext=1,
                firstwall_geom=firstwall_geom,
                pfr_ext=None,
                rec_ext=self.params.rec_ext_out_leg.value,
                x_point_rad=False,
                detachment=False,
                lfs=side == "lfs",
                low_div=True,
                main_chamber_rad=True,
            )
            for side in ["lfs", "hfs"]
        }

    def calculate_sol_radiation_distribution(
        self,
        lfs: np.ndarray,
        hfs: np.ndarray,
    ) -> dict[str, list[list[np.ndarray]]]:
        """
        Radiation profiles calculation.
        For each scrape-off layer sector, it gives the
        radiation profile along each flux tube.

        Parameters
        ----------
        lfs:
            temperature and density poloidal profile along each
            flux tube within the lfs divertor set
        hfs:
            temperature and density poloidal profile along each
            flux tube within the hfs divertor set

        Returns
        -------
        rad["lfs"]:
            radiation poloidal profile along each
            flux tube within the lfs divertor set
        rad["hfs"]:
            radiation poloidal profile along each
            flux tube within the hfs divertor set
        """
        # For each impurity species and for each flux tube,
        # poloidal distribution of the radiative power loss function.
        # Values along the open flux tubes
        loss = {"lfs": lfs[0], "hfs": hfs[0]}

        for side, t_pol in loss.items():
            loss[side] = [
                [radiative_loss_function_values(t, t_ref, l_ref) for t in t_pol]
                for t_ref, l_ref in zip(
                    self.imp_data_t_ref, self.imp_data_l_ref, strict=False
                )
            ]

        # For each impurity species and for each flux tube,
        # poloidal distribution of the line radiation loss.
        # Values along the open flux tubes
        rad_data = {
            "lfs": {"density": lfs[1], "loss": loss["lfs"]},
            "hfs": {"density": hfs[1], "loss": loss["hfs"]},
        }
        self.rad = {}
        for side, ft in rad_data.items():
            self.rad[side] = [
                [
                    calculate_line_radiation_loss(n, l_f, fi)
                    for n, l_f in zip(ft["density"], f, strict=False)
                ]
                for f, fi in zip(ft["loss"], self.impurities_content, strict=False)
            ]
        return self.rad

    def calculate_sol_radiation_map(self, lfs: np.ndarray, hfs: np.ndarray):
        """
        Scrape off layer radiation map calculation.

        Parameters
        ----------
        lfs:
            radiation poloidal profile along each
            flux tube within the lfs divertor set
        hfs:
            radiation poloidal profile along each
            flux tube within the hfs upper divertor set
        """
        # total line radiation loss along the open flux tubes
        self.total_rad_lfs = np.sum(np.array(lfs, dtype=object), axis=0).tolist()
        self.total_rad_hfs = np.sum(np.array(hfs, dtype=object), axis=0).tolist()

        flux_tubes = functools.reduce(
            operator.iadd, [self.flux_tubes_lfs, self.flux_tubes_hfs], []
        )

        self.x_tot = np.concatenate([flux_tube.coords.x for flux_tube in flux_tubes])
        self.z_tot = np.concatenate([flux_tube.coords.z for flux_tube in flux_tubes])
        self.rad_tot = np.concatenate(
            # concatenate these two lists into a single float np.ndarray
            functools.reduce(operator.iadd, [self.total_rad_lfs, self.total_rad_hfs], [])
        )

    def plot_poloidal_radiation_distribution(self, firstwall_geom: Coordinates):
        """
        Plot poloiadal radiation distribution
        within the scrape-off layer

        Parameters
        ----------
        firstwall:
            first wall geometry
        """
        self.radiation_distribution_plot(
            [self.flux_tubes_lfs, self.flux_tubes_hfs],
            [self.total_rad_lfs, self.total_rad_hfs],
            firstwall_geom,
        )


class RadiationSource:
    """
    Simplified solver to easily access the radiation model location inputs.

    Parameters
    ----------
    eq:
        The equilibrium defining flux surfaces.
    firstwall_shape:
        BluemiraWire defining the first wall.
    midplane_profiles:
        Electron density and electron temperature profile at the mid-plane.
    core_impurities:
        The dictionary of impurities in the core (e.g. 'H') and their fractions
        (e.g. 1E-2).
    sol_impurities:
        The dictionary of impurities in the scrape-off layer (e.g. 'H') and their
        fractions (e.g. 1E-2).
    confinement_time_sol:
        Confinement timescale at the scrape-off layer [s]
    confinement_time_core:
        Confinement timescale in the core [s]
    """

    params: RadiationSourceParams
    param_cls: type[RadiationSourceParams] = RadiationSourceParams

    def __init__(
        self,
        eq: Equilibrium,
        firstwall_shape: BluemiraWire,
        params: ParameterFrameLike,
        midplane_profiles: MidplaneProfiles,
        core_impurities: dict[str, float],
        sol_impurities: dict[str, float],
        confinement_time_core: float = np.inf,
        confinement_time_sol: float = 10,
    ):
        self.eq = eq
        self.params = make_parameter_frame(params, self.param_cls)

        # Get impurity data
        impurities_list_core = list(core_impurities)
        impurities_list_sol = list(sol_impurities)
        impurity_data_core = get_impurity_data(
            impurities_list=impurities_list_core,
            confinement_time_ms=confinement_time_core,
        )
        impurity_data_sol = get_impurity_data(
            impurities_list=impurities_list_sol, confinement_time_ms=confinement_time_sol
        )

        self.imp_content_core = core_impurities
        self.imp_data_core = impurity_data_core
        self.imp_content_sol = sol_impurities
        self.imp_data_sol = impurity_data_sol
        self.lcfs = self.eq.get_LCFS()

        self.midplane_profiles = midplane_profiles

        # To be calculated calling analyse
        self.core_rad = None
        self.sol_rad = None

        # To be calculated calling rad_map
        self.x_tot = None
        self.z_tot = None
        self.rad_tot = None

        # Initialising the `FluxSurfaceMaker`
        (
            self.dx_omp,
            self.dx_imp,
            self.flux_surfaces,
            self.x_sep_omp,
            self.x_sep_imp,
        ) = analyse_first_wall_flux_surfaces(
            equilibrium=eq, first_wall=firstwall_shape, dx_mp=0.001
        )

    def analyse(
        self, firstwall_geom: Coordinates
    ) -> tuple[CoreRadiation, ScrapeOffLayerRadiation]:
        """
        Using core radiation model and sol radiation model
        to calculate the radiation source at all points

        Parameters
        ----------
        firstwall_geom:
            The closed first wall geometry

        Returns
        -------
        self.core_rad:
            Core radiation source
        self.sol_rad:
            Scrape-off-layer radiation source
        """
        self.core_rad = CoreRadiation(
            self.eq,
            self.params,
            self.midplane_profiles,
            self.imp_content_core,
            self.imp_data_core,
        )

        if self.eq.is_double_null:
            self.sol_rad = DNScrapeOffLayerRadiation(
                self.eq,
                self.params,
                self.flux_surfaces,
                self.imp_content_sol,
                self.imp_data_sol,
                firstwall_geom,
                self.x_sep_omp,
                self.x_sep_imp,
                self.dx_omp,
                self.dx_imp,
            )
        else:
            self.sol_rad = SNScrapeOffLayerRadiation(
                self.eq,
                self.params,
                self.flux_surfaces,
                self.imp_content_sol,
                self.imp_data_sol,
                firstwall_geom,
                self.x_sep_omp,
                self.dx_omp,
            )

        return self.core_rad, self.sol_rad

    def rad_core_by_psi_n(self, psi_n):
        """
        Calculation of core radiation source for a given (set of) psi norm value(s)

        Parameters
        ----------
        psi_n: float (list)
            The normalised magnetic flux value(s)

        Returns
        -------
        rad_new: float (list)
            Local radiation source value(s) associated to the given psi_n
        """
        core_rad = CoreRadiation(
            self.eq,
            self.params,
            self.midplane_profiles,
            self.imp_content_core,
            self.imp_data_core,
        )
        core_rad.calculate_mp_radiation_profile()
        rad_tot = np.sum(np.array(core_rad.rad_mp, dtype=object), axis=0)
        return interp1d(self.midplane_profiles.psi_n, rad_tot)(psi_n)

    def rad_core_by_points(self, x, z):
        """
        Calculation of core radiation source for a given (set of) x, z coordinates

        Parameters
        ----------
        x:
            The x coordinate(s) of desired radiation source point(s)
        z:
            The z coordinate(s) of desired radiation source point(s)

        Returns
        -------
        self.rad_core_by_psi_n(psi_n):
            Local radiation source value(s) associated to the point(s)
        """
        psi = self.eq.psi(x, z)
        psi_n = calc_psi_norm(psi, *self.eq.get_OX_psis(psi))
        return self.rad_core_by_psi_n(psi_n)

    def rad_sol_by_psi_n(self, psi_n):
        """
        Calculation of sol radiation source for a given psi norm value

        Parameters
        ----------
        psi_n: float
            The normalised magnetic flux value

        Returns
        -------
        list
            Local radiation source values associated to the given psi_n
        """
        f_sol = linear_interpolator(self.x_tot, self.z_tot, self.rad_tot)
        fs = self.eq.get_flux_surface(psi_n)
        return np.concatenate([
            interpolated_field_values(x, z, f_sol)
            for x, z in zip(fs.x, fs.z, strict=False)
        ])

    def rad_sol_by_points(self, x_lst, z_lst):
        """
        Calculation of sol radiation source for a given (set of) x, z coordinates

        Parameters
        ----------
        x:
            The x coordinate(s) of desired radiation source point(s)
        z:
            The z coordinate(s) of desired radiation source point(s)

        Returns
        -------
        list
            Local radiation source value(s) associated to the point(s)
        """
        f_sol = linear_interpolator(self.x_tot, self.z_tot, self.rad_tot)
        return np.concatenate([
            interpolated_field_values(x, z, f_sol)
            for x, z in zip(x_lst, z_lst, strict=False)
        ])

    def rad_by_psi_n(self, psi_n):
        """
        Calculation of any radiation source for a given (set of) psi norm value(s)

        Parameters
        ----------
        psi_n:
            The normalised magnetic flux value(s)

        Returns
        -------
        rad_any:
            Local radiation source value(s) associated to the given psi_n
        """
        if psi_n < 1:
            return self.rad_core_by_psi_n(psi_n)
        return self.rad_sol_by_psi_n(psi_n)

    def rad_by_points(self, x, z):
        """
        Calculation of any radiation source for a given (set of) x, z coordinates

        Parameters
        ----------
        x:
            The x coordinate(s) of desired radiation source point(s)
        z:
            The z coordinate(s) of desired radiation source point(s)

        Returns
        -------
        rad_any:
            Local radiation source value(s) associated to the point(s)
        """
        f = linear_interpolator(self.x_tot, self.z_tot, self.rad_tot)
        return interpolated_field_values(x, z, f)

    def rad_map(self, firstwall_geom: Grid):
        """
        Mapping all the radiation values associated to all the points
        as three arrays containing x coordinates, z coordinates and
        local radiated power density [MW/m^3]

        Returns
        -------
        self.x_tot:
            x-coordinates of flux tubes [m]
        self.z_tot:
            z-coordinates of flux tubes [m]
        self.rad_tot:
            total radiated power density [MW/m^3]
        """
        self.core_rad.calculate_core_radiation_map()

        t_and_n_sol_profiles = self.sol_rad.calculate_sol_distribution(firstwall_geom)
        rad_sector_profiles = self.sol_rad.calculate_sol_radiation_distribution(
            **t_and_n_sol_profiles
        )
        self.sol_rad.calculate_sol_radiation_map(**rad_sector_profiles)

        self.x_tot = np.concatenate([self.core_rad.x_tot, self.sol_rad.x_tot])
        self.z_tot = np.concatenate([self.core_rad.z_tot, self.sol_rad.z_tot])
        self.rad_tot = np.concatenate([self.core_rad.rad_tot, self.sol_rad.rad_tot])

        return self.x_tot, self.z_tot, self.rad_tot

    def plot(self, ax=None) -> plt.Axes:
        """
        Plot the RadiationSolver results.

        Returns
        -------
        ax:
            The axes object on which radiation solver results are plotted.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        separatrix = self.eq.get_separatrix()
        if isinstance(separatrix, Coordinates):
            separatrix = [separatrix]

        for sep in separatrix:
            plot_coordinates(sep, ax=ax, linewidth=0.2)
        cm = ax.scatter(
            self.x_tot,
            self.z_tot,
            c=self.rad_tot,
            s=10,
            cmap="plasma",
            vmin=min(self.rad_tot),
            vmax=max(self.rad_tot),
            zorder=Zorder.RADIATION.value,
        )

        fig.colorbar(cm, label=r"$[MW.m^{-3}]$")

        return ax
