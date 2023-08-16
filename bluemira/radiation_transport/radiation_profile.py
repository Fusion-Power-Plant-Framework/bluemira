# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
1-D radiation model inspired by the PROCESS function "plot_radprofile" in plot_proc.py.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from dataclasses import dataclass, fields
import matplotlib.pyplot as plt
import numpy as np

from typing import Dict, Union
from bluemira.base import constants
from bluemira.base.constants import ureg
from bluemira.base.error import BuilderError
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.physics import calc_psi_norm
from bluemira.display.plotter import plot_coordinates
from bluemira.geometry.coordinates import Coordinates
from scipy.interpolate import interp1d

from bluemira.radiation_transport.radiation_tools import (
    gaussian_decay, 
    exponential_decay,
    radiative_loss_function_values,
    calculate_line_radiation_loss,
    radiative_loss_function_plot,
    electron_density_and_temperature_sol_decay,
    ion_front_distance,
    random_point_temperature,
    target_temperature,
    upstream_temperature,
    linear_interpolator,
    interpolated_field_values,
    
)

if TYPE_CHECKING:
    from radiation_transport.flux_surfaces_maker import FluxSurfaceMaker

@dataclass
class SeparationCorrector:
    DN: float = 5e-3
    SN: float = 5e-2

@dataclass
class RadiationParameterFrame(ParameterFrame):
    n_e_sep: Parameter[float] = Parameter(name="n_e_sep", long_name="Electron density at separatrix", value=3e19, unit='1/m^3', source="default")
    T_e_sep: Parameter[float] = Parameter(name="T_e_sep", long_name="Electron temperature at separatrix", value=2e-01, unit="keV", source="default")

@dataclass
class CoreRadiationParameterFrame(RadiationParameterFrame):
    n_e_0: Parameter[float] =  Parameter(name="n_e_0", long_name="Electron density on axis", value=1.81e+20, unit="1/m^3", source="default")
    T_e_0: Parameter[float] =  Parameter(name="T_e_0", long_name="Electron temperature on axis", value=2.196e+01, unit="keV", source="default")
    rho_ped_n: Parameter[float] =  Parameter(name="rho_ped_n", long_name="Density pedestal r/a location", value=9.4e-01, unit="dimensionless", source="default")
    rho_ped_t: Parameter[float] =  Parameter(name="rho_ped_t", long_name="Temperature pedestal r/a location", value=9.76e-01 , unit="dimensionless", source="default")
    n_e_ped: Parameter[float] =  Parameter(name="n_e_ped", long_name="Electron density pedestal height", value=1.086e+20, unit="1/m^3", source="default")
    T_e_ped: Parameter[float] =  Parameter(name="T_e_ped", long_name="Electron temperature pedestal height", value=3.74, unit="keV", source="default")
    alpha_n: Parameter[float] =  Parameter(name="alpha_n", long_name="Density profile factor", value=1.15, unit="dimensionless", source="default")
    alpha_t: Parameter[float] =  Parameter(name="alpha_t", long_name="Temperature profile index", value=1.905, unit="dimensionless", source="default")
    t_beta: Parameter[float] =  Parameter(name="t_beta", long_name="Temperature profile index beta", value=2, unit="dimensionless", source="default")

@dataclass
class SolRadiationParameterFrame(RadiationParameterFrame):
    P_sep: Parameter[float] =  Parameter(name='P_sep', long_name='Radiation power', value=150, unit='MW', source="default")
    fw_lambda_q_near_omp: Parameter[float] =  Parameter(name='fw_lambda_q_near_omp', long_name='Lambda_q near SOL omp', value=0.002, unit='m', source="default")
    fw_lambda_q_far_omp: Parameter[float] =  Parameter(name='fw_lambda_q_far_omp', long_name='Lambda_q far SOL omp', value=0.10, unit='m', source="default")
    fw_lambda_q_near_imp: Parameter[float] =  Parameter(name='fw_lambda_q_near_imp', long_name='Lambda_q near SOL imp', value=0.002, unit='m', source="default")
    fw_lambda_q_far_imp: Parameter[float] =  Parameter(name='fw_lambda_q_far_imp', long_name='Lambda_q far SOL imp', value=0.10, unit='m', source="default")
    k_0: Parameter[float] =  Parameter(name="k_0", long_name="material's conductivity", value=2000, unit="dimensionless", source="default")
    gamma_sheath: Parameter[float] =  Parameter(name="gamma_sheath", long_name="sheath heat transmission coefficient", value=7, unit="dimensionless", source="default")
    eps_cool: Parameter[float] =  Parameter(name="eps_cool", long_name="electron energy loss", value=25, unit="eV", source="default")
    f_ion_t: Parameter[float] =  Parameter(name="f_ion_t", long_name="Hydrogen first ionization", value=0.01, unit="keV", source="default")
    det_t: Parameter[float] =  Parameter(name="det_t", long_name="Detachment target temperature", value=0.0015, unit="keV", source="default")
    lfs_p_fraction: Parameter[float] =  Parameter(name="lfs_p_fraction", long_name="lfs fraction of SoL power", value=0.9, unit="dimensionless", source="default")
    theta_outer_target: Parameter[float] =  Parameter(name="theta_outer_target", long_name="Outer divertor poloidal angle with the separatrix flux line", value=5, unit="deg", source="default")
    theta_inner_target: Parameter[float] =  Parameter(name="theta_inner_target", long_name="Inner divertor poloidal angle with the separatrix flux line", value=5, unit="deg", source="default")

class Radiation:
    """
    Initial and generic class (no distinction between core and SOL)
    to calculate radiation source within the flux tubes.
    """

    def __init__(
        self,
        eq: Equilibrium,
        flux_surf_solver: FluxSurfaceMaker,
        params: Union[Dict, ParameterFrame],
    ):
        self.params = params
        
        self.flux_surf_solver = flux_surf_solver
        self.eq = eq

        # Constructors
        self.x_sep_omp = None
        self.x_sep_imp = None

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
            sep_corrector = SeparationCorrector.DN
        else:
            ob_ind = np.where(self.separatrix.x > self.points["x_point"]["x"])
            ib_ind = np.where(self.separatrix.x < self.points["x_point"]["x"])
            self.sep_ob = Coordinates({"x": self.separatrix.x[ob_ind], "z": self.separatrix.z[ob_ind]})
            self.sep_ib = Coordinates({"x": self.separatrix.x[ib_ind], "z": self.separatrix.z[ib_ind]})
            sep_corrector = SeparationCorrector.SN
        # The mid-plane radii
        self.x_sep_omp = self.flux_surf_solver.x_sep_omp
        # To move away from the mathematical separatrix which would
        # give infinite connection length
        self.r_sep_omp = self.x_sep_omp + sep_corrector
        # magnetic field components at the midplane
        self.b_pol_sep_omp = self.eq.Bp(self.x_sep_omp, self.z_mp)
        b_tor_sep_omp = self.eq.Bt(self.x_sep_omp)
        self.b_tot_sep_omp = np.hypot(self.b_pol_sep_omp, b_tor_sep_omp)

        if self.eq.is_double_null:
            self.x_sep_imp = self.flux_surf_solver.x_sep_imp
            self.r_sep_imp = self.x_sep_imp - sep_corrector
            self.b_pol_sep_imp = self.eq.Bp(self.x_sep_imp, self.z_mp)
            b_tor_sep_imp = self.eq.Bt(self.x_sep_imp)
            self.b_tot_sep_imp = np.hypot(self.b_pol_sep_imp, b_tor_sep_imp)

    def collect_flux_tubes(self, psi_n):
        """
        Collect flux tubes according to the normalised psi.
        For now only used for the core as for the SoL the
        flux surfaces to calculate the heat flux from charged
        particles are used.

        Parameters
        ----------
        psi_n: np.array
            normalised psi

        Returns
        -------
        flux tubes: list
            list of flux tubes
        """
        return [self.eq.get_flux_surface(psi) for psi in psi_n]

    def flux_tube_pol_t(
        self,
        flux_tube,
        te_mp,
        core=False,
        t_rad_in=None,
        t_rad_out=None,
        rad_i=None,
        rec_i=None,
        t_tar=None,
        x_point_rad=False,
        main_chamber_rad=False,
    ):
        """
        Along a single flux tube, it assigns different temperature values.

        Parameters
        ----------
        flux_tube: loop
            flux tube geometry
        te_mp: float
            electron temperature at the midplane [keV]
        core: boolean
            if True, t is constant along the flux tube. If false,it varies
        t_rad_in: float
            temperature value at the entrance of the radiation region [keV]
        t_rad_out: float
            temperature value at the exit of the radiation region [keV]
        rad_i: np.array
            indexes of points, belonging to the flux tube, which fall
            into the radiation region
        rec_i: np.array
            indexes of points, belonging to the flux tube, which fall
            into the recycling region
        t_tar: float
            electron temperature at the target [keV]
        main_chamber_rad: boolean
            if True, the temperature from the midplane to
            the radiation region entrance is not constant

        Returns
        -------
        te: np.array
            poloidal distribution of electron temperature
        """
        te = np.array([te_mp] * len(flux_tube))

        if core is True:
            return te

        if rad_i is not None and len(rad_i) == 1:
            te[rad_i] = t_rad_in
        elif rad_i is not None and len(rad_i) > 1:
            te[rad_i] = gaussian_decay(t_rad_in, t_rad_out, len(rad_i), decay=True)

        if rec_i is not None and x_point_rad:
            te[rec_i] = exponential_decay(
                t_rad_out * 0.95, t_tar, len(rec_i), decay=True
            )
        elif rec_i is not None and x_point_rad is False:
            te[rec_i] = t_tar

        if main_chamber_rad:
            mask = np.ones_like(te, dtype=bool)
            main_rad = np.concatenate((rad_i, rec_i))
            mask[main_rad] = False
            te[mask] = np.linspace(te_mp, t_rad_in, len(te[mask]))

        return te

    def flux_tube_pol_n(
        self,
        flux_tube,
        ne_mp,
        core=False,
        n_rad_in=None,
        n_rad_out=None,
        rad_i=None,
        rec_i=None,
        n_tar=None,
        main_chamber_rad=False,
    ):
        """
        Along a single flux tube, it assigns different density values.

        Parameters
        ----------
        flux_tube: loop
            flux tube geometry
        ne_mp: float
            electron density at the midplane [1/m^3]
        core: boolean
            if True, n is constant along the flux tube. If false,it varies
        n_rad_in: float
            density value at the entrance of the radiation region [1/m^3]
        n_rad_out: float
            density value at the exit of the radiation region [1/m^3]
        rad_i: np.array
            indexes of points, belonging to the flux tube, which fall
            into the radiation region
        rec_i: np.array
            indexes of points, belonging to the flux tube, which fall
            into the recycling region
        n_tar: float
            electron density at the target [1/m^3]
        main_chamber_rad: boolean
            if True, the temperature from the midplane to
            the radiation region entrance is not constant

        Returns
        -------
        ne: np.array
            poloidal distribution of electron density
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
        if rec_i is not None and len(rec_i) > 0 and len(rad_i) > 1:
            gap = ne[rad_i[-1]] - ne[rad_i[-2]]
            ne[rec_i] = gaussian_decay(n_rad_out - gap, n_tar, len(rec_i))

        if main_chamber_rad:
            mask = np.ones_like(ne, dtype=bool)
            main_rad = np.concatenate((rad_i, rec_i))
            mask[main_rad] = False
            ne[mask] = np.linspace(ne_mp, n_rad_in, len(ne[mask]))

        return ne

    def mp_profile_plot(self, rho, rad_power, imp_name, ax=None):
        """
        1D plot of the radation power distribution along the midplane.
        if rad_power.ndim > 1 the plot shows as many line as the number
        of impurities, plus the total radiated power

        Parameters
        ----------
        rho: np.array
            dimensionless radius. Values between 0 and 1 for the plasma core.
            Values > 1 for the scrape-off layer
        rad_power: np.array
            radiated power at each mid-plane location corresponding to rho [Mw/m^3]
        imp_name: [strings]
            impurity neames
        """
        if ax is None:
            fig, ax = plt.subplots()
            plt.xlabel(r"$\rho$")
            plt.ylabel(r"$[MW.m^{-3}]$")

        if len(rad_power) == 1:
            ax.plot(rho, rad_power)
        else:
            [
                ax.plot(rho, rad_part, label=name)
                for rad_part, name in zip(rad_power, imp_name)
            ]
            rad_tot = np.sum(np.array(rad_power, dtype=object), axis=0).tolist()
            plt.title("Core radiated power density")
            ax.plot(rho, rad_tot, label="total radiation")
            ax.legend(loc="best", borderaxespad=0, fontsize=12)

        return ax


class CoreRadiation(Radiation):
    """
    Specific class to calculate the core radiation source.
    Temperature and density are assumed to be constant along a
    single flux tube.
    """

    def __init__(
        self,
        eq: Equilibrium,
        flux_surf_solver: FluxSurfaceMaker,
        params: ParameterFrame,
        psi_n,
        ne_mp,
        te_mp,
        impurity_content,
        impurity_data,
    ):
        super().__init__(eq, flux_surf_solver, params)

        self.H_content = impurity_content["H"]
        self.impurities_content = [
            frac for key, frac in impurity_content.items() if key != "Ar"
        ]
        self.imp_data_t_ref = [
            data["T_ref"] for key, data in impurity_data.items() if key != "Ar"
        ]
        self.imp_data_l_ref = [
            data["L_ref"] for key, data in impurity_data.items() if key != "Ar"
        ]
        self.impurity_symbols = impurity_content.keys()

        # Midplane profiles
        self.psi_n = psi_n
        self.ne_mp = ne_mp
        self.te_mp = te_mp

    def build_mp_radiation_profile(self):
        """
        1D profile of the line radiation loss at the mid-plane
        through the scrape-off layer
        """
        # Radiative loss function values for each impurity species
        loss_f = [
            radiative_loss_function_values(self.te_mp, t_ref, l_ref)
            for t_ref, l_ref in zip(self.imp_data_t_ref, self.imp_data_l_ref)
        ]

        # Line radiation loss. Mid-plane distribution through the SoL
        self.rad_mp = [
            calculate_line_radiation_loss(self.ne_mp, loss, fi)
            for loss, fi in zip(loss_f, self.impurities_content)
        ]

    def plot_mp_radiation_profile(self):
        """
        Plot one dimensional behaviour of line radiation
        against the adimensional radius
        """
        self.mp_profile_plot(self.psi_n, self.rad_mp, self.impurity_symbols)

    def build_core_distribution(self):
        """
        Build poloidal distribution (distribution along the field lines) of
        line radiation loss in the plasma core.

        Returns
        -------
        rad: [np.array]
            Line core radiation.
            For specie and each closed flux line in the core
        """
        # Closed flux tubes within the separatrix
        self.flux_tubes = self.collect_flux_tubes(self.psi_n)

        # For each flux tube, poloidal density profile.
        self.ne_pol = [
            self.flux_tube_pol_n(ft, n, core=True)
            for ft, n in zip(self.flux_tubes, self.ne_mp)
        ]

        # For each flux tube, poloidal temperature profile.
        self.te_pol = [
            self.flux_tube_pol_t(ft, t, core=True)
            for ft, t in zip(self.flux_tubes, self.te_mp)
        ]

        # For each impurity species and for each flux tube,
        # poloidal distribution of the radiative power loss function.
        self.loss_f = [
            [radiative_loss_function_values(t, t_ref, l_ref) for t in self.te_pol]
            for t_ref, l_ref in zip(self.imp_data_t_ref, self.imp_data_l_ref)
        ]

        # For each impurity species and for each flux tube,
        # poloidal distribution of the line radiation loss.
        self.rad = [
            [
                calculate_line_radiation_loss(n, l_f, fi)
                for n, l_f in zip(self.ne_pol, ft)
            ]
            for ft, fi in zip(self.loss_f, self.impurities_content)
        ]

        return self.rad

    def build_core_radiation_map(self):
        """
        Build core radiation map.
        """
        rad = self.build_core_distribution()
        self.total_rad = np.sum(np.array(rad, dtype=object), axis=0).tolist()

        self.x_tot = np.concatenate([flux_tube.x for flux_tube in self.flux_tubes])
        self.z_tot = np.concatenate([flux_tube.z for flux_tube in self.flux_tubes])
        self.rad_tot = np.concatenate(self.total_rad)

    def radiation_distribution_plot(self, flux_tubes, power_density, ax=None):
        """
        2D plot of the core radation power distribution.

        Parameters
        ----------
        flux_tubes: np.array
            array of the closed flux tubes within the separatrix.
        power_density: np.array([np.array])
            arrays containing the power radiation density of the
            points lying on each flux tube [MW/m^3]
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        p_min = min([np.amin(p) for p in power_density])
        p_max = max([np.amax(p) for p in power_density])

        separatrix = self.eq.get_separatrix()
        if isinstance(separatrix, Coordinates):
            separatrix = [separatrix]

        for sep in separatrix:
            plot_coordinates(sep, ax=ax, linewidth=0.2)
        for flux_tube, p in zip(flux_tubes, power_density):
            cm = ax.scatter(
                flux_tube.x,
                flux_tube.z,
                c=p,
                s=10,
                cmap="plasma",
                vmin=p_min,
                vmax=p_max,
                zorder=40,
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
    """

    def collect_rho_sol_values(self):
        """
        Calculation of SoL dimensionless radial coordinate rho.

        Returns
        -------
        rho_sol: np.array
            dimensionless sol radius. Values higher than 1
        """
        # The dimensionless radius has plasma-core upper limit equal 1.
        # Such limit is the bottom one for the SoL
        r_o_point = self.points["o_point"]["x"]
        a = self.flux_surf_solver.x_sep_omp - r_o_point
        rho_sol = (a + self.flux_surf_solver.dx_omp) / a

        return rho_sol

    def x_point_radiation_z_ext(self, main_ext=None, pfr_ext=0.3, low_div=True):
        """
        Simple definition of a radiation region around the x-point.
        The region is supposed to extend from an arbitrary z coordinate on the
        main plasma side, to an arbitrary z coordinate on the private flux region side.

        Parameters
        ----------
        main_ext: float [m]
            region extension on the main plasma side
        pfr_ext: float [m]
            region extension on the private flux region side
        low_div: boolean
            default=True for the lower divertor. If False, upper divertor

        Returns
        -------
        z_main: float [m]
            vertical (z coordinate) extension of the radiation region
            toward the main plasma
        z_pfr: float [m]
            vertical (z coordinate) extension of the radiation region
            toward the pfr
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

    def radiation_region_ends(self, z_main, z_pfr, lfs=True):
        """
        Entering and exiting points (x, z) of the radiation region
        detected on the separatrix.
        The separatrix is supposed to be given by relevant half.

        Parameters
        ----------
        z_main: float [m]
            vertical (z coordinate) extension of the radiation region
            toward the main plasma
        z_pfr: float [m]
            vertical (z coordinate) extension of the radiation region
            toward the pfr
        lfs: boolean
            default=True for the low field side (right half).
            If False, high field side (left half).

        Returns
        -------
        entrance_x, entrance_z: float, float
            x, z coordinates of the radiation region starting point
        exit_x, exit_z: float, float
            x, z coordinates of the radiation region ending point
        """
        if self.eq.is_double_null:
            sep_loop = self.sep_lfs if lfs else self.sep_hfs
        else:
            sep_loop = self.sep_ob if lfs else self.sep_ib
        if z_main > z_pfr:
            reg_i = np.where((sep_loop.z < z_main) & (sep_loop.z > z_pfr))[0]
            i_in = np.where(sep_loop.z == np.max(sep_loop.z[reg_i]))[0]
            i_out = np.where(sep_loop.z == np.min(sep_loop.z[reg_i]))[0]
        else:
            reg_i = np.where((sep_loop.z > z_main) & (sep_loop.z < z_pfr))[0]
            i_in = np.where(sep_loop.z == np.min(sep_loop.z[reg_i]))[0]
            i_out = np.where(sep_loop.z == np.max(sep_loop.z[reg_i]))[0]

        entrance_x, entrance_z = sep_loop.x[i_in], sep_loop.z[i_in]
        exit_x, exit_z = sep_loop.x[i_out], sep_loop.z[i_out]

        return entrance_x[0], entrance_z[0], exit_x[0], exit_z[0]

    def radiation_region_points(self, flux_tube, z_main, z_pfr, lower=True):
        """
        For a given flux tube, indexes of points which fall respectively
        into the radiation and recycling region

        Parameters
        ----------
        flux_tube: loop
            flux tube geometry
        z_main: float [m]
            vertical (z coordinate) extension of the radiation region toward
            the main plasma.
            Taken on the separatrix
        z_pfr: float [m]
            vertical (z coordinate) extension of the radiation region
            toward the pfr.
            Taken on the separatrix
        lower: boolean
            default=True for the lower divertor. If False, upper divertor

        Returns
        -------
        rad_i: np.array
            indexes of the points within the radiation region
        rec_i: np.array
            indexes pf the points within the recycling region
        """
        if lower:
            rad_i = np.where((flux_tube.z < z_main) & (flux_tube.z > z_pfr))[0]
            rec_i = np.where(flux_tube.z < z_pfr)[0]
        else:
            rad_i = np.where((flux_tube.z > z_main) & (flux_tube.z < z_pfr))[0]
            rec_i = np.where(flux_tube.z > z_pfr)[0]

        return rad_i, rec_i

    def mp_electron_density_temperature_profiles(self, te_sep=None, omp=True):
        """
        Calculation of electron density and electron temperature profiles
        across the SoL at midplane.
        It uses the customised version for the mid-plane of the exponential
        decay law described in "electron_density_and_temperature_sol_decay".

        Parameters
        ----------
        te_sep: float
            electron temperature at the separatrix [keV]
        omp: boolean
            outer mid-plane. Default value True. If False it stands for inner mid-plane

        Returns
        -------
        te_sol: np.array
            radial decayed temperatures through the SoL at the mid-plane. Unit [keV]
        ne_sol: np.array
            radial decayed densities through the SoL at the mid-plane. Unit [1/m^3]
        """
        if omp or not self.eq.is_double_null:
            fw_lambda_q_near = self.params.fw_lambda_q_near_omp
            fw_lambda_q_far = self.params.fw_lambda_q_far_omp
            dx = self.flux_surf_solver.dx_omp
        else:
            fw_lambda_q_near = self.params.fw_lambda_q_near_imp
            fw_lambda_q_far = self.params.fw_lambda_q_far_imp
            dx = self.flux_surf_solver.dx_imp

        if te_sep is None:
            te_sep = self.params.T_e_sep
        ne_sep = self.params.n_e_sep

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
        lfs=True,
    ):
        """
        Calculation of electron density and electron temperature profiles
        across the SoL, starting from any point on the separatrix.
        (The z coordinate is the same. While the x coordinate changes)
        Using the equation to calculate T(s||).

        Parameters
        ----------
        x_p: float
            x coordinate of the point at the separatrix [m]
        z_p: float
            z coordinate of the point at the separatrix [m]
        t_p: float
            point temperature [keV]
        t_u: float
            upstream temperature [keV]
        lfs: boolean
            low (toroidal) field side (outer wall side). Default value True.
            If False it stands for high field side (hfs).

        Returns
        -------
        te_prof: np.array
            radial decayed temperatures through the SoL. Unit [keV]
        ne_prof: np.array
            radial decayed densities through the SoL. Unit [1/m^3]
        """
        # Distinction between lfs and hfs
        if lfs is True or not self.eq.is_double_null:
            r_sep_mp = self.r_sep_omp
            b_pol_sep_mp = self.b_pol_sep_omp
            fw_lambda_q_near = self.params.fw_lambda_q_near_omp
            fw_lambda_q_far = self.params.fw_lambda_q_far_omp
            dx = self.flux_surf_solver.dx_omp
        else:
            r_sep_mp = self.r_sep_imp
            b_pol_sep_mp = self.b_pol_sep_imp
            fw_lambda_q_near = self.params.fw_lambda_q_near_imp
            fw_lambda_q_far = self.params.fw_lambda_q_far_imp
            dx = self.flux_surf_solver.dx_imp

        # magnetic field components at the local point
        b_pol_p = self.eq.Bp(x_p, z_p)

        # flux expansion
        f_p = (r_sep_mp * b_pol_sep_mp) / (x_p * b_pol_p)

        # Ratio between upstream and local temperature
        f_t = t_u / t_p

        # Local electron density
        n_p = self.params.n_e_sep * f_t

        # Temperature and density profiles across the SoL
        te_prof, ne_prof = electron_density_and_temperature_sol_decay(
            t_p,
            n_p,
            fw_lambda_q_near,
            fw_lambda_q_far,
            dx,
            f_exp=f_p,
        )

        return te_prof, ne_prof

    def tar_electron_densitiy_temperature_profiles(
        self, ne_div, te_div, f_m=1, detachment=False
    ):
        """
        Calculation of electron density and electron temperature profiles
        across the SoL at the target.
        From the pressure balance, considering friction losses.
        f_m is the fractional loss of pressure due to friction.
        It can vary between 0 and 1.

        Parameters
        ----------
        ne_div: np.array
            density of the flux tubes at the entrance of the recycling region,
            assumed to be corresponding to the divertor plane [1/m^3]
        te_div: np.array
            temperature of the flux tubes at the entrance of the recycling region,
            assumed to be corresponding to the divertor plane [keV]
        f_m: float
            fractional loss factor

        Returns
        -------
        te_t: np.array
            target temperature [keV]
        ne_t: np.array
            target density [1/m^3]
        """
        if not detachment:
            te_t = te_div
            f_m = f_m
        else:
            te_t = [self.params.det_t] * len(te_div)
            f_m = 0.1
        ne_t = (f_m * ne_div) / 2

        return te_t, ne_t

    def build_sector_distributions(
        self,
        flux_tubes,
        x_strike,
        z_strike,
        main_ext,
        firstwall_geom,
        pfr_ext=None,
        rec_ext=None,
        x_point_rad=False,
        detachment=False,
        lfs=True,
        low_div=True,
        main_chamber_rad=False,
    ):
        """
        Temperature and density profiles builder.
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
        firstwall_geom: grid
            first wall geometry
        pfr_ext: float
            extention of the radiation region from the x-point
            towards private flux region [m]
        rec_ext: float
            extention of the recycling region,
            along the separatrix, from the target [m]
        x_point_rad: boolean
            if True, it assumes there is no radiation at all
            in the recycling region, and pfr_ext MUST be provided.
        detachment: boolean
            if True, it makes the temperature decay through the
            recycling region from the H ionization temperature to
            the assign temperature for detachment at the target.
            Else temperature is constant through the recycling region.
        lfs: boolean
            low field side. Default value True.
            If False it stands for high field side (hfs)
        low_div: boolean
            default=True for the lower divertor.
            If False, upper divertor
        main_chamber_rad: boolean
            if True, the temperature from the midplane to
            the radiation region entrance is not constant

        Returns
        -------
        t_pol: array
            temperature poloidal profile along each
            flux tube within the specified set [keV]
        n_pol: array
            density poloidal profile along each
            flux tube within the specified set [1/m^3]
        """
        # Validity condition for not x-point radiative
        if not x_point_rad and rec_ext is None:
            raise BuilderError("Required recycling region extention: rec_ext")
        elif not x_point_rad and rec_ext is not None and lfs:
            ion_front_z = ion_front_distance(
                x_strike,
                z_strike,
                self.flux_surf_solver.eq,
                self.points["x_point"]["z_low"],
                rec_ext=2,
            )
            pfr_ext = abs(ion_front_z)

        elif not x_point_rad and rec_ext is not None and lfs is False:
            ion_front_z = ion_front_distance(
                x_strike,
                z_strike,
                self.flux_surf_solver.eq,
                self.points["x_point"]["z_low"],
                rec_ext=0.4,
            )
            pfr_ext = abs(ion_front_z)

        # Validity condition for x-point radiative
        elif x_point_rad and pfr_ext is None:
            raise BuilderError("Required extention towards pfr: pfr_ext")

        # setting radiation and recycling regions
        z_main, z_pfr = self.x_point_radiation_z_ext(main_ext, pfr_ext, low_div)

        in_x, in_z, out_x, out_z = self.radiation_region_ends(z_main, z_pfr, lfs)

        reg_i = [
            self.radiation_region_points(f.coords, z_main, z_pfr, low_div)
            for f in flux_tubes
        ]

        # mid-plane parameters
        if lfs or not self.eq.is_double_null:
            t_u_kev = self.t_omp
            b_pol_tar = self.b_pol_out_tar
            b_pol_u = self.b_pol_sep_omp
            r_sep_mp = self.r_sep_omp
            # alpha = self.params.theta_outer_target
            alpha = self.alpha_lfs
            b_tot_tar = self.b_tot_out_tar
            fw_lambda_q_near = self.params.fw_lambda_q_near_omp
        else:
            t_u_kev = self.t_imp
            b_pol_tar = self.b_pol_inn_tar
            b_pol_u = self.b_pol_sep_imp
            r_sep_mp = self.r_sep_imp
            # alpha = self.params.theta_inner_target
            alpha = self.alpha_hfs
            b_tot_tar = self.b_tot_inn_tar
            fw_lambda_q_near = self.params.fw_lambda_q_near_imp

        # Coverting needed parameter units
        t_u_ev = constants.raw_uc(t_u_kev, "keV", "eV")
        p_sol = constants.raw_uc(self.params.P_sep, "MW", "W")
        f_ion_t = constants.raw_uc(self.params.f_ion_t, "keV", "eV")

        if lfs and self.eq.is_double_null:
            p_sol = p_sol*self.params.lfs_p_fraction
        elif not lfs and self.eq.is_double_null:
            p_sol = p_sol*(1-self.params.lfs_p_fraction)

        t_mp_prof, n_mp_prof = self.mp_electron_density_temperature_profiles(
            t_u_kev, lfs
        )

        # entrance of radiation region
        t_rad_in = random_point_temperature(
            in_x,
            in_z,
            t_u_ev,
            p_sol,
            fw_lambda_q_near,
            self.eq,
            r_sep_mp,
            self.points["o_point"]["z"],
            self.params.k_0,
            firstwall_geom,
            lfs,
        )

        # exit of radiation region
        if x_point_rad and pfr_ext is not None:
            t_rad_out = self.params.f_ion_t
        elif detachment:
            t_rad_out = self.params.f_ion_t
        else:
            t_rad_out = target_temperature(
                p_sol,
                t_u_ev,
                self.params.n_e_sep,
                self.params.gamma_sheath,
                self.params.eps_cool,
                f_ion_t,
                b_pol_tar,
                b_pol_u,
                alpha,
                r_sep_mp,
                x_strike,
                fw_lambda_q_near,
                b_tot_tar,
            )

        # condition for occurred detachment
        if t_rad_out <= self.params.f_ion_t:
            x_point_rad = True
            detachment = True

        # profiles through the SoL
        t_in_prof, n_in_prof = self.any_point_density_temperature_profiles(
            in_x,
            in_z,
            t_rad_in,
            t_u_kev,
            lfs,
        )

        t_out_prof, n_out_prof = self.any_point_density_temperature_profiles(
            out_x,
            out_z,
            t_rad_out,
            t_u_kev,
            lfs,
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
            for f, t, t_in, t_out, reg, t_t, in zip(
                flux_tubes,
                t_mp_prof,
                t_in_prof,
                t_out_prof,
                reg_i,
                t_tar_prof,
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
            for f, n, n_in, n_out, reg, n_t, in zip(
                flux_tubes,
                n_mp_prof,
                n_in_prof,
                n_out_prof,
                reg_i,
                n_tar_prof,
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
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        tubes = sum(flux_tubes, [])
        power = sum(power_density, [])

        p_min = min([min(p) for p in power])
        p_max = max([max(p) for p in power])
        
        plot_coordinates(firstwall, ax=ax, linewidth=0.5, fill=False)
        separatrix = self.eq.get_separatrix()
        if isinstance(separatrix, Coordinates):
            separatrix = [separatrix]

        for sep in separatrix:
            plot_coordinates(sep, ax=ax, linewidth=0.2)
        for flux_tube, p in zip(tubes, power):
            cm = ax.scatter(
                flux_tube.coords.x,
                flux_tube.coords.z,
                c=p,
                s=10,
                marker=".",
                cmap="plasma",
                vmin=p_min,
                vmax=p_max,
                zorder=40,
            )

        fig.colorbar(cm, label=r"$[MW.m^{-3}]$")

        return ax

    def poloidal_distribution_plot(
        self, flux_tubes, property, temperature=True, ax=None
    ):
        """
        2D plot of a generic property (density, temperature or radiation)
        as poloidal section the flux tube points.

        Parameters
        ----------
        flux_tubes: np.array
            array of the open flux tubes within the SoL.
        property: np.array([np.array])
            arrays containing the property values associated
            to the points of each flux tube.
        """
        if ax is None:
            _, ax = plt.subplots()
        else:
            _ = ax.figure
        plt.xlabel("Mid-Plane to Target")
        if temperature is True:
            plt.title("Temperature along flux surfaces")
            plt.ylabel(r"$T_e~[keV]$")
        else:
            plt.title("Density along flux surfaces")
            plt.ylabel(r"$n_e~[m^{-3}]$")

        [
            ax.plot(np.linspace(0, len(flux_tube.coords.x), len(flux_tube.coords.x)), val)
            for flux_tube, val in zip(flux_tubes, property)
        ]

        return ax

    def plot_t_vs_n(
        self, flux_tube, t_distribution, n_distribution, rad_distribution=None, ax1=None
    ):
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
    Specific class to build the SOL radiation source for a double null configuration.
    Here the SOL is divided into for regions. From the outer midplane to the
    outer lower target; from the omp to the outer upper target; from the inboard
    midplane to the inner lower target; from the imp to the inner upper target.
    """

    def __init__(
        self,
        eq: Equilibrium,
        flux_surf_solver: FluxSurfaceMaker,
        params: ParameterFrame,
        impurity_content,
        impurity_data,
        firstwall_geom,
    ):
        super().__init__(eq, flux_surf_solver, params)

        self.impurities_content = [
            frac for key, frac in impurity_content.items() if key != "H"
        ]

        self.imp_data_t_ref = [
            data["T_ref"] for key, data in impurity_data.items() if key != "H"
        ]
        self.imp_data_l_ref = [
            data["L_ref"] for key, data in impurity_data.items() if key != "H"
        ]
        self.eq=eq
        # Flux tubes from the particle solver
        # partial flux tube from the mp to the target at the
        # outboard and inboard - lower divertor
        self.flux_tubes_lfs_low = self.flux_surf_solver.flux_surfaces_ob_lfs
        self.flux_tubes_hfs_low = self.flux_surf_solver.flux_surfaces_ib_lfs

        # partial flux tube from the mp to the target at the
        # outboard and inboard - upper divertor
        self.flux_tubes_lfs_up = self.flux_surf_solver.flux_surfaces_ob_hfs
        self.flux_tubes_hfs_up = self.flux_surf_solver.flux_surfaces_ib_hfs

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

        p_sol = constants.raw_uc(self.params.P_sep, "MW", "W")
        p_sol_lfs = p_sol*self.params.lfs_p_fraction
        p_sol_hfs = p_sol*(1-self.params.lfs_p_fraction)

        # upstream temperature and power density
        self.t_omp = upstream_temperature(
            b_pol=self.b_pol_sep_omp,
            b_tot=self.b_tot_sep_omp,
            lambda_q_near=self.params.fw_lambda_q_near_omp,
            p_sol=p_sol_lfs,
            eq=self.eq,
            r_sep_mp=self.r_sep_omp,
            z_mp=self.z_mp,
            k_0=self.params.k_0,
            firstwall_geom=firstwall_geom,
        )

        self.t_imp = upstream_temperature(
            b_pol=self.b_pol_sep_imp,
            b_tot=self.b_tot_sep_imp,
            lambda_q_near=self.params.fw_lambda_q_near_imp,
            p_sol=p_sol_hfs,
            eq=self.eq,
            r_sep_mp=self.r_sep_imp,
            z_mp=self.z_mp,
            k_0=self.params.k_0,
            firstwall_geom=firstwall_geom,
        )

    def build_sol_distribution(self, firstwall_geom: Grid):
        """
        Temperature and density profiles builder.
        For each scrape-off layer sector, it gives temperature
        and density profile along each flux tube.

        Parameters
        ----------
        firstwall_geom: grid
            first wall geometry

        Returns
        -------
        t_and_n_pol["lfs_low"]: array
            temperature and density poloidal profile along each
            flux tube within the lfs lower divertor set
        t_and_n_pol["lfs_up"]: array
            temperature and density poloidal profile along each
            flux tube within the lfs upper divertor set
        t_and_n_pol["hfs_low"]: array
            temperature and density poloidal profile along each
            flux tube within the hfs lower divertor set
        t_and_n_pol["hfs_up"]: array
            temperature and density poloidal profile along each
            flux tube within the hfs upper divertor set
        """
        t_and_n_pol_inputs = {
            f"{side}_{low_up}": {
                "flux_tubes": getattr(self, f"flux_tubes_{side}_{low_up}"),
                "x_strike": getattr(self, f"x_strike_{side}"),
                "z_strike": getattr(self, f"z_strike_{side}"),
                "main_ext": None,
                "firstwall_geom": firstwall_geom,
                "pfr_ext": None,
                "rec_ext": 2,
                "x_point_rad": False,
                "detachment": False,
                "lfs": side == "lfs",
                "low_div": low_up == "low",
                "main_chamber_rad": True,
            }
            for side in ["lfs", "hfs"]
            for low_up in ["low", "up"]
        }

        self.t_and_n_pol = {}
        for side, var in t_and_n_pol_inputs.items():
            self.t_and_n_pol[side] = self.build_sector_distributions(**var)

        return self.t_and_n_pol

    def build_sol_radiation_distribution(
        self,
        t_and_n_pol_lfs_low,
        t_and_n_pol_lfs_up,
        t_and_n_pol_hfs_low,
        t_and_n_pol_hfs_up,
    ):
        """
        Radiation profiles builder.
        For each scrape-off layer sector, it gives the
        radiation profile along each flux tube.

        Parameters
        ----------
        t_and_n_pol_lfs_low: array
            temperature and density poloidal profile along each
            flux tube within the lfs lower divertor set
        t_and_n_pol_lfs_up: array
            temperature and density poloidal profile along each
            flux tube within the lfs upper divertor set
        t_and_n_pol_hfs_low: array
            temperature and density poloidal profile along each
            flux tube within the hfs lower divertor set
        t_and_n_pol_hfs_up: array
            temperature and density poloidal profile along each
            flux tube within the hfs upper divertor set

        Returns
        -------
        rad["lfs_low"]: array
            radiation poloidal profile along each
            flux tube within the lfs lower divertor set
        rad["lfs_up"]: array
            radiation poloidal profile along each
            flux tube within the lfs upper divertor set
        rad["hfs_low"]: array
            radiation poloidal profile along each
            flux tube within the hfs lower divertor set
        rad["hfs_up"]: array
            radiation poloidal profile along each
            flux tube within the hfs upper divertor set
        """
        # For each impurity species and for each flux tube,
        # poloidal distribution of the radiative power loss function.
        # Values along the open flux tubes
        loss = {
            "lfs_low": t_and_n_pol_lfs_low[0],
            "lfs_up": t_and_n_pol_lfs_up[0],
            "hfs_low": t_and_n_pol_hfs_low[0],
            "hfs_up": t_and_n_pol_hfs_up[0],
        }

        for side, t_pol in loss.items():
            loss[side] = [
                [radiative_loss_function_values(t, t_ref, l_ref) for t in t_pol]
                for t_ref, l_ref in zip(self.imp_data_t_ref, self.imp_data_l_ref)
            ]

        # For each impurity species and for each flux tube,
        # poloidal distribution of the line radiation loss.
        # Values along the open flux tubes
        self.rad = {
            "lfs_low": {"density": t_and_n_pol_lfs_low[1], "loss": loss["lfs_low"]},
            "lfs_up": {"density": t_and_n_pol_lfs_up[1], "loss": loss["lfs_up"]},
            "hfs_low": {"density": t_and_n_pol_hfs_low[1], "loss": loss["hfs_low"]},
            "hfs_up": {"density": t_and_n_pol_hfs_up[1], "loss": loss["hfs_up"]},
        }
        for side, ft in self.rad.items():
            self.rad[side] = [
                [
                    calculate_line_radiation_loss(n, l_f, fi)
                    for n, l_f in zip(ft["density"], f)
                ]
                for f, fi in zip(ft["loss"], self.impurities_content)
            ]
        return self.rad

    def build_sol_radiation_map(self, rad_lfs_low, rad_lfs_up, rad_hfs_low, rad_hfs_up):
        """
        Scrape off layer radiation map builder.

        Parameters
        ----------
        rad["lfs_low"]: array
            radiation poloidal profile along each
            flux tube within the lfs lower divertor set
        rad["lfs_up"]: array
            radiation poloidal profile along each
            flux tube within the lfs upper divertor set
        rad["hfs_low"]: array
            radiation poloidal profile along each
            flux tube within the hfs lower divertor set
        rad["hfs_up"]: array
            radiation poloidal profile along each
            flux tube within the hfs upper divertor set
        firstwall_geom: grid
            first wall geometry
        """
        # total line radiation loss along the open flux tubes
        self.total_rad_lfs_low = np.sum(
            np.array(rad_lfs_low, dtype=object), axis=0
        ).tolist()
        self.total_rad_lfs_up = np.sum(
            np.array(rad_lfs_up, dtype=object), axis=0
        ).tolist()
        self.total_rad_hfs_low = np.sum(
            np.array(rad_hfs_low, dtype=object), axis=0
        ).tolist()
        self.total_rad_hfs_up = np.sum(
            np.array(rad_hfs_up, dtype=object), axis=0
        ).tolist()

        rads = [
            self.total_rad_lfs_low,
            self.total_rad_hfs_low,
            self.total_rad_lfs_up,
            self.total_rad_hfs_up,
        ]
        power = sum(rads, [])

        flux_tubes = [
            self.flux_surf_solver.flux_surfaces_ob_lfs,
            self.flux_surf_solver.flux_surfaces_ib_lfs,
            self.flux_surf_solver.flux_surfaces_ob_hfs,
            self.flux_surf_solver.flux_surfaces_ib_hfs,
        ]
        flux_tubes = sum(flux_tubes, [])
    
        self.x_tot = np.concatenate([flux_tube.coords.x for flux_tube in flux_tubes])
        self.z_tot = np.concatenate([flux_tube.coords.z for flux_tube in flux_tubes])
        self.rad_tot = np.concatenate(power)

    def plot_poloidal_radiation_distribution(self, firstwall_geom: Grid):
        """
        Plot poloiadal radiation distribution
        within the scrape-off layer

        Parameters
        ----------
        firstwall: Grid
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
    Specific class to build the SOL radiation source for a double null configuration.
    Here the SOL is divided into for regions. From the outer midplane to the
    outer lower target; from the omp to the outer upper target; from the inboard
    midplane to the inner lower target; from the imp to the inner upper target.
    """

    def __init__(
        self,
        eq: Equilibrium,
        flux_surf_solver: FluxSurfaceMaker,
        params: ParameterFrame,
        impurity_content,
        impurity_data,
        firstwall_geom,
    ):
        super().__init__(eq, flux_surf_solver, params)

        self.impurities_content = [
            frac for key, frac in impurity_content.items() if key != "H"
        ]

        self.imp_data_t_ref = [
            data["T_ref"] for key, data in impurity_data.items() if key != "H"
        ]
        self.imp_data_l_ref = [
            data["L_ref"] for key, data in impurity_data.items() if key != "H"
        ]
        self.eq=eq
        # Flux tubes from the particle solver
        # partial flux tube from the mp to the target at the
        # outboard and inboard - lower divertor
        self.flux_tubes_lfs = self.flux_surf_solver.flux_surfaces_ob_lfs
        self.flux_tubes_hfs = self.flux_surf_solver.flux_surfaces_ob_hfs

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

        p_sol = constants.raw_uc(self.params.P_sep, "MW", "W")

        # upstream temperature and power density
        self.t_omp = upstream_temperature(
            b_pol=self.b_pol_sep_omp,
            b_tot=self.b_tot_sep_omp,
            lambda_q_near=self.params.fw_lambda_q_near_omp,
            p_sol=p_sol,
            eq=self.eq,
            r_sep_mp=self.r_sep_omp,
            z_mp=self.points["o_point"]["z"],
            k_0=self.params.k_0,
            firstwall_geom=firstwall_geom,
        )

    def build_sol_distribution(self, firstwall_geom: Grid):
        """
        Temperature and density profiles builder.
        For each scrape-off layer sector, it gives temperature
        and density profile along each flux tube.

        Parameters
        ----------
        firstwall_geom: grid
            first wall geometry

        Returns
        -------
        t_and_n_pol["lfs_low"]: array
            temperature and density poloidal profile along each
            flux tube within the lfs lower divertor set
        t_and_n_pol["lfs_up"]: array
            temperature and density poloidal profile along each
            flux tube within the lfs upper divertor set
        t_and_n_pol["hfs_low"]: array
            temperature and density poloidal profile along each
            flux tube within the hfs lower divertor set
        t_and_n_pol["hfs_up"]: array
            temperature and density poloidal profile along each
            flux tube within the hfs upper divertor set
        """
        t_and_n_pol_inputs = {
            f"{side}": {
                "flux_tubes": getattr(self, f"flux_tubes_{side}"),
                "x_strike": getattr(self, f"x_strike_{side}"),
                "z_strike": getattr(self, f"z_strike_{side}"),
                "main_ext": 1,
                "firstwall_geom": firstwall_geom,
                "pfr_ext": None,
                "rec_ext": 2,
                "x_point_rad": False,
                "detachment": False,
                "lfs": side == "lfs",
                "low_div": True,
                "main_chamber_rad": True,
            }
            for side in ["lfs", "hfs"]
        }

        self.t_and_n_pol = {}
        for side, var in t_and_n_pol_inputs.items():
            self.t_and_n_pol[side] = self.build_sector_distributions(**var)

        return self.t_and_n_pol

    def build_sol_radiation_distribution(
        self,
        t_and_n_pol_lfs,
        t_and_n_pol_hfs,
    ):
        """
        Radiation profiles builder.
        For each scrape-off layer sector, it gives the
        radiation profile along each flux tube.

        Parameters
        ----------
        t_and_n_pol_lfs_low: array
            temperature and density poloidal profile along each
            flux tube within the lfs lower divertor set
        t_and_n_pol_lfs_up: array
            temperature and density poloidal profile along each
            flux tube within the lfs upper divertor set
        t_and_n_pol_hfs_low: array
            temperature and density poloidal profile along each
            flux tube within the hfs lower divertor set
        t_and_n_pol_hfs_up: array
            temperature and density poloidal profile along each
            flux tube within the hfs upper divertor set

        Returns
        -------
        rad["lfs_low"]: array
            radiation poloidal profile along each
            flux tube within the lfs lower divertor set
        rad["lfs_up"]: array
            radiation poloidal profile along each
            flux tube within the lfs upper divertor set
        rad["hfs_low"]: array
            radiation poloidal profile along each
            flux tube within the hfs lower divertor set
        rad["hfs_up"]: array
            radiation poloidal profile along each
            flux tube within the hfs upper divertor set
        """
        # For each impurity species and for each flux tube,
        # poloidal distribution of the radiative power loss function.
        # Values along the open flux tubes
        loss = {
            "lfs": t_and_n_pol_lfs[0],
            "hfs": t_and_n_pol_hfs[0],
        }

        for side, t_pol in loss.items():
            loss[side] = [
                [radiative_loss_function_values(t, t_ref, l_ref) for t in t_pol]
                for t_ref, l_ref in zip(self.imp_data_t_ref, self.imp_data_l_ref)
            ]

        # For each impurity species and for each flux tube,
        # poloidal distribution of the line radiation loss.
        # Values along the open flux tubes
        self.rad = {
            "lfs": {"density": t_and_n_pol_lfs[1], "loss": loss["lfs"]},
            "hfs": {"density": t_and_n_pol_hfs[1], "loss": loss["hfs"]},
        }
        for side, ft in self.rad.items():
            self.rad[side] = [
                [
                    calculate_line_radiation_loss(n, l_f, fi)
                    for n, l_f in zip(ft["density"], f)
                ]
                for f, fi in zip(ft["loss"], self.impurities_content)
            ]
        return self.rad

    def build_sol_radiation_map(self, rad_lfs, rad_hfs):
        """
        Scrape off layer radiation map builder.

        Parameters
        ----------
        rad["lfs_low"]: array
            radiation poloidal profile along each
            flux tube within the lfs lower divertor set
        rad["lfs_up"]: array
            radiation poloidal profile along each
            flux tube within the lfs upper divertor set
        rad["hfs_low"]: array
            radiation poloidal profile along each
            flux tube within the hfs lower divertor set
        rad["hfs_up"]: array
            radiation poloidal profile along each
            flux tube within the hfs upper divertor set
        firstwall_geom: grid
            first wall geometry
        """
        # total line radiation loss along the open flux tubes
        self.total_rad_lfs = np.sum(
            np.array(rad_lfs, dtype=object), axis=0
        ).tolist()
        self.total_rad_hfs = np.sum(
            np.array(rad_hfs, dtype=object), axis=0
        ).tolist()

        rads = [
            self.total_rad_lfs,
            self.total_rad_hfs,
        ]
        power = sum(rads, [])

        flux_tubes = [
            self.flux_tubes_lfs,
            self.flux_tubes_hfs,
        ]
        flux_tubes = sum(flux_tubes, [])

        self.x_tot = np.concatenate([flux_tube.coords.x for flux_tube in flux_tubes])
        self.z_tot = np.concatenate([flux_tube.coords.z for flux_tube in flux_tubes])
        self.rad_tot = np.concatenate(power)

    def plot_poloidal_radiation_distribution(self, firstwall_geom: Grid):
        """
        Plot poloiadal radiation distribution
        within the scrape-off layer

        Parameters
        ----------
        firstwall: Grid
            first wall geometry
        """
        self.radiation_distribution_plot(
            [
                self.flux_tubes_lfs,
                self.flux_tubes_hfs,
            ],
            [
                self.total_rad_lfs,
                self.total_rad_hfs,
            ],
            firstwall_geom,
        )


class RadiationSolver:
    """
    Simplified solver to easily access the radiation model location inputs.
    """

    def __init__(
        self,
        eq: Equilibrium,
        flux_surf_solver: FluxSurfaceMaker,
        params: ParameterFrame,
        psi_n,
        ne_mp,
        te_mp,
        impurity_content_core,
        impurity_data_core,
        impurity_content_sol,
        impurity_data_sol,
    ):

        self.eq = eq
        self.flux_surf_solver = flux_surf_solver
        self.params = self._make_params(params)
        self.imp_content_core = impurity_content_core
        self.imp_data_core = impurity_data_core
        self.imp_content_sol = impurity_content_sol
        self.imp_data_sol = impurity_data_sol
        self.lcfs = self.eq.get_LCFS()

        # Midplane parameters
        self.psi_n = psi_n
        self.ne_mp = ne_mp
        self.te_mp = te_mp

        # To be calculated calling analyse
        self.core_rad = None
        self.sol_rad = None

        # To be calculated calling rad_map
        self.x_tot = None
        self.z_tot = None
        self.rad_tot = None

    def analyse(self, firstwall_geom: Grid):
        """
        Using core radiation model and sol radiation model
        to calculate the radiation source at all points

        Parameters
        ----------
        first_wall: Loop
            The closed first wall geometry

        Returns
        -------
        x_all: np.array
            The x coordinates of all the points included within the flux surfaces
        z_all: np.array
            The z coordinates of all the points included within the flux surfaces
        rad_all: np.array
            The local radiation source at all points included within
            the flux surfaces [MW/m^3]
        """
        self.core_rad = CoreRadiation(
            self.eq,
            self.flux_surf_solver,
            self.params,
            self.psi_n,
            self.ne_mp,
            self.te_mp,
            self.imp_content_core,
            self.imp_data_core,
        )

        if self.eq.is_double_null:
            self.sol_rad = DNScrapeOffLayerRadiation(
                self.eq,
                self.flux_surf_solver,
                self.params,
                self.imp_content_sol,
                self.imp_data_sol,
                firstwall_geom,
            )
        else:
            self.sol_rad = SNScrapeOffLayerRadiation(
                self.eq,
                self.flux_surf_solver,
                self.params,
                self.imp_content_sol,
                self.imp_data_sol,
                firstwall_geom,
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
            self.flux_surf_solver,
            self.params,
            self.psi_n,
            self.ne_mp,
            self.te_mp,
            self.imp_content_core,
            self.imp_data_core,
        )
        core_rad.build_mp_rad_profile()
        rad_tot = np.sum(np.array(core_rad.rad, dtype=object), axis=0)
        f_rad = interp1d(core_rad.rho_core, rad_tot)
        return f_rad(np.sqrt(psi_n))

    def rad_core_by_points(self, x, z):
        """
        Calculation of core radiation source for a given (set of) x, z coordinates

        Parameters
        ----------
        x: float (list)
            The x coordinate(s) of desired radiation source point(s)
        z: float(list)
            The z coordinate(s) of desired radiation source point(s)

        Returns
        -------
        self.rad_core_by_psi_n(psi_n): float (list)
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
        f_sol = linear_interpolator(self.x_sol, self.z_sol, self.rad_sol)
        fs = self.eq.get_flux_surface(psi_n)
        return np.concatenate(
            [interpolated_field_values(x, z, f_sol) for x, z in zip(fs.x, fs.z)]
        )

    def rad_sol_by_points(self, x_lst, z_lst):
        """
        Calculation of sol radiation source for a given (set of) x, z coordinates

        Parameters
        ----------
        x: float (list)
            The x coordinate(s) of desired radiation source point(s)
        z: float(list)
            The z coordinate(s) of desired radiation source point(s)

        Returns
        -------
        list
            Local radiation source value(s) associated to the point(s)
        """
        f_sol = linear_interpolator(self.x_tot, self.z_tot, self.rad_tot)
        return np.concatenate(
            [interpolated_field_values(x, z, f_sol) for x, z in zip(x_lst, z_lst)]
        )

    def rad_by_psi_n(self, psi_n):
        """
        Calculation of any radiation source for a given (set of) psi norm value(s)

        Parameters
        ----------
        psi_n: float (list)
            The normalised magnetic flux value(s)

        Returns
        -------
        rad_any: float (list)
            Local radiation source value(s) associated to the given psi_n
        """
        if psi_n < 1:
            return self.rad_core_by_psi_n(psi_n)
        else:
            return self.rad_sol_by_psi_n(psi_n)

    def rad_by_points(self, x, z):
        """
        Calculation of any radiation source for a given (set of) x, z coordinates

        Parameters
        ----------
        x: float (list)
            The x coordinate(s) of desired radiation source point(s)
        z: float(list)
            The z coordinate(s) of desired radiation source point(s)

        Returns
        -------
        rad_any: float (list)
            Local radiation source value(s) associated to the point(s)
        """
        f = linear_interpolator(self.x_tot, self.z_tot, self.rad_tot)
        return interpolated_field_values(x, z, f)

    def rad_map(self, firstwall_geom: Grid):
        """
        Mapping all the radiation values associated to all the points
        as three arrays containing x coordinates, z coordinates and
        local radiated power density [MW/m^3]
        """
        self.core_rad.build_core_radiation_map()

        t_and_n_sol_profiles = self.sol_rad.build_sol_distribution(firstwall_geom)
        rad_sector_profiles = self.sol_rad.build_sol_radiation_distribution(
            *t_and_n_sol_profiles.values()
        )
        self.sol_rad.build_sol_radiation_map(*rad_sector_profiles.values())

        self.x_tot = np.concatenate([self.core_rad.x_tot, self.sol_rad.x_tot])
        self.z_tot = np.concatenate([self.core_rad.z_tot, self.sol_rad.z_tot])
        self.rad_tot = np.concatenate([self.core_rad.rad_tot, self.sol_rad.rad_tot])

        return self.x_tot, self.z_tot, self.rad_tot

    def plot(self, ax=None):
        """
        Plot the RadiationSolver results.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        p_min = min(self.rad_tot)
        p_max = max(self.rad_tot)

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
            vmin=p_min,
            vmax=p_max,
            zorder=40,
        )

        fig.colorbar(cm, label=r"$[MW.m^{-3}]$")

        return ax
    
    def _make_params(self, config):
        """Convert the given params to ``RadiationSolverParams``"""
        if isinstance(config, dict):
            try:
                return RadiationSolverParams(**config)
            except TypeError:
                unknown = [
                    k for k in config if k not in fields(RadiationSolverParams)
                ]
                raise TypeError(f"Unknown config parameter(s) {str(unknown)[1:-1]}")
        elif isinstance(config, RadiationSolverParams):
            return config
        else:
            raise TypeError(
                "Unsupported type: 'config' must be a 'dict', or "
                "'ChargedParticleSolverParams' instance; found "
                f"'{type(config).__name__}'."
            )
        
@dataclass
class RadiationSolverParams:
    rho_ped_n: float = 0.94
    """???"""

    n_e_0: float = 21.93e19
    """???"""

    n_e_ped: float = 8.117e19
    """???"""

    n_e_sep: float = 1.623e19
    """???"""

    alpha_n: float = 1.15
    """???"""

    rho_ped_t: float = 0.976
    """???"""

    T_e_0: float = 21.442
    """???"""

    T_e_ped: float = 5.059
    """???"""

    T_e_sep: float = 0.16
    """???"""

    alpha_t: float = 1.905
    """???"""

    t_beta: float = 2.0
    """???"""

    P_sep: float = 150
    """???"""

    k_0: float = 2000.0
    """???"""

    gamma_sheath: float = 7.0
    """???"""

    eps_cool: float = 25.0
    """???"""

    f_ion_t: float = 0.01
    """???"""

    det_t: float = 0.0015
    """???"""

    lfs_p_fraction: float = 0.9
    """???"""

    div_p_sharing: float = 0.5
    """???"""

    theta_outer_target: float = 5.0
    """???"""

    theta_inner_target: float = 5.0
    """???"""

    f_p_sol_near: float = 0.65
    """???"""

    fw_lambda_q_near_omp: float = 0.003
    """???"""

    fw_lambda_q_far_omp: float = 0.1
    """???"""

    fw_lambda_q_near_imp: float = 0.003
    """???"""

    fw_lambda_q_far_imp: float = 0.1
    """???"""

