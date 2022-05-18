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
1-D radiation model inspired by the PROCESS function "plot_radprofile" in plot_proc.py.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

from bluemira.base import constants
from bluemira.base.error import BuilderError
from bluemira.base.parameter import ParameterFrame
from bluemira.equilibria.flux_surfaces import calculate_connection_length_flt
from bluemira.equilibria.grid import Grid
from bluemira.radiation_transport.constants import SEP_CORRECTOR

if TYPE_CHECKING:
    from bluemira.radiation_transport.advective_transport import ChargedParticleSolver


class Radiation:
    """
    A simplified radiation model based on the line emission.
    """

    # fmt: off
    plasma_params = [
        ["n_el_0", "Electron density on axis", 1.81e+20, "1/m^3", None, "Input"],
        ["T_el_0", "Electron temperature on axis", 2.196e+01, "keV", None, "Input"],
        ["rho_ped_n", "Density pedestal r/a location", 9.4e-01, "dimensionless", None, "Input"],
        ["rho_ped_t", "Temperature pedestal r/a location", 9.76e-01 , "dimensionless", None, "Input"],
        ["n_el_ped", "Electron density pedestal height", 1.086e+20, "1/m^3", None, "Input"],
        ["T_el_ped", "Electron temperature pedestal height", 3.74, "keV", None, "Input"],
        ["alpha_n", "Density profile factor", 1.15, "dimensionless", None, "Input"],
        ["alpha_t", "Temperature profile index", 1.905, "dimensionless", None, "Input"],
        ["t_beta", "Temperature profile index beta", 2, "dimensionless", None, "Input"],
        ["n_el_sep", "Electron density at separatrix", 1.5515e+19, "1/m^3", None, "Input"],
        ["T_el_sep", "Electron temperature at separatrix", 4.8e-01, "keV", None, "Input"],
        ["q_95", "Safety factor at 0.95 flux_surface", 4.9517, "dimensionless", None, "Input"],
        ["rminor", "Minor_radius", 2.183, "m", None, "Input"],
        ["kappa", "Elongation", 2.8, "dimensionless", None, "Input"],
    ]
    # fmt: on

    def __init__(
        self,
        transport_solver: ChargedParticleSolver,
        plasma_params: ParameterFrame,
    ):
        self.plasma_params = ParameterFrame(self.plasma_params)
        self.plasma_params.update_kw_parameters(
            plasma_params, f"{self.__class__.__name__} input"
        )

        self.transport_solver = transport_solver

        self.collect_x_and_o_point_coordinates()

        # Separatrix parameters
        self.collect_separatrix_parameters()

    def collect_x_and_o_point_coordinates(self):
        """
        Magnetic axis coordinates and x-point(s) coordinates.
        """
        o_point, x_point = self.transport_solver.eq.get_OX_points()
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
        Some parameters related to the separatrix
        """
        self.separatrix = self.transport_solver.eq.get_separatrix()
        # The two halves
        self.sep_lfs = self.separatrix[0]
        self.sep_hfs = self.separatrix[1]
        # The mid-plane radii
        self.x_sep_omp = self.transport_solver.x_sep_omp
        self.x_sep_imp = self.transport_solver.x_sep_imp
        # To move away from the mathematical separatrix which would
        # give infinite connection length
        self.r_sep_omp = self.x_sep_omp + SEP_CORRECTOR
        self.r_sep_imp = self.x_sep_imp - SEP_CORRECTOR
        # Mid-plane z coordinate
        self.z_mp = self.points["o_point"]["z"]
        # magnetic field components at the midplane
        self.Bp_sep_omp = self.transport_solver.eq.Bp(self.x_sep_omp, self.z_mp)
        Bt_sep_omp = self.transport_solver.eq.Bt(self.x_sep_omp)
        self.Btot_sep_omp = np.hypot(self.Bp_sep_omp, Bt_sep_omp)
        self.pitch_angle_omp = self.Btot_sep_omp / self.Bp_sep_omp
        self.Bp_sep_imp = self.transport_solver.eq.Bp(self.x_sep_imp, self.z_mp)
        Bt_sep_imp = self.transport_solver.eq.Bt(self.x_sep_imp)
        self.Btot_sep_imp = np.hypot(self.Bp_sep_imp, Bt_sep_imp)
        self.pitch_angle_imp = self.Btot_sep_imp / self.Bp_sep_imp

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
        return [self.transport_solver.eq.get_flux_surface(psi) for psi in psi_n]

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
    ):
        """
        Along a single flux tube, it assignes different temperature values.

        Parameters
        ----------
        flux_tube: loop
            flux tube geometry
        te_mp: float
            electron temperature at the midplane
        core: boolean
            if True, t is constant along the flux tube. If false,it varies
        t_rad_in: float
            temperature value at the entrance of the radiation region
        t_rad_out: float
            temperature value at the exit of the radiation region
        rad_i: np.array
            indeces of points, belonging to the flux tube, which fall
            into the radiation region
        rec_i: np.array
            indeces of points, belonging to the flux tube, which fall
            into the recycling region
        t_tar: float
            electron temperature at the target

        Returns
        -------
        te: np.array
            poloidal distribution of electron temperature
        """
        te = np.array([te_mp] * len(flux_tube))
        if core is True:
            return te

        if rad_i is not None:
            te[rad_i] = self.exponential_decay(
                t_rad_in, t_rad_out, len(rad_i), decay=True
            )
            te[rec_i] = t_tar

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
        x_point_rad=False,
    ):
        """
        Along a single flux tube, it assignes different density values.

        Parameters
        ----------
        flux_tube: loop
            flux tube geometry
        ne_mp: float
            electron density at the midplane
        core: boolean
            if True, n is constant along the flux tube. If false,it varies
        n_rad_in: float
            density value at the entrance of the radiation region
        n_rad_out: float
            density value at the exit of the radiation region
        rad_i: np.array
            indeces of points, belonging to the flux tube, which fall
            into the radiation region
        rec_i: np.array
            indeces of points, belonging to the flux tube, which fall
            into the recycling region
        n_tar: float
            electron density at the target
        x_point_rad: boolean
            if True, it assumes there is no radiation at all in the recycling region

        Returns
        -------
        ne: np.array
            poloidal distribution of electron desnity
        """
        # initialising ne with same values all along the flux tube
        ne = np.array([ne_mp] * len(flux_tube))
        if core is True:
            return ne

        # choosing between lower and upper divertor
        if rad_i is not None:
            ne[rad_i] = self.exponential_decay(n_rad_out, n_rad_in, len(rad_i))

        # changing ne values according to the region
        if rec_i is not None and x_point_rad:
            ne[rec_i] = self.gaussian_decay(n_rad_out, 0.1, len(rec_i))
        elif rec_i is not None and x_point_rad is False:
            ne[rec_i] = self.gaussian_decay(n_rad_out, n_tar, len(rec_i))

        return ne

    def plot_1d_profile(self, rho, rad_power, ax=None):
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
            radiated power at each mid-plane location corresponding to rho.
        """
        if ax is None:
            fig, ax = plt.subplots()
            plt.xlabel("rho")
            plt.ylabel("MW/m^3")

        if rad_power.ndim == 1:
            ax.plot(rho, rad_power)
        else:
            [ax.plot(rho, rad_part) for rad_part in rad_power]
            rad_tot = np.sum(rad_power, axis=0)
            ax.plot(rho, rad_tot)


class TwoPointModelTools(Radiation):
    """
    Equations from the Two Point Model and other used formulas
    """

    # fmt: off
    rad_params = [
        ["p_sol", "power entering the SoL", 300e6, "W", None, "Input"],
        ["lambda_q_n", "near SoL decay length", 0.01, "m", None, "Input"],
        ["lambda_q_f", "far SoL decay length", 0.1, "m", None, "Input"],
        ["k_0", "material's conductivity", 2000, "dimensionless", None, "Input"],
        ["gamma", "sheat heat transmission coefficient", 7, "dimensionless", None, "Input"],
        ["eps_cool", "electron energy loss", 25, "eV", None, "Input"],
        ["f_ion_t", "Hydrogen first ionization", 0.01, "keV", None, "Input"],
        ["lfs_p_fraction", "lfs fraction of SoL power", 0.8, "dimensionless", None, "Input"],
        ["div_p_sharing", "Power fraction towards each divertor", 0.5, "dimensionless", None, "Input"],
    ]
    # fmt: on

    def __init__(
        self,
        transport_solver: ChargedParticleSolver,
        plasma_params,
        rad_params,
    ):
        super().__init__(transport_solver, plasma_params)

        self.rad_params = ParameterFrame(self.rad_params)
        self.rad_params.update_kw_parameters(
            rad_params, f"{self.__class__.__name__} input"
        )

    def upstream_temperature(self, firstwall_geom: Grid, n=2):
        """
        Calculation of the temperature at the upstream location according
        to PROCESS parameters and the total power crossing the separatrix.

        Parameters
        ----------
        firstwall_geom: grid
            first wall geometry
        n: float
            number of nulls

        Returns
        -------
        t_upstream: float
            upstream temperature. Unit [keV]
        q_u: float
            upstream power density [W/m^2]
        """
        # minor radius
        a = self.plasma_params.rminor
        # elongation
        k = self.plasma_params.kappa
        # safety factor
        q_95 = self.plasma_params.q_95

        # SoL cross-section at the midplane (???)
        a_par = 4 * np.pi * a * (k ** (1 / 2)) * n * self.rad_params.lambda_q_f
        # power density at the upstream (???)
        q_u = (self.rad_params.p_sol * q_95) / a_par

        # connection length from the midplane to the target
        self.l_tot = calculate_connection_length_flt(
            self.transport_solver.eq,
            self.r_sep_omp,
            self.z_mp,
            first_wall=firstwall_geom,
        )

        # upstream temperature [keV]
        t_upstream_ev = (3.5 * (q_u / self.rad_params.k_0) * self.l_tot) ** (2 / 7)
        t_upstream_kev = constants.raw_uc(t_upstream_ev, "eV", "keV")

        return t_upstream_kev, q_u

    def target_temperature(self, q_u, t_u, lfs=True):
        """
        Calculation of the temperature at the target location.
        Extended 2PM. It includes hydrogen recycle loss energy.
        Ref. Stangeby, "The Plasma Boundary of Magnetic Fusion
        Devices", 2000.

        Parameters
        ----------
        q_u: float
            upstream power density [W/m^2]
        t_u: float
            upstream temperature. Unit [keV]
        lfs: boolean
            low field side. Default value True.
            If False it stands for high field side (hfs).

        Returns
        -------
        t_tar: float
            target temperature. Unit [keV]
        """
        p_fraction = (
            self.rad_params.lfs_p_fraction if lfs else 1 - self.rad_params.lfs_p_fraction
        )
        q_u = p_fraction * q_u * self.rad_params.div_p_sharing
        # Speed of light to convert kg to eV/c^2
        light_speed = constants.C_LIGHT
        # deuterium ion mass
        m_i_amu = constants.D_MOLAR_MASS
        m_i_kg = constants.raw_uc(m_i_amu, "amu", "kg")
        m_i = m_i_kg / (light_speed**2)
        # From keV to eV
        t_u = constants.raw_uc(t_u, "keV", "eV")
        n_u = self.plasma_params.n_el_0
        # Numerator and denominator of the upstream forcing function
        num_f = m_i * 4 * (q_u**2)
        den_f = (
            2
            * constants.E_CHARGE
            * (self.rad_params.gamma**2)
            * (constants.E_CHARGE**2)
            * (n_u**2)
            * (t_u**2)
        )
        # To address all the conversion from J to eV
        f_ev = constants.raw_uc(num_f / den_f, "J", "eV")
        # Critical target temperature
        t_crit = self.rad_params.eps_cool / self.rad_params.gamma
        # Finding roots of the target temperature quadratic equation
        coeff_2 = 2 * (self.rad_params.eps_cool / self.rad_params.gamma) - f_ev
        coeff_3 = (self.rad_params.eps_cool**2) / (self.rad_params.gamma**2)
        coeff = [1, coeff_2, coeff_3]
        roots = np.roots(coeff)
        if roots.dtype == complex:
            t_tar = constants.raw_uc(self.rad_params.f_ion_t, "keV", "eV")
        else:
            # Excluding unstable solution
            sol_i = np.where(roots > t_crit)[0][0]
            # Target temperature
            t_tar = roots[sol_i]
        t_tar = constants.raw_uc(t_tar, "eV", "keV")

        return t_tar

    def x_point_temperature(self, q_u, t_u, firstwall_geom: Grid):
        """
        Calculation of the temperature at the x-point

        Parameters
        ----------
        q_u: float
            upstream power density [W/m^2]
        t_upstream: float
            upstream temperature. Unit [keV]
        firstwall_geom: grid
            first wall geometry

        Returns
        -------
        t_x: float
            x-point temperature. Unit [keV]
        """
        # From keV to eV
        t_u = constants.raw_uc(t_u, "keV", "eV")

        # Distance between x-point and target
        s_x = calculate_connection_length_flt(
            self.transport_solver.eq,
            self.points["x_point"]["x"] + SEP_CORRECTOR,
            self.points["x_point"]["z_low"],
            first_wall=firstwall_geom,
        )

        # connection length from mp to x-point
        l_x = self.l_tot - s_x
        # poca differe
        t_x = ((t_u**3.5) - 3.5 * (q_u / self.rad_params.k_0) * l_x) ** (2 / 7)

        # From eV to keV
        t_x = constants.raw_uc(t_x, "eV", "keV")

        return t_x

    def random_point_temperature(
        self, x_p, z_p, t_u, q_u, firstwall_geom: Grid, lfs=True
    ):
        """
        Calculation of the temperature at a random point above the x-point

        Parameters
        ----------
        x_p: float
            x coordinate of the point
        z_p: float
            z coordinate of the point
        t_u: float
            upstream temperature [keV]
        q_u: float
            upstream power density flux [W/m^2]
        firstwall_geom: grid
            first wall geometry
        lfs: boolean
            low (toroidal) field side (outer wall side). Default value True.
            If False it stands for high field side (hfs).

        Returns
        -------
        t_p: float
            point temperature. Unit [keV]
        """
        # From keV to eV
        t_u = constants.raw_uc(t_u, "keV", "eV")

        # Distinction between lfs and hfs
        if lfs:
            p_fraction = self.rad_params.lfs_p_fraction
            d = SEP_CORRECTOR
        else:
            p_fraction = 1 - self.rad_params.lfs_p_fraction
            d = -SEP_CORRECTOR

        q_u = p_fraction * q_u * self.rad_params.div_p_sharing

        # Distance between the chosen point and the the target
        if (lfs and z_p < self.points["o_point"]["z"]) or (
            not lfs and z_p > self.points["o_point"]["z"]
        ):
            forward = True
        else:
            forward = False
        l_p = calculate_connection_length_flt(
            self.transport_solver.eq,
            x_p + d,
            z_p,
            forward=forward,
            first_wall=firstwall_geom,
        )

        # connection length from mp to p point
        s_p = self.l_tot - l_p

        # Local temperature
        t_p = ((t_u**3.5) - 3.5 * (q_u / self.rad_params.k_0) * s_p) ** (2 / 7)

        # From eV to keV
        t_p = constants.raw_uc(t_p, "eV", "keV")

        return t_p

    def electron_density_and_temperature_sol_decay(
        self,
        t_sep: float,
        n_sep: float,
        f_exp=1,
        lfs=True,
    ):
        """
        Generic radial esponential decay to be applied from a generic starting point
        at the separatrix (not only at the mid-plane).
        The vertical location is dictated by the choice of the flux expansion f_exp.
        By default f_exp = 1, meaning mid-plane.
        The boolean "omp" set by default as "True", gives the option of choosing
        either outer mid-plane (True) or inner mid-plane (False)
        From the power decay length it calculates the temperature decay length and the
        density decay length.

        Parameters
        ----------
        t_sep: float
            initial temperature value at the separatrix [keV]
        n_sep: float
            initial density value at the separatrix [1/m^3]
        f_exp: float
            flux expansion. Default value=1 referred to the mid-plane
        lfs: boolean
            low (toroidal) field side (outer wall side). Default value True.
            If False it stands for high field side (hfs).

        Returns
        -------
        te_sol: np.array
            radial decayed temperatures through the SoL. Unit [keV]
        ne_sol: np.array
            radial decayed densities through the SoL. unit [1/m^3]
        """
        # temperature and density decay factors
        t_factor = 7 / 2
        n_factor = 1

        # power decay length modified according to the flux expansion
        lambda_q_n = self.rad_params.lambda_q_n * f_exp
        lambda_q_f = self.rad_params.lambda_q_f * f_exp

        # radial distance of flux tubes from the separatrix
        dr = (
            self.transport_solver.dx_omp * f_exp
            if lfs
            else self.transport_solver.dx_imp * f_exp
        )

        # Assuming conduction-limited regime.
        lambda_t_n = t_factor * lambda_q_n
        lambda_t_f = t_factor * lambda_q_f
        lambda_n_n = n_factor * lambda_t_n
        lambda_n_f = n_factor * lambda_t_f

        # dividing between near and far SoL
        i_sol_near = np.where(dr < lambda_q_n)
        i_sol_far = np.where(dr > lambda_q_n)

        te_sol_near = t_sep * np.exp(-dr / lambda_t_n)
        ne_sol_near = n_sep * np.exp(-dr / lambda_n_n)
        te_sol_near = t_sep * np.exp(-dr[i_sol_near] / lambda_t_n)
        ne_sol_near = n_sep * np.exp(-dr[i_sol_near] / lambda_n_n)

        te_sol_far = te_sol_near[-1] * np.exp(-dr[i_sol_far] / lambda_t_f)
        ne_sol_far = ne_sol_near[-1] * np.exp(-dr[i_sol_far] / lambda_n_f)

        te_sol = np.append(te_sol_near, te_sol_far)
        ne_sol = np.append(ne_sol_near, ne_sol_far)

        return te_sol, ne_sol

    def gaussian_decay(self, max_value, min_value, no_points):
        """
        Generic gaussian decay to be applied between two extreme values and for a
        given number of points.

        Parameters
        ----------
        max_value: float
            maximum value of the parameters
        min_value: float
            minimum value of the parameters
        no_points: float
            number of points through which make the parameter decay

        Returns
        -------
        dec_param: np.array
            decayed parameter
        """
        no_points = max(no_points, 1)

        # setting values on the horizontal axis
        x = np.linspace(no_points, 0, no_points)

        # centering the gaussian on its highest value
        mu = max(x)

        # setting sigma to be consistent with min_value
        frac = max_value / min_value
        lg = np.log(frac)
        denominator = 2 * lg * (1 / (mu**2))
        sigma = np.sqrt(1 / denominator)
        h = 1 / np.sqrt(2 * sigma**2)

        # decaying param
        dec_param = max_value * (np.exp((-(h**2)) * ((x - mu) ** 2)))
        i_near_minimum = np.where(dec_param < min_value)
        dec_param[i_near_minimum[0]] = min_value

        return dec_param

    def exponential_decay(self, max_value, min_value, no_points, decay=False):
        """
        Generic exponential decay to be applied between two extreme values and for a
        given number of points.

        Parameters
        ----------
        max_value: float
            maximum value of the parameters
        min_value: float
            minimum value of the parameters
        no_points: float
            number of points through which make the parameter decay
        decay: boolean
            to define either a decay or increment

        Returns
        -------
        dec_param: np.array
            decayed parameter
        """
        no_points = max(no_points, 1)

        x = np.linspace(1, no_points, no_points)
        a = np.array([x[0], min_value])
        b = np.array([x[-1], max_value])
        if decay:
            arg = x / b[0]
            base = a[0] / b[0]
        else:
            arg = x / a[0]
            base = b[0] / a[0]
        my_log = np.log(arg) / np.log(base)
        f = a[1] + (b[1] - a[1]) * my_log

        return f

    def ion_front_distance(
        self,
        x_strike,
        z_strike,
        t_tar=None,
        sv_i=None,
        sv_m=None,
        n_r=None,
        rec_ext=None,
    ):
        """
        Manual definition of penetration depth.
        TODO: Find sv_i and sv_m

        Parameters
        ----------
        x_strike: float [m]
            x coordinate of the strike point
        z_strike: float [m]
            z coordinate of the strike point
        t_tar: float [keV]
            target temperature
        sv_i: float
            average ion loss coefficient
        sv_m: float
            average momentum loss coefficient
        n_r: float
            density at the recycling region entrance
        rec_ext: float [m]
            recycling region extention (along the field line)
            from the target

        Returns
        -------
        z_front: float [m]
            z coordinate of the ionization front
        """
        # Speed of light to convert kg to eV/c^2
        light_speed = constants.C_LIGHT
        # deuterium ion mass
        m_i_amu = constants.D_MOLAR_MASS
        m_i_kg = constants.raw_uc(m_i_amu, "amu", "kg")
        m_i = m_i_kg / (light_speed**2)

        # Magnetic field at the strike point
        Bp = self.transport_solver.eq.Bp(x_strike, z_strike)
        Bt = self.transport_solver.eq.Bt(x_strike)
        Btot = np.hypot(Bp, Bt)

        # From total length to poloidal length
        pitch_angle = Btot / Bp
        if rec_ext is None:
            den_lambda = 3 * np.pi * m_i * sv_i * sv_m
            z_ext = np.sqrt((8 * t_tar) / den_lambda) ** (1 / n_r)
        else:
            z_ext = rec_ext * np.sin(pitch_angle)

        # z coordinate (from the midplane)
        z_front = abs(z_strike - self.points["x_point"]["z_low"]) - z_ext

        return z_front

    def calculate_z_species(self, t_ref, z_ref, species_frac, te):
        """
        Calculation of species ion charge, in condition of quasi-neutrality.

        Parameters
        ----------
        t_ref: np.array
            temperature reference
        z_ref: np.array
            effective charge reference
        species_frac: float
            fraction of relevant impurity
        te: array
            electron temperature

        Returns
        -------
        species_frac*z_val**2: np.array
            species ion charge
        """
        z_interp = interp1d(t_ref, z_ref)
        z_val = z_interp(te)

        return species_frac * z_val**2

    def radiative_loss_function_values(self, te, t_ref, l_ref):
        """
        By interpolation, from reference values, it returns the
        radiative power loss values for a given set of electron temperature.

        Parameters
        ----------
        te: np.array
            electron temperature
        t_ref: np.array
            temperature reference
        l_ref: np.array
            radiative power loss reference

        Returns
        -------
        l_val: np.array [W m^3]
            local values of the radiative power loss function
        """
        interp_func = interp1d(t_ref, l_ref)

        return interp_func(te)

    def calculate_line_radiation_loss(self, ne, p_loss_f, species_frac):
        """
        Calculation of Line radiation losses.
        For a given impurity this is the total power lost, per unit volume,
        by all line-radiation processes INCLUDING Bremsstrahlung.

        Parameters
        ----------
        ne: np.array
            electron density
        p_loss_f: np.array
            local values of the radiative power loss function
        species_frac: float
            fraction of relevant impurity

        Returns
        -------
        rad_loss: np.array
            Line radiation losses [MW m^-3]
        """
        return (species_frac * (ne**2) * p_loss_f) / 1e6


class Core(Radiation):
    """
    Specific for the core emission.
    """

    def collect_rho_core_values(self):
        """
        Calculation of core dimensionless radial coordinate rho.

        Returns
        -------
        rho_core: np.array
            dimensionless core radius. Values between 0 and 1
        """
        # The plasma bulk is divided into plasma core and plasma mantle according to rho
        # rho is a nondimensional radial coordinate: rho = r/a (r varies from 0 to a)
        self.rho_ped = (
            self.plasma_params["rho_ped_n"] + self.plasma_params.rho_ped_t
        ) / 2.0

        # Plasma core for rho < rho_core
        rho_core1 = np.linspace(0, 0.95 * self.rho_ped)
        rho_core2 = np.linspace(0.95 * self.rho_ped, self.rho_ped)
        rho_core = np.append(rho_core1, rho_core2)

        # Plasma mantle for rho_core < rho < 1
        rho_sep = np.linspace(self.rho_ped, 0.99)

        rho_core = np.append(rho_core, rho_sep)

        return rho_core

    def core_electron_density_temperature_profile(self, rho_core):
        """
        Calculation of electron density and electron temperature,
        as function of rho, from the magnetic axis to the separatrix,
        along the midplane.
        The region that extends through the plasma core until its
        outer layer, named pedestal, is referred as "interior".
        The region that extends from the pedestal to the separatrix
        is referred as "exterior".

        Returns
        -------
        ne: np.array
            electron densities at the mid-plane. Unit [1/m^3]
        te: np.array
            electron temperature at the mid-plane. Unit [keV]
        """
        i_interior = np.where((rho_core >= 0) & (rho_core <= self.rho_ped))[0]

        n_grad_ped0 = self.plasma_params.n_el_0 - self.plasma_params.n_el_ped
        t_grad_ped0 = self.plasma_params.T_el_0 - self.plasma_params.T_el_ped

        rho_ratio_n = (
            1 - ((rho_core[i_interior] ** 2) / (self.rho_ped**2))
        ) ** self.plasma_params.alpha_n

        rho_ratio_t = (
            1
            - (
                (rho_core[i_interior] ** self.plasma_params.t_beta)
                / (self.rho_ped**self.plasma_params.t_beta)
            )
        ) ** self.plasma_params.alpha_t

        ne_i = self.plasma_params.n_el_ped + (n_grad_ped0 * rho_ratio_n)
        te_i = self.plasma_params.T_el_ped + (t_grad_ped0 * rho_ratio_t)

        i_exterior = np.where((rho_core > self.rho_ped) & (rho_core <= 1))[0]

        n_grad_sepped = self.plasma_params.n_el_ped - self.plasma_params.n_el_sep
        t_grad_sepped = self.plasma_params.T_el_ped - self.plasma_params.T_el_sep

        rho_ratio = (1 - rho_core[i_exterior]) / (1 - self.rho_ped)

        ne_e = self.plasma_params.n_el_sep + (n_grad_sepped * rho_ratio)
        te_e = self.plasma_params.T_el_sep + (t_grad_sepped * rho_ratio)

        ne_core = np.append(ne_i, ne_e)
        te_core = np.append(te_i, te_e)

        return ne_core, te_core

    def plot_2d_map(self, flux_tubes, power_density, ax=None):
        """
        2D plot of the core radation power distribution.

        Parameters
        ----------
        flux_tubes: np.array
            array of the closed flux tubes within the separatrix.
        power_density: np.array([np.array])
            arrays containing the power radiation density of the
            points lying on each flux tube.
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        p_min = min([np.amin(p) for p in power_density])
        p_max = max([np.amax(p) for p in power_density])

        separatrix = self.transport_solver.eq.get_separatrix()
        for sep in separatrix:
            sep.plot(ax, linewidth=2)
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

        fig.colorbar(cm, label="MW/m^3")


class ScrapeOffLayer(Radiation):
    """
    Specific for the SoL emission.
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
        a = self.transport_solver.x_sep_omp - r_o_point
        rho_sol = (a + self.transport_solver.dx_omp) / a

        return rho_sol

    def x_point_radiation_z_ext(self, main_ext=0.2, pfr_ext=0.3, low_div=True):
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
        sep_loop = self.sep_lfs if lfs else self.sep_hfs
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
        For a given flux tube, indeces of points which fall respectively
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
            indeces of the points within the radiation region
        rec_i: np.array
            indeces pf the points within the recycling region
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
        omp: boolean
            outer mid-plane. Default value True. If False it stands for inner mid-plane

        Returns
        -------
        te_sol: np.array
            radial decayed temperatures through the SoL at the mid-plane. Unit [keV]
        ne_sol: np.array
            radial decayed densities through the SoL at the mid-plane. Unit [1/m^3]
        """
        if te_sep is None:
            te_sep = self.plasma_params.T_el_sep
        ne_sep = self.plasma_params.n_el_sep
        te_sol, ne_sol = self.electron_density_and_temperature_sol_decay(
            te_sep, ne_sep, lfs=omp
        )

        return te_sol, ne_sol

    def any_point_n_t_profiles(
        self,
        x_p,
        z_p,
        t_p,
        t_u,
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
            x coordinate of the point at the separatrix
        z_p: float
            z coordinate of the point at the separatrix
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
        if lfs is True:
            r_sep_mp = self.r_sep_omp
            Bp_sep_mp = self.Bp_sep_omp
        else:
            r_sep_mp = self.r_sep_imp
            Bp_sep_mp = self.Bp_sep_imp

        # magnetic field components at the local point
        Bp_p = self.transport_solver.eq.Bp(x_p, z_p)

        # flux expansion
        f_p = (r_sep_mp * Bp_sep_mp) / (x_p * Bp_p)

        # Ratio between upstream and local temperature
        f_t = t_u / t_p

        # Local electron temperature
        n_p = self.plasma_params.n_el_sep * f_t

        # Temperature and density profiles across the SoL
        te_prof, ne_prof = self.electron_density_and_temperature_sol_decay(
            t_p, n_p, f_exp=f_p
        )

        return te_prof, ne_prof

    def tar_electron_densitiy_temperature_profiles(self, ne_div, te_div, f_m=1):
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
            assumed to be corresponding to the divertor plane
        te_div: np.array
            temperature of the flux tubes at the entrance of the recycling region,
            assumed to be corresponding to the divertor plane
        f_m: float
            fractional loss factor

        Returns
        -------
        te_t: np.array
            target temperature
        ne_t: np.array
            target density
        """
        te_t = te_div
        ne_t = (f_m * ne_div) / 2

        return te_t, ne_t

    def plot_2d_map(self, flux_tubes, power_density, firstwall, ax=None):
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
            expected len(flux_tubes) = len(power_desnity)
        firstwall: Grid
            first wall geometry
        """
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        tubes = np.concatenate(flux_tubes)
        power = np.concatenate(power_density)

        p_min = min([np.amin(p) for p in power])
        p_max = max([np.amax(p) for p in power])

        firstwall.plot(ax, linewidth=0.5, fill=False)
        separatrix = self.transport_solver.eq.get_separatrix()
        for sep in separatrix:
            sep.plot(ax, linewidth=2)
        for flux_tube, p in zip(tubes, power):
            cm = ax.scatter(
                flux_tube.loop.x,
                flux_tube.loop.z,
                c=p,
                s=10,
                cmap="plasma",
                vmin=p_min,
                vmax=p_max,
                zorder=40,
            )

        fig.colorbar(cm, label="MW/m^3")


class ScrapeOffLayerSector(ScrapeOffLayer, TwoPointModelTools):
    """
    To build a single sector, from the upstream location
    to one of the targets.
    """

    def build_sector_profiles(
        self,
        flux_tubes,
        x_strike,
        z_strike,
        main_ext,
        firstwall_geom,
        pfr_ext=None,
        rec_ext=None,
        x_point_rad=False,
        lfs=True,
        low_div=True,
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
            x coordinate of the first open flux surface strike point
        z_strike: float
            z coordinate of the first open flux surface strike point
        main_ext: float [m]
            extention of the radiation region from the x-point
            towards the main plasma
        firstwall_geom: grid
            first wall geometry
        pfr_ext: float [m]
            extention of the radiation region from the x-point
            towards private flux region
        rec_ext: float [m]
            extention of the recycling region,
            along the separatrix, from the target
        x_point_rad: boolean
            if True, it assumes there is no radiation at all
            in the recycling region, and pfr_ext MUST be provided.
        lfs: boolean
            low field side. Default value True.
            If False it stands for high field side (hfs)
        low_div: boolean
            default=True for the lower divertor.
            If False, upper divertor

        Returns
        -------
        t_pol: array
            temperature poloidal profile along each
            flux tube within the specified set
        n_pol: array
            density poloidal profile along each
            flux tube within the specified set
        """
        # Validity condition for not x-point radiative
        if not x_point_rad and rec_ext is None:
            raise BuilderError("Required recycling region extention: rec_ext")
        if not x_point_rad and rec_ext is not None:
            ion_front_z = self.ion_front_distance(x_strike, z_strike, rec_ext=rec_ext)
            pfr_ext = ion_front_z

        # Validity condition for x-point radiative
        if x_point_rad and pfr_ext is None:
            raise BuilderError("Required extention towards pfr: pfr_ext")

        # setting radiation and recycling regions
        z_main, z_pfr = self.x_point_radiation_z_ext(main_ext, pfr_ext, low_div)

        in_x, in_z, out_x, out_z = self.radiation_region_ends(z_main, z_pfr, lfs)

        reg_i = [
            self.radiation_region_points(f.loop, z_main, z_pfr, low_div)
            for f in flux_tubes
        ]

        # mid-plane parameters
        t_mp_prof, n_mp_prof = self.mp_electron_density_temperature_profiles(
            self.t_u, lfs
        )

        # entrance of radiation region
        t_rad_in = self.random_point_temperature(
            in_x,
            in_z,
            self.t_u,
            self.q_u,
            firstwall_geom,
            lfs,
        )

        # exit of radiation region
        if x_point_rad and pfr_ext is not None:
            t_rad_out = self.rad_params.f_ion_t
        else:
            t_rad_out = self.target_temperature(
                self.q_u,
                self.t_u,
                lfs,
            )

        # profiles through the SoL
        t_in_prof, n_in_prof = self.any_point_n_t_profiles(
            in_x,
            in_z,
            t_rad_in,
            self.t_u,
            lfs,
        )

        t_out_prof, n_out_prof = self.any_point_n_t_profiles(
            out_x,
            out_z,
            t_rad_out,
            self.t_u,
            lfs,
        )

        t_tar_prof, n_tar_prof = self.tar_electron_densitiy_temperature_profiles(
            n_out_prof,
            t_out_prof,
        )

        # temperature poloidal distribution
        t_pol = [
            self.flux_tube_pol_t(
                f.loop,
                t,
                t_rad_in=t_in,
                t_rad_out=t_out,
                rad_i=reg[0],
                rec_i=reg[1],
                t_tar=t_t,
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

        # condition for occurred detachment
        if t_rad_out <= self.rad_params.f_ion_t:
            x_point_rad = True

        # density poloidal distribution
        n_pol = [
            self.flux_tube_pol_n(
                f.loop,
                n,
                n_rad_in=n_in,
                n_rad_out=n_out,
                rad_i=reg[0],
                rec_i=reg[1],
                n_tar=n_t,
                x_point_rad=x_point_rad,
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


class STCore(Core, TwoPointModelTools):
    """
    Specific class for the core emission of STEP
    """

    def __init__(
        self,
        transport_solver: ChargedParticleSolver,
        impurity_content,
        impurity_data,
        plasma_params,
        rad_params,
    ):
        super().__init__(transport_solver, plasma_params, rad_params)

        self.H_content = impurity_content["H"]
        self.impurities_content = impurity_content.values()

        self.imp_data_t_ref = [data["T_ref"] for key, data in impurity_data.items()]
        self.imp_data_l_ref = [data["L_ref"] for key, data in impurity_data.items()]

        # Adimensional radius at the mid-plane.
        # From the core to the last closed flux surface
        self.rho_core = self.collect_rho_core_values()

        # For each flux tube, density and temperature at the mid-plane
        self.ne_mp, self.te_mp = self.core_electron_density_temperature_profile(
            self.rho_core
        )

    def build_mp_rad_profile(self):
        """
        1D profile of the line radiation loss at the mid-plane
        through the scrape-off layer
        """
        # Radiative loss function values for each impurity species
        loss_f = np.array(
            [
                self.radiative_loss_function_values(self.te_mp, t_ref, l_ref)
                for t_ref, l_ref in zip(self.imp_data_t_ref, self.imp_data_l_ref)
            ]
        )

        # Line radiation loss. Mid-plane distribution through the SoL
        rad = np.array(
            [
                self.calculate_line_radiation_loss(self.ne_mp, loss, fi)
                for loss, fi in zip(loss_f, self.impurities_content)
            ]
        )

        self.plot_1d_profile(self.rho_core, rad)

    def build_core_radiation_map(self):
        """
        2D map of the line radiation loss in the plasma core.
        """
        # Closed flux tubes within the separatrix
        flux_tubes = self.collect_flux_tubes(self.rho_core)

        # For each flux tube, poloidal density profile.
        ne_pol = [
            self.flux_tube_pol_n(ft, n, core=True)
            for ft, n in zip(flux_tubes, self.ne_mp)
        ]

        # For each flux tube, poloidal temperature profile.
        te_pol = [
            self.flux_tube_pol_t(ft, t, core=True)
            for ft, t in zip(flux_tubes, self.te_mp)
        ]

        # For each impurity species and for each flux tube,
        # poloidal distribution of the radiative power loss function.
        loss_f = [
            [self.radiative_loss_function_values(t, t_ref, l_ref) for t in te_pol]
            for t_ref, l_ref in zip(self.imp_data_t_ref, self.imp_data_l_ref)
        ]

        # For each impurity species and for each flux tube,
        # poloidal distribution of the line radiation loss.
        rad = [
            [
                self.calculate_line_radiation_loss(n, l_f, fi)
                for n, l_f in zip(ne_pol, ft)
            ]
            for ft, fi in zip(loss_f, self.impurities_content)
        ]

        # Total line radiation loss along each flux tube
        total_rad = np.sum(rad, axis=0)

        self.plot_2d_map(flux_tubes, total_rad)


class STScrapeOffLayer(ScrapeOffLayerSector, ScrapeOffLayer, TwoPointModelTools):
    """
    Specific class for the scrape-off layer emission of ST
    """

    def __init__(
        self,
        transport_solver: ChargedParticleSolver,
        impurity_content,
        impurity_data,
        plasma_params,
        rad_params,
        firstwall_geom,
    ):
        super().__init__(transport_solver, plasma_params, rad_params)

        self.H_content = impurity_content["H"]
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
        self.flux_tubes_lfs_low = self.transport_solver.flux_surfaces_ob_lfs
        self.flux_tubes_hfs_low = self.transport_solver.flux_surfaces_ib_lfs
        # partial flux tube from the mp to the target at the
        # outboard and inboard - upper divertor
        self.flux_tubes_lfs_up = self.transport_solver.flux_surfaces_ob_hfs
        self.flux_tubes_hfs_up = self.transport_solver.flux_surfaces_ib_hfs

        # strike points from the first open flux tube
        self.x_strike_lfs = self.flux_tubes_lfs_low[0].loop.x[-1]
        self.z_strike_lfs = self.flux_tubes_lfs_low[0].loop.z[-1]
        self.x_strike_hfs = self.flux_tubes_hfs_low[0].loop.x[-1]
        self.z_strike_hfs = self.flux_tubes_hfs_low[0].loop.z[-1]

        # upstream temperature and power density
        self.t_u, self.q_u = self.upstream_temperature(firstwall_geom)

        # temperature and density profiles along the flux tubes
        (
            t_and_n_pol_lfs_low,
            t_and_n_pol_lfs_up,
            t_and_n_pol_hfs_low,
            t_and_n_pol_hfs_up,
        ) = self.build_sol_profiles(firstwall_geom)

        # radiation distribution along the flux tubes
        (
            rad_lfs_low,
            rad_lfs_up,
            rad_hfs_low,
            rad_hfs_up,
        ) = self.build_sol_rad_distribution(
            t_and_n_pol_lfs_low,
            t_and_n_pol_lfs_up,
            t_and_n_pol_hfs_low,
            t_and_n_pol_hfs_up,
        )

        # scrape off layer radiation map
        self.build_sol_radiation_map(
            rad_lfs_low,
            rad_lfs_up,
            rad_hfs_low,
            rad_hfs_up,
            firstwall_geom,
        )

    def build_sol_profiles(self, firstwall_geom):
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
        t_and_n_pol = {
            f"{side}_{low_up}": {
                "flux_tube": getattr(self, f"flux_tubes_{side}_{low_up}"),
                "x_strike": getattr(self, f"x_strike_{side}"),
                "z_strike": getattr(self, f"z_strike_{side}"),
                "main_ext": 0.2,
                "wall_profile": firstwall_geom,
                "pfr_ext": None,
                "rec_ext": 0.5,
                "x_point_rad": False,
                "lfs": side == "lfs",
                "low_div": low_up == "low",
            }
            for side in ["lfs", "hfs"]
            for low_up in ["low", "up"]
        }

        for side, var in t_and_n_pol.items():
            t_and_n_pol[side] = self.build_sector_profiles(
                var["flux_tube"],
                var["x_strike"],
                var["z_strike"],
                var["main_ext"],
                var["wall_profile"],
                var["pfr_ext"],
                var["rec_ext"],
                var["x_point_rad"],
                var["lfs"],
                var["low_div"],
            )

        return (
            t_and_n_pol["lfs_low"],
            t_and_n_pol["lfs_up"],
            t_and_n_pol["hfs_low"],
            t_and_n_pol["hfs_up"],
        )

    def build_sol_rad_distribution(
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
        # radiative loss function along the open flux tubes
        loss = {
            "lfs_low": t_and_n_pol_lfs_low[0],
            "lfs_up": t_and_n_pol_lfs_up[0],
            "hfs_low": t_and_n_pol_hfs_low[0],
            "hfs_up": t_and_n_pol_hfs_up[0],
        }
        for side, t_pol in loss.items():
            loss[side] = [
                [self.radiative_loss_function_values(t, t_ref, l_ref) for t in t_pol]
                for t_ref, l_ref in zip(self.imp_data_t_ref, self.imp_data_l_ref)
            ]

        # line radiation loss along the open flux tubes
        rad = {
            "lfs_low": {"density": t_and_n_pol_lfs_low[1], "loss": loss["lfs_low"]},
            "lfs_up": {"density": t_and_n_pol_lfs_up[1], "loss": loss["lfs_up"]},
            "hfs_low": {"density": t_and_n_pol_hfs_low[1], "loss": loss["hfs_low"]},
            "hfs_up": {"density": t_and_n_pol_hfs_up[1], "loss": loss["hfs_up"]},
        }
        for side, ft in rad.items():
            rad[side] = [
                [
                    self.calculate_line_radiation_loss(n, l_f, fi)
                    for n, l_f in zip(ft["density"], f)
                ]
                for f, fi in zip(ft["loss"], self.impurities_content)
            ]

        return rad["lfs_low"], rad["lfs_up"], rad["hfs_low"], rad["hfs_up"]

    def build_sol_radiation_map(
        self, rad_lfs_low, rad_lfs_up, rad_hfs_low, rad_hfs_up, firstwall_geom
    ):
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
        total_rad_lfs_low = np.sum(rad_lfs_low, axis=0)
        total_rad_lfs_up = np.sum(rad_lfs_up, axis=0)
        total_rad_hfs_low = np.sum(rad_hfs_low, axis=0)
        total_rad_hfs_up = np.sum(rad_hfs_up, axis=0)

        self.plot_2d_map(
            [
                self.flux_tubes_lfs_low,
                self.flux_tubes_hfs_low,
                self.flux_tubes_lfs_up,
                self.flux_tubes_hfs_up,
            ],
            [total_rad_lfs_low, total_rad_hfs_low, total_rad_lfs_up, total_rad_hfs_up],
            firstwall_geom,
        )
