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

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import process.data.impuritydata as imp_data
import scipy.constants as sc
from scipy.interpolate import interp1d

from bluemira.base.parameter import ParameterFrame
from bluemira.equilibria.flux_surfaces import calculate_connection_length_flt
from bluemira.geometry._deprecated_loop import Loop


class Radiation:
    """
    A simplified radiation model based on the line emission.
    """

    def __init__(self, process_solver, transport_solver):

        self.process_solver = process_solver
        self.transport_solver = transport_solver
        # Useful parameters from MFILE.DAT
        # Some of these parameters might be worth to re-calculate (e.g. tesep)
        parameter_names = [
            "ne0",
            "te0",
            "rhopedn",
            "rhopedt",
            "neped",
            "teped",
            "alphan",
            "alphat",
            "tbeta",
            "nesep",
            "tesep",
            "q95",
            "rminor",
            "kappa",
        ]
        parameter_values = process_solver.get_raw_variables(parameter_names)
        self.process_params = {k: v for k, v in zip(parameter_names, parameter_values)}
        # Impurities from the PROCESS database
        self.impurity_id = ["fimp(01", "fimp(02", "fimp(13", "fimp(14"]
        impurity_data = [
            "H_Lzdata.dat",
            "HeLzdata.dat",
            "XeLzdata.dat",
            "W_Lzdata.dat",
        ]

        # Radiative loss function data values for each impurity
        self.species_data = np.array(
            [
                Path(Path(imp_data.__file__).parent, impurity)
                for impurity in impurity_data
            ]
        )

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
        # Loops
        self.separatrix = self.transport_solver.eq.get_separatrix()
        # The two halves
        self.sep_lfs = self.separatrix[0]
        self.sep_hfs = self.separatrix[1]
        # The mid-plane radii
        self.x_sep_omp = self.transport_solver.x_sep_omp
        self.x_sep_imp = self.transport_solver.x_sep_imp
        # To move away from the mathematical separatrix which would
        # give infinite connection length
        self.sep_corrector = 0.004
        self.r_sep_omp = self.x_sep_omp + self.sep_corrector
        self.r_sep_imp = self.x_sep_imp - self.sep_corrector
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
        flux tubes
        """
        return np.array(
            [self.transport_solver.eq.get_flux_surface(psi) for psi in psi_n],
            dtype=object,
        )

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
        x_point_rad=True,
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
            ne[rad_i] = self.exponential_decay(n_rad_in, n_rad_out, len(rad_i))

        # changing ne values according to the region
        if rec_i is not None and x_point_rad:
            ne[rec_i] = 0
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


class Mathematics(Radiation):
    """
    Equations from the Two Point Model and other used formulas
    """

    # fmt: off
    base_default_params = [
        ["p_sol", "power entering the SoL", 300e6, "W", None, "Input"],
        ["lambda_q_n", "near SoL decay length", 0.01, "m", None, "Input"],
        ["lambda_q_f", "far SoL decay length", 0.1, "m", None, "Input"],
        ["k_0", "material's conductivity", 2000, "dimensionless", None, "Input"],
        ["gamma", "sheat heat transmission coefficient", 7, "dimensionless", None, "Input"],
        ["eps_cool", "electron energy loss", 25, "eV", None, "Input"],
        ["f_ion_t", "Hydrogen first ionization", 10, "eV", None, "Input"],
        ["lfs_p_fraction", "lfs fraction of SoL power", 0.8, "dimensionless", None, "Input"],
        ["div_p_sharing", "Power fraction towards each divertor", 0.5, "dimensionless", None, "Input"],
    ]
    # fmt: on

    def __init__(self, process_solver, transport_solver, config):
        super().__init__(process_solver, transport_solver)

        self.params = ParameterFrame(self.base_default_params)
        self.params.update_kw_parameters(config, f"{self.__class__.__name__} input")

    def upstream_temperature(self, firstwall_loop: Loop, n=2):
        """
        Calculation of the temperature at the upstream location according
        to PROCESS parameters and the total power crossing the separatrix.

        Parameters
        ----------
        firstwall_loop: loop
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
        a = self.process_params["rminor"]
        # elongation
        k = self.process_params["kappa"]
        # safety factor
        # q_95 = self.params["q95"]
        q_95 = 3.5

        # SoL cross-section at the midplane (???)
        a_par = 4 * np.pi * a * (k ** (1 / 2)) * n * self.params.lambda_q_f
        # a_par_test = 4 * np.pi * self.x_sep_omp * lambda_q
        # power density at the upstream (???)
        q_u = (self.params.p_sol * q_95) / a_par
        # q_u_test = (p_sol * self.pitch_angle_omp) / a_par_test

        # connection length from the midplane to the target
        self.l_tot = calculate_connection_length_flt(
            self.transport_solver.eq,
            self.r_sep_omp,
            self.z_mp,
            first_wall=firstwall_loop,
        )

        # upstream temperature [keV]
        t_upstream = ((3.5 * (q_u / self.params.k_0) * self.l_tot) ** (2 / 7)) * 1.0e-3

        return t_upstream, q_u

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
        if lfs:
            q_u = self.params.lfs_p_fraction * q_u * self.params.div_p_sharing
        else:
            q_u = (1 - self.params.lfs_p_fraction) * q_u * self.params.div_p_sharing

        # Conversion factor from Joule to eV
        j_to_ev = sc.physical_constants["joule-electron volt relationship"][0]
        # Speed of light to convert kg to eV/c^2
        light_speed = sc.physical_constants["speed of light in vacuum"][0]
        # deuterium ion mass
        m_i = sc.physical_constants["deuteron mass"][0]
        m_i = m_i / (light_speed**2)
        # From keV to eV
        t_u = t_u * 1.0e3
        n_u = self.process_params["ne0"]
        # Numerator and denominator of the upstream forcing function
        num_f = m_i * 4 * (q_u**2)
        den_f = (
            2 * sc.e * (self.params.gamma**2) * (sc.e**2) * (n_u**2) * (t_u**2)
        )
        # Upstream forcing function
        f = num_f / den_f
        # To address all the conversion from J to eV
        f = f * j_to_ev
        # Critical target temperature
        t_crit = self.params.eps_cool / self.params.gamma
        # Finding roots of the target temperature quadratic equation
        coeff_2 = 2 * (self.params.eps_cool / self.params.gamma) - f
        coeff_3 = (self.params.eps_cool**2) / (self.params.gamma**2)
        coeff = [1, coeff_2, coeff_3]
        roots = np.roots(coeff)
        if roots.dtype == complex:
            t_tar = self.params.f_ion_t
        else:
            # Excluding unstable solution
            sol_i = np.where(roots > t_crit)[0][0]
            # Target temperature
            t_tar = roots[sol_i]
        t_tar = t_tar * 1.0e-3

        return t_tar

    def x_point_temperature(self, q_u, t_u, firstwall_loop: Loop):
        """
        Calculation of the temperature at the x-point

        Parameters
        ----------
        q_u: float
            upstream power density [W/m^2]
        t_upstream: float
            upstream temperature. Unit [keV]
        firstwall_loop: loop
            first wall geometry

        Returns
        -------
        t_x: float
            x-point temperature. Unit [keV]
        """
        # From keV to eV
        t_u = t_u * 1.0e3

        # Distance between x-point and target
        s_x = calculate_connection_length_flt(
            self.transport_solver.eq,
            self.points["x_point"]["x"] + self.sep_corrector,
            self.points["x_point"]["z_low"],
            first_wall=firstwall_loop,
        )

        # connection length from mp to x-point
        l_x = self.l_tot - s_x
        # poca differe
        t_x = ((t_u**3.5) - 3.5 * (q_u / self.params.k_0) * l_x) ** (2 / 7)

        # From eV to keV
        t_x = t_x * 1.0e-3

        return t_x

    def random_point_temperature(
        self, x_p, z_p, t_u, q_u, firstwall_loop: Loop, lfs=True
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
        firstwall_loop: loop
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
        t_u = t_u * 1.0e3

        # Distinction between lfs and hfs
        if lfs is True:
            q_u = self.params.lfs_p_fraction * q_u * self.params.div_p_sharing
            d = self.sep_corrector
        else:
            q_u = (1 - self.params.lfs_p_fraction) * q_u * self.params.div_p_sharing
            d = -self.sep_corrector

        # Distance between the chosen point and the the target
        if lfs is True and z_p < self.points["o_point"]["z"]:
            l_p = calculate_connection_length_flt(
                self.transport_solver.eq,
                x_p + d,
                z_p,
                first_wall=firstwall_loop,
            )

        elif lfs is True and z_p > self.points["o_point"]["z"]:
            l_p = calculate_connection_length_flt(
                self.transport_solver.eq,
                x_p + d,
                z_p,
                forward=False,
                first_wall=firstwall_loop,
            )

        elif lfs is False and z_p < self.points["o_point"]["z"]:
            l_p = calculate_connection_length_flt(
                self.transport_solver.eq,
                x_p + d,
                z_p,
                forward=False,
                first_wall=firstwall_loop,
            )

        elif lfs is False and z_p > self.points["o_point"]["z"]:
            l_p = calculate_connection_length_flt(
                self.transport_solver.eq,
                x_p + d,
                z_p,
                first_wall=firstwall_loop,
            )

        # connection length from mp to p point
        s_p = self.l_tot - l_p

        # Local temperature
        t_p = ((t_u**3.5) - 3.5 * (q_u / self.params.k_0) * s_p) ** (2 / 7)

        # From eV to keV
        t_p = t_p * 1.0e-3

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
        lambda_q_n = self.params.lambda_q_n * f_exp
        lambda_q_f = self.params.lambda_q_f * f_exp

        # radial distance of flux tubes from the separatrix
        if lfs is True:
            dr = self.transport_solver.dx_omp * f_exp
        else:
            dr = self.transport_solver.dx_imp * f_exp

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

    def linear_decay(self, max_value, min_value, no_points):
        """
        Generic linear decay to be applied between two extreme values and for a
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
        decayed parameter: np.array
        """
        return np.linspace(max_value, min_value, no_points)

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
        if no_points == 0:
            no_points = 1

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
        if no_points == 0:
            no_points = 1

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
        light_speed = sc.physical_constants["speed of light in vacuum"][0]
        # deuterium ion mass
        m_i = sc.physical_constants["deuteron mass"][0]
        m_i = m_i / (light_speed**2)

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

    def calculate_z_species(self, species_file, fimp_no, te):
        """
        Calculation of species ion charge, in condition of quasi-neutrality.

        Parameters
        ----------
        species_file: string
            impurity data file (.dat)
            currently located in process/process/data/impuritydata (__Lzdata.dat)
            e.g. 1 letter acronym: Hydrogen -> H_Lzdata.dat
            e.g. 2 letters acronym: Helium -> HeLzdata.dat
        fimp_no: string
            impurity fraction to find in MFILE.dat: fimp(no -> no = [01,14]
        te: array
            electron temperature

        Returns
        -------
        species_frac*z_val**2: np.array
            species ion charge
        """
        species_data = Path(Path(imp_data.__file__).parent, species_file)
        species_frac = self.process_solver.get_raw_variables(fimp_no)

        t_ref, z_ref = np.genfromtxt(species_data).T[
            (0, 2),
        ]

        z_interp = interp1d(t_ref, z_ref)

        z_val = z_interp(te)

        return species_frac * z_val**2

    def radiative_loss_function_values(self, te, species_file):
        """
        From the impurity data file, by interpolation, it returns the values
        relative to the radiative power loss function.
        For each impurity species, the radiative power loss function is
        calculated from quantum mechanics codes and tabulated in the ADAS database.

        Parameters
        ----------
        te: np.array
            electron temperature
        species_file: string
            impurity data file (.dat)

        Returns
        -------
        l_val: np.array [W m^3]
            local values of the radiative power loss function
        """
        t_ref, l_ref = np.genfromtxt(species_file).T[
            (0, 1),
        ]
        interp_func = interp1d(t_ref, l_ref)
        l_val = interp_func(te)

        return l_val

    def calculate_line_radiation_loss(self, ne, p_loss_f, fimp_id):
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
        fimp_no: string
            impurity fraction to find in MFILE.dat

        Returns
        -------
        rad_loss: np.array
            Line radiation losses [MW m^-3]
        """
        species_frac = self.process_solver.get_raw_variables(fimp_id)
        rad_loss = (species_frac * (ne**2) * p_loss_f) / 1e6

        return rad_loss


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
            self.process_params["rhopedn"] + self.process_params["rhopedt"]
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

        n_grad_ped0 = self.process_params["ne0"] - self.process_params["neped"]
        t_grad_ped0 = self.process_params["te0"] - self.process_params["teped"]

        rho_ratio_n = (
            1 - ((rho_core[i_interior] ** 2) / (self.rho_ped**2))
        ) ** self.process_params["alphan"]

        rho_ratio_t = (
            1
            - (
                (rho_core[i_interior] ** self.process_params["tbeta"])
                / (self.rho_ped ** self.process_params["tbeta"])
            )
        ) ** self.process_params["alphat"]

        ne_i = self.process_params["neped"] + (n_grad_ped0 * rho_ratio_n)
        te_i = self.process_params["teped"] + (t_grad_ped0 * rho_ratio_t)

        i_exterior = np.where((rho_core > self.rho_ped) & (rho_core <= 1))[0]

        n_grad_sepped = self.process_params["neped"] - self.process_params["nesep"]
        t_grad_sepped = self.process_params["teped"] - self.process_params["tesep"]

        rho_ratio = (1 - rho_core[i_exterior]) / (1 - self.rho_ped)

        ne_e = self.process_params["nesep"] + (n_grad_sepped * rho_ratio)
        te_e = self.process_params["tesep"] + (t_grad_sepped * rho_ratio)

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

        ax = plt.gca()
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
        entrance: float, float
            x, z coordinates of the radiation region starting point
        exit: float, float
            x, z coordinates of the radiation region ending point
        """
        if lfs:
            sep_loop = self.sep_lfs
        else:
            sep_loop = self.sep_hfs
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
            te_sep = self.process_params["tesep"]
        ne_sep = self.process_params["nesep"]
        if omp is True:
            te_sol, ne_sol = self.electron_density_and_temperature_sol_decay(
                te_sep, ne_sep
            )
        else:
            te_sol, ne_sol = self.electron_density_and_temperature_sol_decay(
                te_sep, ne_sep, lfs=False
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
            Btot_sep_mp = self.Btot_sep_omp
        else:
            r_sep_mp = self.r_sep_imp
            Bp_sep_mp = self.Bp_sep_imp
            Btot_sep_mp = self.Btot_sep_imp

        # magnetic field components at the local point
        Bp_p = self.transport_solver.eq.Bp(x_p, z_p)
        Bt_p = self.transport_solver.eq.Bt(x_p)
        B_p = np.hypot(Bp_p, Bt_p)

        # flux expansion
        f_p = (r_sep_mp * Bp_sep_mp) / (x_p * Bp_p)

        # Ratio between upstream and local temperature
        f_t = t_u / t_p

        # Local electron temperature
        n_p = self.process_params["nesep"] * f_t

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
        firstwall: Loop
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

        ax = plt.gca()
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


class ScrapeOffLayerSector(ScrapeOffLayer, Mathematics):
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
        firstwall_loop,
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
        firstwall_loop: loop
            first wall geometry
        pfr_ext: float [m]
            extention of the radiation region from the x-point
            towards private flux region
        rec_ext: float [m]
            extention of the recycling region,
            along the separatrix, from the target
        x_point_rad: boolean
            if True, it assumes there is no radiation at all
            in the recycling region
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
        if x_point_rad is False:
            ion_front_z = self.ion_front_distance(x_strike, z_strike, rec_ext=rec_ext)
            pfr_ext = ion_front_z

        z_main, z_pfr = self.x_point_radiation_z_ext(main_ext, pfr_ext, low_div)

        in_x, in_z, out_x, out_z = self.radiation_region_ends(z_main, z_pfr, lfs)

        reg_i = np.array(
            [
                self.radiation_region_points(f.loop, z_main, z_pfr, low_div)
                for f in flux_tubes
            ],
            dtype=object,
        )

        t_mp_prof, n_mp_prof = self.mp_electron_density_temperature_profiles(
            self.t_u, lfs
        )

        t_rad_in = self.random_point_temperature(
            in_x,
            in_z,
            self.t_u,
            self.q_u,
            firstwall_loop,
            lfs,
        )

        t_rad_out = self.target_temperature(
            self.q_u,
            self.t_u,
            lfs,
        )

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

        t_pol = np.array(
            [
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
            ],
            dtype=object,
        )

        n_pol = np.array(
            [
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
            ],
            dtype=object,
        )

        return t_pol, n_pol


class StepCore(Core, Mathematics):
    """
    Specific class for the core emission of STEP
    """

    def __init__(self, process_solver, transport_solver, config):
        super().__init__(process_solver, transport_solver, config)

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
                self.radiative_loss_function_values(self.te_mp, data)
                for data in self.species_data
            ]
        )

        # Line radiation loss. Mid-plane distribution through the SoL
        rad = np.array(
            [
                self.calculate_line_radiation_loss(self.ne_mp, loss, fi)
                for loss, fi in zip(loss_f, self.impurity_id)
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
        ne_pol = np.array(
            [
                self.flux_tube_pol_n(ft, n, core=True, x_point_rad=False)
                for ft, n in zip(flux_tubes, self.ne_mp)
            ],
            dtype=object,
        )

        # For each flux tube, poloidal temperature profile.
        te_pol = np.array(
            [
                self.flux_tube_pol_t(ft, t, core=True)
                for ft, t in zip(flux_tubes, self.te_mp)
            ],
            dtype=object,
        )

        # For each impurity species and for each flux tube,
        # poloidal distribution of the radiative power loss function.
        loss_f = np.array(
            [
                np.array(
                    [self.radiative_loss_function_values(t, data) for t in te_pol],
                    dtype=object,
                )
                for data in self.species_data
            ]
        )

        # For each impurity species and for each flux tube,
        # poloidal distribution of the line radiation loss.
        rad = np.array(
            [
                np.array(
                    [
                        self.calculate_line_radiation_loss(n, l_f, fi)
                        for n, l_f in zip(ne_pol, ft)
                    ],
                    dtype=object,
                )
                for ft, fi in zip(loss_f, self.impurity_id)
            ]
        )

        # Total line radiation loss along each flux tube
        total_rad = np.sum(rad, axis=0)

        self.plot_2d_map(flux_tubes, total_rad)


class StepScrapeOffLayer(ScrapeOffLayerSector, ScrapeOffLayer, Mathematics):
    """
    Specific class for the scrape-off layer emission of STEP
    """

    def __init__(self, process_solver, transport_solver, config, firstwall_loop):
        super().__init__(process_solver, transport_solver, config)

        self.params = ParameterFrame(self.base_default_params)
        self.params.update_kw_parameters(config, f"{self.__class__.__name__} input")

        """
        2D map of the line radiation loss in the scrape-off layer.
        """
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
        self.t_u, self.q_u = self.upstream_temperature(firstwall_loop)

        # temperature and density profiles along the flux tubes
        (
            t_and_n_pol_lfs_low,
            t_and_n_pol_lfs_up,
            t_and_n_pol_hfs_low,
            t_and_n_pol_hfs_up,
        ) = self.build_sol_profiles(firstwall_loop)

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
            firstwall_loop,
        )

    def build_sol_profiles(self, firstwall_loop):
        """
        Temperature and density profiles builder.
        For each scrape-off layer sector, it gives temperature
        and density profile along each flux tube.

        Parameters
        ----------
        firstwall_loop: loop
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
            "lfs_low": {
                "flux_tube": self.flux_tubes_lfs_low,
                "x_strike": self.x_strike_lfs,
                "z_strike": self.z_strike_lfs,
                "main_ext": 0.2,
                "wall_profile": firstwall_loop,
                "pfr_ext": None,
                "rec_ext": 0.5,
                "x_point_rad": False,
                "lfs": True,
                "low_div": True,
            },
            "lfs_up": {
                "flux_tube": self.flux_tubes_lfs_up,
                "x_strike": self.x_strike_lfs,
                "z_strike": self.z_strike_lfs,
                "main_ext": 0.2,
                "wall_profile": firstwall_loop,
                "pfr_ext": None,
                "rec_ext": 0.5,
                "x_point_rad": False,
                "lfs": True,
                "low_div": False,
            },
            "hfs_low": {
                "flux_tube": self.flux_tubes_hfs_low,
                "x_strike": self.x_strike_hfs,
                "z_strike": self.z_strike_hfs,
                "main_ext": 0.2,
                "wall_profile": firstwall_loop,
                "pfr_ext": None,
                "rec_ext": 0.5,
                "x_point_rad": False,
                "lfs": False,
                "low_div": True,
            },
            "hfs_up": {
                "flux_tube": self.flux_tubes_hfs_up,
                "x_strike": self.x_strike_hfs,
                "z_strike": self.z_strike_hfs,
                "main_ext": 0.2,
                "wall_profile": firstwall_loop,
                "pfr_ext": None,
                "rec_ext": 0.5,
                "x_point_rad": False,
                "lfs": False,
                "low_div": False,
            },
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
            loss[side] = np.array(
                [
                    np.array(
                        [self.radiative_loss_function_values(t, data) for t in t_pol],
                        dtype=object,
                    )
                    for data in self.species_data
                ]
            )

        # line radiation loss along the open flux tubes
        rad = {
            "lfs_low": {"density": t_and_n_pol_lfs_low[1], "loss": loss["lfs_low"]},
            "lfs_up": {"density": t_and_n_pol_lfs_up[1], "loss": loss["lfs_up"]},
            "hfs_low": {"density": t_and_n_pol_hfs_low[1], "loss": loss["hfs_low"]},
            "hfs_up": {"density": t_and_n_pol_hfs_up[1], "loss": loss["hfs_up"]},
        }
        for side, ft in rad.items():
            rad[side] = np.array(
                [
                    np.array(
                        [
                            self.calculate_line_radiation_loss(n, l_f, fi)
                            for n, l_f in zip(ft["density"], f)
                        ],
                        dtype=object,
                    )
                    for f, fi in zip(ft["loss"], self.impurity_id)
                ]
            )

        return rad["lfs_low"], rad["lfs_up"], rad["hfs_low"], rad["hfs_up"]

    def build_sol_radiation_map(
        self, rad_lfs_low, rad_lfs_up, rad_hfs_low, rad_hfs_up, firstwall_loop
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
        firstwall_loop: loop
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
            firstwall_loop,
        )
