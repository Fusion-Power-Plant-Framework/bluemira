import math
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

import bluemira.geometry.tools as geotools
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.constants import MU_0
from bluemira.base.look_and_feel import bluemira_error
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.placement import BluemiraPlacement
from bluemira.utilities.plot_tools import set_component_view


def parallel(quantities: np.array):
    return np.sum([1 / q for q in quantities]) ** -1


def serie(quantities: np.array):
    return np.sum(quantities)


class Material:
    @property
    def Young_moduli(self):
        """Return the material Young's moduli

        Note
        ----
            Not implemented. It raises an error.
        """
        bluemira_error(f"Young's moduli not implemented for {self.__class__.__name__}")
        return 0

    def resistance(self, *kwargs):
        """Calculate the resistance as function of temperature, magnetic field
        and residual resistance ratio.

        Note
        ----
            Not implemented. It raises an error.
        """
        bluemira_error(f"Resistance not implemented for {self.__class__.__name__}")
        return 0

    def specific_heat_capacity(self, *kwargs):
        """Calculate the specific heat capacity as function of temperature
        and cross section.

        Note
        ----
            Not implemented. It raises an error.
        """
        bluemira_error(
            f"Specific heat capacity not implemented for {self.__class__.__name__}"
        )
        return 0


class DummyInsulator(Material):
    @property
    def Young_moduli(self):
        return 12


class DummySteel(Material):
    @property
    def Young_moduli(self):
        return 205


class Copper(Material):
    def resistivity(self, T: float, B: float, RRR: float):
        """Calculate the resistivity as function of temperature, magnetic field
        and residual resistance ratio.

        Parameters
        ----------
            T: material temperature [K]
            B: magnetic field [T]
            RRR: residual resistance ratio [-]

        Returns
        -------
            float: resistivity at specified conditions

        Note
        ----
            Add reference for the used equations (just trusting L. Giannini at present)
        """

        rho1 = (1.171 * (10**-17) * (T**4.49)) / (
            1 + (4.5 * (10**-7) * (T**3.35) * (math.exp(-((50 / T) ** 6.428))))
        )
        rho2 = (
            (1.69 * (10**-8) / RRR)
            + rho1
            + 0.4531 * ((1.69 * (10**-8) * rho1) / (RRR * rho1 + 1.69 * (10**-8)))
        )

        A = np.log10(1.553 * (10**-8) * B / rho2)
        a = (
            -2.662
            + 0.3168 * A
            + 0.6229 * (A**2)
            - 0.1839 * (A**3)
            + 0.01827 * (A**4)
        )
        rhoCu = rho2 * (1 + (10**a))

        return rhoCu

    def specific_heat_capacity(self, T: float, cross_section: float):
        """Calculate the specific heat capacity as function of temperature
        and cross section.

        Parameters
        ----------
            T: material temperature [K]
            cross_section: material cross section [m²]

        Returns
        -------
            float: specific heat capacity

        Note
        ----
            Add reference for the used equations (just trusting L. Giannini at present)
        """
        density = 8960  # Kg/m^3
        cp300 = 3.454e6  # J/K/m^3 known data point at 300K
        gamma = 0.011  # J/K^2/Kg
        beta = 0.0011  # J/K^4/Kg
        c_plow = (beta * (T**3)) + (gamma * T)  # J/K-Kg low temperature range
        Cp_Cu = (
            1 / ((1 / cp300) + (1 / (c_plow * density)))
        ) * cross_section  # J/K/m^3 volumetric specific heat for the whole temperature range
        return Cp_Cu


class Nb3Sn(Material):
    def resistance(self, *kwargs):
        """Calculate the resistance as function of temperature, magnetic field
        and residual resistance ratio.

        Note
        ----
            Not implemented. It raises an error.
        """
        bluemira_error(f"Resistance not implemented for {self.__class__.__name__}")
        return 0

    def specific_heat_capacity(self, T: float, NB3SN: float):
        """Calculate the specific heat capacity as function of temperature
        and material fraction.

        Parameters
        ----------
            T: material temperature [K]
            NB3SN: material fraction [m²]

        Returns
        -------
            float: specific heat capacity

        Note
        ----
            Add reference for the used equations (just trusting L. Giannini at present)
        """
        gamma_Nb = 0.1  # J/K^2/Kg
        beta_Nb = 0.001  # J/K^4/Kg
        density_Nb = 8040  # Kg/m^3
        Cp300_Nb = 210  # J/K/Kg

        Cp_low_NC = (beta_Nb * (T**3)) + (gamma_Nb * T)  # J/K/Kg NORMAL

        Cp_Nb3Sn = 1 / ((1 / Cp300_Nb) + (1 / Cp_low_NC))

        Cp_Nb3Sn = Cp_Nb3Sn * density_Nb * NB3SN
        return Cp_Nb3Sn


@dataclass
class Nb3Sn_WST(Nb3Sn):
    Young_moduli = 0.1

    def critical_current(self, B: float, d_fili: float):
        """
        Calculate the critical current for Nb3Sn_WST material given the magnetic field
        and the diamiter of the strand

        Parameters
        ----------
            B: magnetic field
            d_fili: strand diamiter (in mm)

        Returns
        -------
            float: specific heat capacity

        Note
        ----
            Add reference for the used equations (just trusting L. Giannini at present)
        """
        strand_d = d_fili
        strand_A = np.pi * strand_d**2 / (4 * (1 + self.CunonCu))
        # area di superconduttore nello strand
        c_ = 1.0
        Ca1 = 50.06
        # Deviatoric strain
        Ca2 = 0.00
        # Deviatoric strain
        eps_0a = 0.00312
        # Hydrostatic strain
        eps_m = -0.00059
        # Thermal pre-strain
        Bc20max = 33.24
        # Maximum upper critical feld [T]
        Tc0max = 16.34
        # Maximum critical temperature[K]
        C = 83075 * strand_A
        # Pre-constant [AT]
        p = 0.593
        q = 2.156

        # inputs
        T = 4.2 + 1.5
        # temp
        int_eps = -0.55 / 100
        # intrinsic strain __ R&W -0.36 __ W&R -0.55

        # fit functions
        eps_sh = Ca2 * eps_0a / (np.sqrt(Ca1**2 - Ca2**2))
        s_eps = 1 + (
            Ca1
            * (
                np.sqrt(eps_sh**2 + eps_0a**2)
                - np.sqrt((int_eps - eps_sh) ** 2 + eps_0a**2)
            )
            - Ca2 * int_eps
        ) / (1 - Ca1 * eps_0a)
        Bc0_eps = Bc20max * s_eps
        Tc0_eps = Tc0max * (s_eps) ** (1 / 3)
        t = T / Tc0_eps
        BcT_eps = Bc0_eps * (1 - t ** (1.52))
        TcB_eps = Tc0max * (s_eps) ** (1 / 3) * (1 - B / Bc0_eps) ** (1 / 1.52)
        b = B / BcT_eps
        hT = (1 - t ** (1.52)) * (1 - t**2)
        fPb = b**p * (1 - b) ** q

        # critical values
        Ic_A = 1e6 * c_ * (C / B) * s_eps * fPb * hT
        Jc_sc = Ic_A / strand_A
        Je_strand = Ic_A / (np.pi * strand_d**2 / 4)

        return Jc_sc


@dataclass
class REBCO(Material):
    Young_moduli = 120

    def critical_current(self, B: float):
        """Calculate the critical current for the superconductor.

        Parameters
        ----------
            B: magnetic field [T]

        Returns
        -------
            float: critical current

        Note
        ----
            Add reference for the used equations (just trusting L. Giannini at present)
        """
        A_tapes = 0.4 * 1
        # mm2
        T = 6.2
        Tc = 92.83
        # K
        Birr0 = 120
        # T B ortogonale
        C = 12510
        # A T
        p = 0.5
        q = 1.7
        a = 1.52
        b = 2.33
        Birr = Birr0 * (1 - T / Tc) ** a
        Ic_REBCO = C / B * (Birr / Birr0) ** b * (B / Birr) ** p * (1 - B / Birr) ** q
        Jc = 1e6 * Ic_REBCO / A_tapes
        return Jc

####################################
class Cable:



####################################
class Conductor:
    def __init__(
        self,
        mat_ins: Material,
        mat_jacket: Material,
        mat_cable: Material,
        dx_ins: float,
        dy_ins: float,
        dx_jacket: float,
        dy_jacket: float,
        dx_cable: float,
        dy_cable: float,
        xc: float = 0,
        yc: float = 0,
        name: str = "",
    ):
        self.mat_ins = mat_ins
        self.mat_jacket = mat_jacket
        self.mat_cable = mat_cable
        self.dx_ins = dx_ins
        self.dy_ins = dy_ins
        self.dx_jacket = dx_jacket
        self.dy_jacket = dy_jacket
        self.dx_cable = dx_cable
        self.dy_cable = dy_cable
        self.xc = xc
        self.yc = yc
        self.name = name

        self.CunonCu = 1
        self.RRR_seg = 300
        self.RRR_inSC = 100

    @property
    def dx_conductor(self):
        return 2 * self.dx_ins + 2 * self.dx_jacket + self.dx_cable

    @property
    def dy_conductor(self):
        return 2 * self.dy_ins + 2 * self.dy_jacket + self.dy_cable

    @property
    def component(self):
        points_cable = np.array(
            [
                [self.xc - self.dx_cable / 2, self.yc - self.dy_cable / 2, 0],
                [self.xc + self.dx_cable / 2, self.yc - self.dy_cable / 2, 0],
                [self.xc + self.dx_cable / 2, self.yc + self.dy_cable / 2, 0],
                [self.xc - self.dx_cable / 2, self.yc + self.dy_cable / 2, 0],
            ]
        )
        wire_cable = geotools.make_polygon(points_cable.T, closed=True)
        face_cable = BluemiraFace(boundary=[wire_cable])

        points_jacket = points_cable + np.array(
            [
                [-self.dx_jacket, -self.dy_jacket, 0],
                [self.dx_jacket, -self.dy_jacket, 0],
                [self.dx_jacket, +self.dy_jacket, 0],
                [-self.dx_jacket, +self.dy_jacket, 0],
            ]
        )
        wire_jacket = geotools.make_polygon(points_jacket.T, closed=True)
        face_jacket = BluemiraFace(boundary=[wire_jacket, wire_cable])

        points_ins = points_jacket + np.array(
            [
                [-self.dx_ins, -self.dy_ins, 0],
                [self.dx_ins, -self.dy_ins, 0],
                [self.dx_ins, +self.dy_ins, 0],
                [-self.dx_ins, +self.dy_ins, 0],
            ]
        )
        wire_ins = geotools.make_polygon(points_ins.T, closed=True)
        face_ins = BluemiraFace(boundary=[wire_ins, wire_jacket])

        comp = Component(name=self.name)
        comp_cable = PhysicalComponent(name="cable", shape=face_cable, parent=comp)
        comp_jacket = PhysicalComponent(name="jacket", shape=face_jacket, parent=comp)
        comp_ins = PhysicalComponent(name="insulator", shape=face_ins, parent=comp)

        set_component_view(comp, "xy")
        return comp

    def plot_xy(self, axis=None, show=False):
        self.component.plot_2d(ax=axis, show=show)

    def radial_stress(self, B: float):
        """
        Calculate the radial stress on the conductor jacket due to the magnetic pressure
        on the TF coil leg

        Parameters
        ----------
            B: magnetic field [T]

        Returns
        -------
            float: radial stress on the conductor jacket

        Note
        ----
            to be checked
        """
        K_cable = self.mat_cable.Young_moduli * self.dx_cable / self.dy_cable
        K_jacket_lat = self.mat_jacket.Young_moduli * self.dx_jacket / self.dy_conductor
        K_jacket_top_bot = self.mat_jacket.Young_moduli * self.dx_cable / self.dx_jacket
        K_ins_lat = self.mat_ins.Young_moduli * self.dx_ins / self.dy_conductor
        K_ins_top_bot = self.mat_ins.Young_moduli * self.dx_conductor / self.dx_ins

        # K_cond = (
        #     2 * K_jacket_lat
        #     + 2 * K_ins_lat
        #     + (1 / K_cable + 2 * 1 / K_jacket_top_bot + 2 * 1 / K_ins_top_bot) ** -1
        # )

        K_cond = serie(
            [
                K_ins_lat,
                K_jacket_lat,
                parallel(
                    [
                        K_ins_top_bot,
                        K_jacket_top_bot,
                        K_cable,
                        K_jacket_top_bot,
                        K_ins_top_bot,
                    ]
                ),
                K_jacket_lat,
                K_ins_lat,
            ]
        )

        # stiffening ratio
        sr_jacket = 2 * K_jacket_lat / K_cond

        # fraction of jacket material in the conductor
        frac_jacket = self.dx_conductor / (self.dx_conductor - self.dx_cable)

        # magnetic pressure
        press = B**2 / 2 * MU_0

        # radial stress
        sigma_r = press * sr_jacket * frac_jacket

        return sigma_r


class SquareConductor(Conductor):
    def __init__(
        self,
        mat_ins: Material,
        mat_jacket: Material,
        mat_cable: Material,
        d_ins: float,
        d_jacket: float,
        d_cable: float,
        xc: float = 0,
        yc: float = 0,
        name: str = "",
    ):
        super().__init__(
            mat_ins=mat_ins,
            mat_jacket=mat_jacket,
            mat_cable=mat_cable,
            dx_ins=d_ins,
            dy_ins=d_ins,
            dx_jacket=d_jacket,
            dy_jacket=d_jacket,
            dx_cable=d_cable,
            dy_cable=d_cable,
            xc=xc,
            yc=yc,
            name=name,
        )


def rearrange_TF_conductors(
    Ri: float,
    Rk: float,
    n_sectors: int,
    cond: Conductor,
    n_conductors: int,
    dy_ps: float,
    dx0_wp: float,
    dx_min: float,
    n_turns_reduction: int,
):
    """
    Rearrange the total number of necessary conductors into the TF coil cross section

    Parameters
    ----------
        Ri: inner leg outer radius [m]
        Rk: inner leg inner radius [m]
        n_sectors: number of sectors
        n_conductors: number of supercoductors
        dy_ps: distance between inner leg outer face and winding pack outer face [m]
        dx0_wp: toroidal distance between inner leg lateral face and most outer layer in the winding pack
        dx_min: minimum toroidal distance between winding pack and tf coils lateral faces
        dx_wp_reduction: number of turns to be removed when calculating a new pancake

    Returns
    -------
        np.array: number of turns and layers for each "pancake"

    Note
    ----
        The final number of allocated superconductors could slightly differ from the one defined
        in n_conductors due to the necessity to close the final layer.
    """

    theta_TF = 2 * np.pi / n_sectors

    dx_case = 2 * Ri * np.tan(theta_TF / 2)

    # maximum toroidal dimension of the WP most outer pancake
    dx_WP = dx_case - 2 * dy_ps * np.tan(theta_TF / 2) - 2 * dx0_wp

    # number of conductors to be allocated
    remaining_conductors = n_conductors

    # maximum number of turns on the considered pancake
    n_turns_max = math.ceil(dx_WP / cond.dx_conductor)

    # array to allocate turns and layers for each pancake
    n_turns = []
    n_layers = []

    # temporary variables
    dy_case_temp = Ri - dy_ps
    dx_case_temp = 2 * dy_case_temp * np.tan(theta_TF / 2)
    n_turns_temp = n_turns_max

    # maximum number of internal iterations
    i_max = 50
    i = 0
    while i < i_max:
        dx_layer = n_turns_temp * cond.dx_conductor
        n_layers_temp = math.ceil(
            ((dx_case_temp / 2 - dx_layer / 2 - dx_min) / np.tan(theta_TF / 2))
            / cond.dy_conductor
        )

        n_spire_temp = n_layers_temp * n_turns_temp

        if n_spire_temp >= remaining_conductors:
            n_layers_temp = math.ceil(remaining_conductors / n_turns_temp)
            i = i_max

        dy_layer = n_layers_temp * cond.dy_conductor

        n_turns.append(n_turns_temp)
        n_layers.append(n_layers_temp)

        remaining_conductors = remaining_conductors - n_turns_temp * n_layers_temp

        print(
            f"{n_turns_temp} - {n_layers_temp} --- remaining conductors: {remaining_conductors}"
        )

        n_turns_temp = n_turns_temp - n_turns_reduction

        dy_case_temp = dy_case_temp - dy_layer

        if dy_case_temp <= Rk:
            bluemira_error("There is not enough space to allocate all the conductors")
            i = i_max
        else:
            dx_case_temp = 2 * dy_case_temp * np.tan(theta_TF / 2)
            i = i + 1

    return np.array(n_turns), np.array(n_layers)


def create_wp_component(cond: Conductor, n_turns: np.array, n_layers: np.array):
    """
    Create a component with the geometry of the winding pack

    Parameters
    ----------
        cond: the superconductor cable
        n_turns: number of turns
        n_layers: number of layers

    Returns
    -------
        a component with the winding pack geometry
    """

    dx = cond.dx_conductor
    dy = cond.dy_conductor

    wp_comp = Component(name="WP")

    x0 = 0
    y0 = 0
    for i in range(len(n_turns)):
        points = np.array(
            [
                [x0 - dx * n_turns[i] / 2, y0, 0],
                [x0 + dx * n_turns[i] / 2, y0, 0],
                [x0 + dx * n_turns[i] / 2, y0 - dy * n_layers[i], 0],
                [x0 - dx * n_turns[i] / 2, y0 - dy * n_layers[i], 0],
            ]
        )

        wire_wp_layer = geotools.make_polygon(points=points, closed=True)
        PhysicalComponent(name="pancake" + str(i), shape=wire_wp_layer, parent=wp_comp)
        y0 = y0 - dy * n_layers[i]

    set_component_view(wp_comp, "xy")

    return wp_comp


def create_tf_cross_section_comp(
    Ri: float,
    Rk: float,
    n_sectors: int,
    cond: Conductor,
    n_conductors: int,
    dy_ps: float,
    dx0_wp: float,
    dx_min: float,
    n_turns_reduction: int,
):
    """
    Create a component with the geometry of the TF coil cross section geometry

    Parameters
    ----------
        Ri: inner leg outer radius [m]
        Rk: inner leg inner radius [m]
        n_sectors: number of sectors
        n_conductors: number of supercoductors
        dy_ps: distance between inner leg outer face and winding pack outer face [m]
        dx0_wp: toroidal distance between inner leg lateral face and most outer layer in the winding pack
        dx_min: minimum toroidal distance between winding pack and tf coils lateral faces
        dx_wp_reduction: number of turns to be removed when calculating a new pancake

    Returns
    -------
        np.array: number of turns and layers for each "pancake"

    Note
    ----
        The final number of allocated superconductors could slightly differ from the one defined
        in n_conductors due to the necessity to close the final layer.
    """

    tf_cross_section = Component(name="TF_cross_section")

    theta_TF = 2 * np.pi / n_sectors

    dxi_case = 2 * Ri * np.tan(theta_TF / 2)
    dxk_case = 2 * Rk * np.tan(theta_TF / 2)

    points = np.array(
        [
            [-dxi_case / 2, Ri, 0],
            [dxi_case / 2, Ri, 0],
            [dxk_case / 2, Rk, 0],
            [-dxk_case / 2, Rk, 0],
        ]
    )

    wire_case_out = geotools.make_polygon(points=points, closed=True)
    PhysicalComponent(name="Case", shape=wire_case_out, parent=tf_cross_section)

    n_turns, n_layers = rearrange_TF_conductors(
        Ri,
        Rk,
        n_sectors,
        cond,
        n_conductors,
        dy_ps,
        dx0_wp,
        dx_min,
        n_turns_reduction,
    )

    set_component_view(tf_cross_section, "xy")

    wp_comp = create_wp_component(cond, n_turns, n_layers)
    wp_comp.parent = tf_cross_section
    wp_comp_placement = BluemiraPlacement(
        base=[0, Ri - dy_ps, 0], axis=[0, 0, 1], angle=0.0
    )
    set_component_view(wp_comp, wp_comp_placement)

    return tf_cross_section


##### Inputs
# field on the axis
B0 = 9.4
# major radius
R0 = 8.5
# aspect ratio
A = 4.0
# TF number
n_TF = 16
# SB+VV+gap
d = 1.6
# max allowable ripple in the plasma region
rho = 0.006


# minor radius
a = R0 / A
# inner leg outer radius
Ri = R0 - a - d
# outer leg inner radius
Re = (R0 + a) * (1 + 1.0 / rho) ** (1 / n_TF)

####
S_amm = 667e6
# Steel allowable limit
R_VV = Ri * 1.05
# Vacuum vessel radius
S_VV = 100e6
# Vacuum vessel steel limit

###
Rk = Ri / 2
n_spire = 300
dx_min = 0.05
dy_ps = 0.06
dx0_wp = 0.15
n_turns_reduction = 4

########
theta_TF = 2 * np.pi / n_TF
# Case width (toroidal direction)
dx_case = 2 * Ri * np.tan(theta_TF / 2)

dr_plasma_side = R0 * 2 / 3 * 1e-2
gap = R0 * 2 / 3 * 1e-2

# Total TF current [MA]
NI = (2 * np.pi * R0 * B0 / MU_0) / n_TF
# Toroidal field peak
sf = 1.08
BI_TF = MU_0 * n_TF * NI / (2 * np.pi * Ri) * sf

# Conductor operational current
Iop = 80e3

# Number of conductors
n_spire = np.floor(NI / Iop)
L = MU_0 * R0 * (n_TF * n_spire) ** 2 * (1 - np.sqrt(1 - (R0 - Ri) / R0)) / n_TF * 1.1
E = 1 / 2 * L * n_TF * Iop**2 * 1e-6
V_MAX = (7 * R0 - 3) / 6 * 1.1e3
# [V]
Tau_discharge1 = L * Iop / V_MAX
# [s] - scarico in gruppi di tre bobine
Tau_discharge2 = B0 * NI * n_TF * (R0 / A) ** 2 / (R_VV * S_VV)
Tau_discharge = max([Tau_discharge1, Tau_discharge2, 4])

# materials
mat_insulator = DummyInsulator()
mat_jacket = DummySteel()
mat_case = DummySteel()
cu = Copper()
nb3sn = Nb3Sn_WST()
rebco = REBCO()


c1 = SquareConductor(
    mat_ins=mat_insulator,
    mat_jacket=mat_jacket,
    mat_cable=nb3sn,
    d_ins=0.001,
    d_jacket=0.01,
    d_cable=0.05,
)
c1.plot_xy(show=True)


tf_cross_section = create_tf_cross_section_comp(
    Ri, Rk, n_TF, c1, n_spire, dy_ps, dx0_wp, dx_min, n_turns_reduction
)


flag_plot_tf = 1

if flag_plot_tf == 1:
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.axis("equal")
    TitleName = "TF coil Winding Pack"
    ax.set_title(TitleName, fontsize=20, fontweight="bold")
    ax.set_xlabel("[m]", fontsize=14)
    ax.set_ylabel("[m]", fontsize=14)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.set_ylim([0, Ri * 1.3])
    tf_cross_section.plot_2d(ax=ax, show=False)

    tf_case_bounding_box = tf_cross_section.get_component(name="Case").shape.bounding_box

    dx_case = tf_case_bounding_box.x_max - tf_case_bounding_box.x_min
    ax.text(0, Ri * 1.2, f"Case(w)={round(dx_case, 2)} m", fontsize=12)
    ax.plot(
        [tf_case_bounding_box.x_min, tf_case_bounding_box.x_max],
        [Ri * 1.175, Ri * 1.175],
        "--.r",
        linewidth=1,
    )

    tf_wp_bounding_box = tf_cross_section.get_component(
        name="pancake0"
    ).shape.bounding_box
    dx_wp = tf_wp_bounding_box.x_max - tf_wp_bounding_box.x_min
    ax.text(0, Ri * 1.1, f"WP(w)={round(dx_wp, 2)} m", fontsize=12)
    ax.plot(
        [tf_wp_bounding_box.x_min, tf_wp_bounding_box.x_max],
        [Ri * 1.075, Ri * 1.075],
        "--.r",
        linewidth=1,
    )

    dx_ = -dx_case
    ax.plot([dx_, dx_], [0, tf_case_bounding_box.y_max], "--.r", linewidth=1)
    ax.text(
        dx_,
        tf_case_bounding_box.y_max,
        f"Ri={round(tf_case_bounding_box.y_max, 2)} m",
        fontsize=12,
    )

    dx_ = -dx_case
    ax.plot([dx_, dx_], [0, tf_case_bounding_box.y_min], "--.r", linewidth=1)
    ax.text(dx_, Rk, f"Rk={round(tf_case_bounding_box.y_min, 2)} m", fontsize=12)

    plt.show()
