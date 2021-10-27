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
A collection of useful 0-D physics calculations
"""
import numpy as np
from typing import Union

from bluemira.base.look_and_feel import bluemira_warn
from BLUEPRINT.utilities.tools import PowerLawScaling
from bluemira.base.constants import (
    C_LIGHT,
    T_MOLAR_MASS,
    D_MOLAR_MASS,
    HE_MOLAR_MASS,
    NEUTRON_MOLAR_MASS,
    J_TO_EV,
    EV_TO_J,
    AMU_TO_KG,
    PROTON_MOLAR_MASS,
    HE3_MOLAR_MASS,
    ELECTRON_MOLAR_MASS,
    N_AVOGADRO,
)


def estimate_kappa95(A, m_s_limit):
    """
    Estimate the maximum kappa_95 for a given aspect ratio and margin to
    stability. It is always better to have as high a kappa_95 as possible, so
    we maximise it here, for a specified margin to stability value.

    Parameters
    ----------
    A: float
        The aspect ratio of the plasma
    m_s_limit: float
        The margin to stability (typically ~0.3)

    Returns
    -------
    kappa_95: float
        The maximum elongation for the specified input values

    Notes
    -----
    The model used here is a 2nd order polynomial surface fit, generated using
    data from CREATE. A quadratic equation is then solved for kappa_95, based
    on the polynomial surface fit.
    The data are stored in: data/equilibria/vertical_stability_data.json

    This is only a crude model, and is only relevant for EU-DEMO-like machines.

    Furthermore, this is only for flat-top..! Ramp-up and ramp-down may be
    design driving. Exercise caution.

    \t:math:`m_{s} = a\\kappa_{95}^{2}+bA^{2}+c\\kappa A+d\\kappa+eA+f`\n
    \t:math:`\\kappa_{95}(A, m_{s}) = \\dfrac{-d-cA-\\sqrt{(c^{2}-4ab)A^{2}+(2dc-4ae)A+d^{2}-4af+4am_{s})}}{2a}`
    """  # noqa (W505)
    a = 8.39148185
    b = -0.17713049
    c = 1.9031585
    d = -37.17364535
    e = -2.54598909
    f = 38.75101822

    kappa_95 = (
        -d
        - c * A
        - np.sqrt(
            (c ** 2 - 4 * a * b) * A ** 2
            + (2 * d * c - 4 * a * e) * A
            + d ** 2
            - 4 * a * f
            + 4 * a * m_s_limit
        )
    ) / (2 * a)

    # Include power correction for more conservative extrapolation from low
    # number of data points
    return kappa_95 ** 0.98


# TODO UPDATE:  EF says this is wrong or not the best
def plasma_resistance(R_0, A, z_eff, kappa, t_e):
    """
    Plasma resistance, from loop voltage calculation in IPDG89 (??)

    Neo-classical resistivity enhancement factor: rpfac

    Parameters
    ----------
    R_0: float
        The plasma major radius
    A: float
        The plasma aspect ratio
    z_eff: float
        Plasma effective charge (density?!) [units??]
    t_e: float
        Plasma density weighted average plasma temperature [keV]

    Returns
    -------
    r_plasma: float
        The plasma resistance [Ohm]

    Notes
    -----
    The Uckan et al. [1] expression is valid for aspect ratios in the range
    2.5 to 4.

    \t:math:`R_{plasma} = \\dfrac{2.15e^{9}Z_{eff}R_{0}}{\\kappa a^{2}(T_{e}/10)^{1.5}}`

    [1] N. A. Uckan et al, Fusion Technology 13 (1988) p.411.
        The expression is valid for aspect ratios in the range 2.5 to 4.
    """  # noqa (W505)
    a = R_0 / A
    r_plasma = 2.15e-9 * z_eff * R_0 / (kappa * a ** 2 * (t_e / 10) ** 1.5)
    rpfac = 4.3 - 0.6 * R_0 / a
    return r_plasma * rpfac


def calc_P_ohmic(f_ind, r_plasma, Ip):  # noqa (N802)
    """
    Berechnet die Ohmische Heizung des Plasmas

    Parameters
    ----------
    f_ind: 0 < float < 1
        The fraction of inductive current drive
    r_plasma: float
        The total resistance of the plasma [Ohm]
    Ip: float
        The plasma current [MA]

    Returns
    -------
    P_ohm: float
        The Ohmic heating power deposited in the plasma [MW]
    """
    return f_ind * r_plasma * (Ip * 1e6) ** 2 / 1e6


def joule_heating(rho, length, area, current):
    """
    La puissance thermique de Joule pour un conducteur

    Parameters
    ----------
    rho: float
        The specific resistivity [Ohms.m]
    length: float
        The length of the conductor [m]
    area: float
        The cross-sectional area of the conductor [m^2]
    current: float
        The current in the conductor [A]

    Returns
    -------
    P: float
        The total resistive Joule heating in the conductor [W]
    """
    return rho * length / area * current ** 2


def IPB98y2(Ip, b_tor, p_sep, n19, R_0, A, kappa):  # noqa (N802)
    """
    ITER IPB98(y, 2) Confinement time scaling [2]

    Parameters
    ----------
    Ip: float
        Plasma current [MA]
    b_tor: float
        Toroidal field at R_0 [T]
    p_sep: float
        Separatrix power [MW]
    n19: float
        Line average plasma density [10^19 1/m^3]
    R_0: float
        Major radius [m]
    A: float
        Aspect ratio
    kappa: float
        Plasma elongation

    Returns
    -------
    tau_E: float
        The energy confinement time [s]

    Notes
    -----
    [2] ITER Physics Expert Group, Nucl. Fus. 39, 12, <https://iopscience.iop.org/article/10.1088/0029-5515/39/12/302/pdf>

    \t:math:`\\tau_{E}=0.0562I_p^{0.93}B_t^{0.15}P_{sep}^{-0.69}n^{0.41}M^{0.19}R_0^{1.97}A^{-0.57}\\kappa^{0.78}`
    """  # noqa (W505)
    bluemira_warn("IPB98y2 parameterisation possibly incorrect!")
    m_t = T_MOLAR_MASS - ELECTRON_MOLAR_MASS
    m_d = D_MOLAR_MASS - ELECTRON_MOLAR_MASS
    m_he = HE_MOLAR_MASS - 2 * ELECTRON_MOLAR_MASS
    mass = np.average([m_t, m_d, m_he])
    law = PowerLawScaling(
        c=0.0562, exponents=[0.93, 0.15, -0.69, 0.41, 0.19, 1.97, -0.58, 0.78]
    )
    return law(Ip, b_tor, p_sep, n19, mass, R_0, A, kappa)


def separatrix_power(p_fus, p_aux, p_rad):
    """
    0-D relation for separatrix power

    \t:math:`P_{sep} = P_{fus}/5+P_{aux}-P_{rad}`
    """
    return p_fus / 5 + p_aux - p_rad


def P_LH(n_e20, b_tor, a, R_0, error=False):  # noqa (N802)
    """
    Power requirement for accessing H-mode, Martin scaling [3]

    Parameters
    ----------
    n_e20: float
        Density [1/m^3]
    b_tor: float
        Toroidal field at the major radius [T]
    a: float
        Plasma minor radius [m]
    R_0: float
        Plasma major radius [m]

    Returns
    -------
    P_LH: float
        Power required to access H-mode [MW]

    Notes
    -----
    [3] Martin et al., 2008,
    <https://infoscience.epfl.ch/record/135655/files/1742-6596_123_1_012033.pdf>

    \t:math:`P_{LH}=2.15e^{\\pm 0.107}n_{e20}^{0.782 \\pm 0.037}`
    \t:math:`B_{T}^{0.772 \\pm 0.031}a^{0.975 \\pm 0.08}R_{0}^{0.999 \\pm 101}`
    """  # noqa (W505)
    law = PowerLawScaling(
        c=2.15,
        cerr=0,
        cexperr=0.107,
        exponents=[0.782, 0.772, 0.0975, 0.999],
        err=[0.037, 0.031, 0.08, 0.101],
    )
    n_e20 /= 1e20
    if error:
        return list(law.error(n_e20, b_tor, a, R_0))
    else:
        return law(n_e20, b_tor, a, R_0)


def lambda_q(Bt, q_95, p_sol, R_0, error=False):
    """
    Scrape-off layer power width scaling (Eich, 2013) [4]

    \t:math:`\\lambda_q=(0.7\\pm0.2)B_t^{-0.77\\pm0.14}q_{95}^{1.05\\pm0.13}P_{SOL}^{0.09\\pm0.08}R_{0}^{0\\pm0.14}`

    Parameters
    ----------
    Bt: float
        Toroidal field [T]
    q_95: float
        Safety factor at the 95th percentile
    p_sol: float
        Power in the scrape-off layer [MW]
    R_0: float
        Major radius [m]

    Returns
    -------
    lambda_q: float
        Scrape-off layer width at the outboard midplane [m]

    Notes
    -----
    [4] Eich, 2013, <http://iopscience.iop.org/article/10.1088/0029-5515/53/9/093031/meta>
    For conventional aspect ratios
    """  # noqa (W505)
    law = PowerLawScaling(
        c=0.0007,
        cerr=0.0002,
        exponents=[-0.77, 1.05, 0.09, 0],
        err=[0.14, 0.13, 0.08, 0.14],
    )
    if error:
        return list(law.error(Bt, q_95, p_sol, R_0))
    else:
        return law(Bt, q_95, p_sol, R_0)


def MeV_to_MW(p_fus, e_per_neutron):  # noqa (N802)
    """
    Converts MeV per birth neutron to MW of power

    Parameters
    ----------
    p_fus: float
        The fusion power [MW]
    e_per_neutron: float
        The energy per birth neutron [MeV]

    Returns
    -------
    P_n: float
        The total neutronic power [MW]
    """
    return e_per_neutron * n_DT_reactions(p_fus) * EV_TO_J


def E_DT_fusion():  # noqa (N802)
    """
    Calculates the total energy released from the D-T fusion reaction

    Returns
    -------
    delta_E: float
        The energy released from the D-T fusion reaction [eV]

    Notes
    -----
    .. math::
        {^{2}_{1}H}+{^{3}_{1}H}~\\rightarrow~{^{4}_{2}He}~
        (3.5~\\text{MeV})+\\text{n}^{0} (14.1 ~\\text{MeV})\n
        \\Delta E = \\Delta m c^2
    """
    delta_m = (D_MOLAR_MASS + T_MOLAR_MASS) - (HE_MOLAR_MASS + NEUTRON_MOLAR_MASS)
    return delta_m * C_LIGHT ** 2 * AMU_TO_KG * J_TO_EV


def E_DD_fusion():  # noqa (N802)
    """
    Calculates the total energy released from the D-D fusion reaction

    Returns
    -------
    delta_E: float
        The energy released from the D-D fusion reaction [eV]

    Notes
    -----
    .. math::
        {^{2}_{1}H}+{^{2}_{1}H}~\\rightarrow~{^{3}_{1}H}
        (1.01 ~\\text{MeV})+\\text{p} (3.02~\\text{MeV})~~[50 \\textrm{\\%}]
        ~~~~~~~~~~\\rightarrow~{^{3}_{2}He} (0.82~\\text{MeV})+\\text{n}^{0} (2.45~\\text{MeV})~~[50 \\text{\\%}]\n
        \\Delta E = \\Delta m c^2
    """  # noqa (W505)
    # NOTE: Electron mass must be included with proton mass
    delta_m = np.array(
        [
            D_MOLAR_MASS
            + D_MOLAR_MASS
            - (T_MOLAR_MASS + PROTON_MOLAR_MASS + ELECTRON_MOLAR_MASS),
            (D_MOLAR_MASS + D_MOLAR_MASS) - (HE3_MOLAR_MASS + NEUTRON_MOLAR_MASS),
        ]
    )
    delta_m = np.average(delta_m)
    return delta_m * C_LIGHT ** 2 * AMU_TO_KG * J_TO_EV


def n_DT_reactions(p_fus: Union[int, float]) -> float:
    """
    Calculates the number of D-T fusion reactions per s for a given D-T fusion
    power

    :math:`n_{reactions} = \\frac{P_{fus}[MW]}{17.58 [MeV]eV[J]} [1/s]`

    Parameters
    ----------
    p_fus: float
        D-T fusion power [MW]

    Returns
    -------
    n_reactions: float
        Number of D-T reactions per second [1/s]
    """
    e_dt = E_DT_fusion()
    return float(p_fus * 1e6 / (e_dt * EV_TO_J))


def n_DD_reactions(p_fus: Union[int, float]) -> float:  # noqa (N802)
    """
    Calculates the number of D-D fusion reactions per s for a given D-D fusion
    power

    :math:`n_{reactions} = \\frac{P_{fus}[MW]}{E_{DD} [MeV] eV[J]} [1/s]`

    Parameters
    ----------
    p_fus: float
        D-D fusion power [MW]

    Returns
    -------
    n_reactions: float
        Number of D-D reactions per second [1/s]
    """
    e_dd = E_DD_fusion()
    return float(p_fus * 1e6 / (e_dd * EV_TO_J))


def r_T_burn(p_fus):  # noqa (N802)
    """
    Calculates the tritium burn rate for a given fusion power

    :math:`\\dot{m_{b}} = \\frac{P_{fus}[MW]M_{T}[g/mol]}{17.58 [MeV]eV[J]N_{A}[1/mol]} [g/s]`

    Parameters
    ----------
    p_fus: float
        D-T fusion power [MW]

    Returns
    -------
    r_burn: float
        T burn rate in the plasma [g/s]
    """  # noqa (W505)
    return n_DT_reactions(p_fus) * T_MOLAR_MASS / N_AVOGADRO


def r_D_burn_DT(p_fus):  # noqa (N802)
    """
    Calculates the deuterium burn rate for a given fusion power in D-T

    Parameters
    ----------
    p_fus: float
        D-T fusion power [MW]

    Returns
    -------
    r_burn: float
        D burn rate in the plasma [g/s]

    Notes
    -----
    .. math::
        \\dot{m_{b}} = \\frac{P_{fus}[MW]M_{D}[g/mol]}
        {17.58 [MeV]eV[J]N_{A}[1/mol]} [g/s]
    """
    return n_DT_reactions(p_fus) * D_MOLAR_MASS / N_AVOGADRO


def r_D_burn_DD(p_fus):  # noqa (N802) # TODO: FIX
    """
    Calculates the deuterium burn rate for a given fusion power in D-D

    Parameters
    ----------
    p_fus: float
        D-D fusion power [MW]

    Returns
    -------
    r_burn: float
        D burn rate in the plasma [g/s]

    Notes
    -----
    .. math::
        \\dot{m_{b}} = 2\\frac{P_{fus}[MW]M_{D}[g/mol]}
        {17.58 [MeV]eV[J]N_{A}[1/mol]} [g/s]
    """
    print("NOT CHECKED YET DIPSHTI")
    return 2 * n_DD_reactions(p_fus) * D_MOLAR_MASS / N_AVOGADRO
