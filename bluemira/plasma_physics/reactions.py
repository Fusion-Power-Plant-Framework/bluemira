# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2022 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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
Fusion reactions
"""

from dataclasses import dataclass

import numpy as np

from bluemira.base.constants import (
    AMU_TO_KG,
    C_LIGHT,
    D_MOLAR_MASS,
    ELECTRON_MOLAR_MASS,
    EV_TO_J,
    HE3_MOLAR_MASS,
    HE_MOLAR_MASS,
    J_TO_EV,
    N_AVOGADRO,
    NEUTRON_MOLAR_MASS,
    PROTON_MOLAR_MASS,
    T_MOLAR_MASS,
    raw_uc,
)


def E_DT_fusion():  # noqa :N802
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
    return delta_m * C_LIGHT**2 * AMU_TO_KG * J_TO_EV


def E_DD_fusion():  # noqa :N802
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
    """  # noqa :W505
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
    return delta_m * C_LIGHT**2 * AMU_TO_KG * J_TO_EV


def n_DT_reactions(p_fus) -> float:
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
    return raw_uc(p_fus, "MW", "W") / (e_dt * EV_TO_J)


def n_DD_reactions(p_fus) -> float:  # noqa :N802
    """
    Calculates the number of D-D fusion reactions per s for a given D-D fusion
    power

    :math:`n_{reactions} = \\frac{P_{fus}[MW]}{E_{DD} [MeV] eV[J]} [1/s]`

    Parameters
    ----------
    p_fus: float
        D-D fusion power [W]

    Returns
    -------
    n_reactions: float
        Number of D-D reactions per second [1/s]
    """
    e_dd = E_DD_fusion()
    return p_fus / (e_dd * EV_TO_J)


def r_T_burn(p_fus):  # noqa :N802
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
    """  # noqa :W505
    return n_DT_reactions(p_fus) * T_MOLAR_MASS / N_AVOGADRO


def r_D_burn_DT(p_fus):  # noqa :N802
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


REACTIONS = {
    "D-T": 0,  # D + T --> 4He + n reaction
    "D-D": -1,  # D + D --> 0.5 D-D1 + 0.5 D-D2
    "D-D1": 1,  # D + D --> 3He + n reaction [50 %]
    "D-D2": 2,  # D + D --> T + p reaction [50 %]
    "D-He3": 3,  # D + 3He --> 4He + p reaction
}


def reactivity(temp_kev, reaction="D-T", method="Bosch-Hale"):
    """
    Calculate the thermal reactivity of a fusion reaction in Maxwellian plasmas,
    \\t:math:`<\\sigma v>`

    Parameters
    ----------
    temp_kev: float
        Temperature [keV]
    reaction: str
        The fusion reaction
    method: str
        The parameterisation to use when calculating the reactivity

    Returns
    -------
    sigma_v: float
    """
    if reaction not in REACTIONS:
        raise ValueError(f"Unknown reaction: {reaction}")

    mapping = {
        "Bosch-Hale": _bosch_hale,
    }
    if method not in mapping:
        raise ValueError(f"Unknown method: {method}")

    func = mapping[method]
    return func(temp_kev, reaction)


@dataclass
class BoschHale_DT_4Hen:
    """
    Bosch-Hale parameterisation data for the reaction:

    D + T --> 4He + n
    """

    t_min = 0.2  # [keV]
    t_max = 100  # [keV]
    bg = 34.3827  # [keV**0.5]
    mrc2 = 1.124656e6  # [keV]
    c = np.array(
        [
            1.17302e-9,
            1.51361e-2,
            7.51886e-2,
            4.60643e-3,
            1.35000e-2,
            -1.06750e-4,
            1.36600e-5,
        ]
    )


@dataclass
class BoschHale_DD_3Hen:
    """
    Bosch-Hale parameterisation data for the reaction:

    D + D --> 3He + n
    """

    t_min = 0.2  # [keV]
    t_max = 100  # [keV]
    bg = 31.3970  # [keV**0.5]
    mrc2 = 0.937814e6  # [keV]
    c = np.array(
        [
            5.43360e-12,
            5.85778e-3,
            7.68222e-3,
            0.0,
            -2.96400e-6,
            0.0,
            0.0,
        ]
    )


@dataclass
class BoschHale_DD_Tp:
    """
    Bosch-Hale parameterisation data for the reaction:

    D + D --> T + p
    """

    t_min = 0.2  # [keV]
    t_max = 100  # [keV]
    bg = 31.3970  # [keV**0.5]
    mrc2 = 0.937814e6  # [keV]
    c = np.array(
        [
            5.65718e-12,
            3.41267e-3,
            1.99167e-3,
            0.0,
            1.05060e-5,
            0.0,
            0.0,
        ]
    )


@dataclass
class BoschHale_DHe3_4Hep:
    """
    Bosch-Hale parameterisation data for the reaction:

    D + 3He --> 4He + p
    """

    t_min = 0.5  # [keV]
    t_max = 190  # [keV]
    bg = 68.7508  # [keV**0.5]
    mrc2 = 1.124572e6  # [keV]
    c = np.array(
        [
            5.51036e-10,
            6.41918e-3,
            -2.02896e-3,
            -1.91080e-5,
            1.35776e-4,
            0.0,
            0.0,
        ]
    )


def _bosch_hale(temp_kev, reaction):
    if reaction == "D-D":
        return 0.5 * (_bosch_hale(temp_kev, "D-D1") + _bosch_hale(temp_kev, "D-D2"))
    mapping = {
        "D-T": BoschHale_DT_4Hen,
        "D-D1": BoschHale_DD_3Hen,
        "D-D2": BoschHale_DD_Tp,
        "D-He3": BoschHale_DHe3_4Hep,
    }
    data = mapping[reaction]
    frac = (data.c[1] + temp_kev * (data.c[3] + temp_kev * data.c[5])) / (
        1 + temp_kev * (data.c[2] + temp_kev * (data.c[4] + temp_kev * data.c[6]))
    )
    theta = temp_kev / (1 - temp_kev * frac)
    chi = (data.bg**2 / (4 * theta)) ** (1 / 3)
    return (
        1e-6 * data.c[0] * np.sqrt(chi / (data.mrc2 * temp_kev**3)) * np.exp(-3 * chi)
    )
