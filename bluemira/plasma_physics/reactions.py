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
        D-D fusion power [MW]

    Returns
    -------
    n_reactions: float
        Number of D-D reactions per second [1/s]
    """
    e_dd = E_DD_fusion()
    return raw_uc(p_fus, "MW", "W") / (e_dd * EV_TO_J)


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
