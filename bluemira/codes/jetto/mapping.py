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

# BLUEPRINT -> JETTO
# with JETTO short descriptions

from bluemira.base.parameter import ParameterMapping

mappings = {
    "BM_INP": ParameterMapping("q_min", True, False),  # minimum safety factor
    "q_95": ParameterMapping("q_95", True, False),  # edge safety factor
    "BM_INP": ParameterMapping("n_GW", True, False),  # Greenwalf fraction
    "BM_INP": ParameterMapping("n_GW_95", True, False),  # Greenwald fraction @rho=0.95
    "BM_INP": ParameterMapping("s", True, False),  # magnetic shear
    "BM_INP": ParameterMapping(None, True, False),  # Troyon limit
    "BM_INP": ParameterMapping(None, True, False),  # resistive wall mode (no wall) limit
    "beta_N": ParameterMapping("beta_N", True, False),  # Normalised Beta
    "l_i": ParameterMapping("li3", True, False),  # Internal inductance   - UNSURE
    "BM_INP": ParameterMapping("dWdt", True, False),  # ...
    "BM_INP": ParameterMapping(None, True, False),  # Fast particle pressure
    "kappa": ParameterMapping("kappa", True, False),  # Elongaton
    "delta": ParameterMapping("delta", True, False),  # Triangularity (absolute value)
    "B_0": ParameterMapping("Btor", True, False),  # Toroidal field on axis
    "I_p": ParameterMapping("Ip", True, False),  # Plasma current
    "A": ParameterMapping("A", True, False),  # Aspect Ratio
    "BM_INP": ParameterMapping(
        "H_98", True, False
    ),  # confinement relative to scaling law
    "BM_INP": ParameterMapping("alpha", True, False),  # Normalised pedistal pressure
    "BM_INP": ParameterMapping("P_fus", True, False),  # fusion power
    "BM_INP": ParameterMapping("f_boot", True, False),  # bootstrap fraction
    "BM_INP": ParameterMapping("P_aux", True, False),  # Heating and CD power (coupled)
    "BM_INP": ParameterMapping("I_aux", True, False),  # Current drive
    "BM_INP": ParameterMapping("eta_CD", True, False),  # Current drive efficiency
    "BM_INP": ParameterMapping("Q_fus", True, False),  # fusion gain
    "BM_INP": ParameterMapping("Q_eng", True, False),  # Net energy gain
    "BM_INP": ParameterMapping("P_sep", True, False),  # Power crossing separatrix
    "BM_INP": ParameterMapping("f_rad", True, False),  # Radiation fraction
    "BM_INP": ParameterMapping("Q_max", True, False),  # target heat load
    "BM_INP": ParameterMapping(None, True, False),  # Target incident angle
    "BM_INP": ParameterMapping(None, True, False),  # Flux expansion
    "BM_INP": ParameterMapping(
        "Q_max_fw", True, False
    ),  # Maximum heat flux to the first wall
    "BM_INP": ParameterMapping(
        "eta_pll", True, False
    ),  # Transient energy fluence to the divertor
}
