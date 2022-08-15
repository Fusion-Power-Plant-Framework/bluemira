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
from typing import Dict, Union

from EUDEMO_builders.pf_coils import make_coilset, make_grid

from bluemira.base.parameter_frame import NewParameter as Parameter
from bluemira.base.parameter_frame import NewParameterFrame as ParameterFrame
from bluemira.base.parameter_frame import parameter_frame
from bluemira.equilibria import Equilibrium
from bluemira.equilibria.profiles import BetaIpProfile
from bluemira.geometry.wire import BluemiraWire

KAPPA_95_TO_100 = 1.12


@parameter_frame
class EquilibriumParams(ParameterFrame):
    A: Parameter[float]
    B_0: Parameter[float]
    beta_p: Parameter[float]
    CS_bmax: Parameter[float]
    CS_jmax: Parameter[float]
    delta_95: Parameter[float]
    g_cs_mod: Parameter[float]
    I_p: Parameter[float]
    kappa_95: Parameter[float]
    n_CS: Parameter[float]
    n_PF: Parameter[float]
    PF_bmax: Parameter[float]
    PF_jmax: Parameter[float]
    R_0: Parameter[float]
    r_cs_in: Parameter[float]
    tk_cs_casing: Parameter[float]
    tk_cs_insulation: Parameter[float]
    tk_cs: Parameter[float]


def make_equilibrium(
    _params: Union[EquilibriumParams, Dict], tf_coil_boundary: BluemiraWire
):
    if isinstance(_params, dict):
        params = EquilibriumParams.from_dict(_params)
    else:
        params = _params

    kappa = KAPPA_95_TO_100 * params.kappa_95.value
    coilset = make_coilset(
        tf_coil_boundary,
        R_0=params.R_0.value,
        kappa=kappa,
        delta=params.delta_95.value,
        r_cs=params.r_cs_in.value + params.tk_cs.value / 2,
        tk_cs=params.tk_cs.value / 2,
        g_cs=params.g_cs_mod.value,
        tk_cs_ins=params.tk_cs_insulation.value,
        tk_cs_cas=params.tk_cs_casing.value,
        n_CS=params.n_CS.value,
        n_PF=params.n_PF.value,
        CS_jmax=params.CS_jmax.value,
        CS_bmax=params.CS_bmax.value,
        PF_jmax=params.PF_jmax.value,
        PF_bmax=params.PF_bmax.value,
    )
    profiles = BetaIpProfile(
        params.beta_p.value,
        params.I_p.value * 1e6,  # TODO(hsaunders1904): unit change?
        params.R_0.value,
        params.B_0.value,
    )
    grid = make_grid(
        params.R_0.value, params.A.value, kappa, scale_x=1.6, scale_z=1.7, nx=65, nz=65
    )

    return Equilibrium(coilset, grid, profiles)
