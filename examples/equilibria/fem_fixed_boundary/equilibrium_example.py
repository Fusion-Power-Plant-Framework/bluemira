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
An example that shows how to set up the problem for the fixed boundary equilibrium.
"""

# %%
import os
from dataclasses import dataclass

from bluemira.base.file import get_bluemira_root
from bluemira.base.logs import set_log_level
from bluemira.base.parameter_frame import NewParameter as Parameter
from bluemira.base.parameter_frame import NewParameterFrame as ParameterFrame
from bluemira.equilibria.fem_fixed_boundary.equilibrium import (
    FemGradShafranovOptions,
    PlasmaFixedBoundary,
    solve_transport_fixed_boundary,
)
from bluemira.equilibria.shapes import JohnerLCFS

set_log_level("NOTSET")

# %%
main_params = PlasmaFixedBoundary(
    **{
        "r_0": 8.9830e00,
        "a": 3.1,
        "kappa_u": 1.6,
        "kappa_l": 1.75,
        "delta_u": 0.33,
        "delta_l": 0.45,
    }
)


@dataclass
class TransportSolverParams(ParameterFrame):
    A: Parameter[float]
    R_0: Parameter[float]
    I_p: Parameter[float]
    B_0: Parameter[float]
    V_p: Parameter[float]
    v_burn: Parameter[float]
    kappa_95: Parameter[float]
    delta_95: Parameter[float]
    delta: Parameter[float]
    kappa: Parameter[float]
    q_95: Parameter[float]
    f_ni: Parameter[float]


source = "Plasmod Example"
plasmod_params = TransportSolverParams.from_dict(
    {
        "A": {"value": main_params.a, "unit": "", "source": source},
        "R_0": {"value": main_params.r_0, "unit": "m", "source": source},
        "I_p": {"value": 19e6, "unit": "A", "source": source},
        "B_0": {"value": 5.31, "unit": "T", "source": source},
        "V_p": {"value": -2500, "unit": "m^3", "source": source},
        "v_burn": {"value": -1.0e6, "unit": "V", "source": source},
        "kappa_95": {"value": 1.652, "unit": "", "source": source},
        "delta_95": {"value": 0.333, "unit": "", "source": source},
        "delta": {
            "value": (main_params.delta_l + main_params.delta_u) / 2,
            "unit": "",
            "source": source,
        },
        "kappa": {
            "value": (main_params.kappa_l + main_params.kappa_u) / 2,
            "unit": "",
            "source": source,
        },
        "q_95": {"value": 3.25, "unit": "", "source": source},
        "f_ni": {"value": 0, "unit": "", "source": source},
    }
)

# PLASMOD options
PLASMOD_PATH = os.path.join(os.path.split(get_bluemira_root())[:-1][0], "plasmod/bin")
binary = os.path.join(PLASMOD_PATH, "plasmod")

problem_settings = {
    "amin": plasmod_params.R_0.value / plasmod_params.A.value,
    "pfus_req": 2000.0,
    "pheat_max": 100.0,
    "q_control": 50.0,
    "i_impmodel": "PED_FIXED",
    "i_modeltype": "GYROBOHM_2",
    "i_equiltype": "q95_sawtooth",
    "i_pedestal": "SAARELMA",
}

plasmod_build_config = {
    "problem_settings": problem_settings,
    "mode": "run",
    "binary": binary,
}

gs_options = FemGradShafranovOptions(
    **{
        "p_order": 2,
        "max_iter": 30,
        "iter_err_max": 1e-4,
        "relaxation": 0.05,
    }
)

# target values
delta95_t = 0.333
kappa95_t = 1.652

solve_transport_fixed_boundary(
    JohnerLCFS,
    main_params,
    plasmod_params,
    plasmod_build_config,
    gs_options,
    delta95_t,
    kappa95_t,
    lcar_mesh=0.3,
    max_iter=15,
    iter_err_max=1e-4,
    relaxation=0.0,
    plot=False,
)
