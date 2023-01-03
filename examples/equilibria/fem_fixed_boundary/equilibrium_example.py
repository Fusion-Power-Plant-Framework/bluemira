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
An example that shows how to set up the problem for the fixed boundary equilibrium.
"""

# %%
import os
import shutil

from bluemira.base.file import get_bluemira_path, get_bluemira_root
from bluemira.base.logs import set_log_level
from bluemira.codes import transport_code_solver
from bluemira.equilibria.fem_fixed_boundary.equilibrium import (
    solve_transport_fixed_boundary,
)
from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemGradShafranovFixedBoundary,
)
from bluemira.equilibria.shapes import JohnerLCFS

set_log_level("NOTSET")

# %%[markdown]
# Setup the Plasma shape parameterisation variables

# %%
johner_parameterisation = JohnerLCFS(
    {
        "r_0": {"value": 8.9830e00},
        "a": {"value": 3.1},
        "kappa_u": {"value": 1.6},
        "kappa_l": {"value": 1.75},
        "delta_u": {"value": 0.33},
        "delta_l": {"value": 0.45},
    }
)

# %%[markdown]
# Initialise the transport solver in this case PLASMOD is used

# %%
if plasmod_binary := shutil.which("plasmod"):
    PLASMOD_PATH = os.path.dirname(plasmod_binary)
else:
    PLASMOD_PATH = os.path.join(os.path.dirname(get_bluemira_root()), "plasmod/bin")
binary = os.path.join(PLASMOD_PATH, "plasmod")


source = "Plasmod Example"
plasmod_params = {
    "A": {"value": johner_parameterisation.variables.a, "unit": "", "source": source},
    "R_0": {
        "value": johner_parameterisation.variables.r_0,
        "unit": "m",
        "source": source,
    },
    "I_p": {"value": 19e6, "unit": "A", "source": source},
    "B_0": {"value": 5.31, "unit": "T", "source": source},
    "V_p": {"value": -2500, "unit": "m^3", "source": source},
    "v_burn": {"value": -1.0e6, "unit": "V", "source": source},
    "kappa_95": {"value": 1.652, "unit": "", "source": source},
    "delta_95": {"value": 0.333, "unit": "", "source": source},
    "delta": {
        "value": (
            johner_parameterisation.variables.delta_l
            + johner_parameterisation.variables.delta_u
        )
        / 2,
        "unit": "",
        "source": source,
    },
    "kappa": {
        "value": (
            johner_parameterisation.variables.kappa_l
            + johner_parameterisation.variables.kappa_u
        )
        / 2,
        "unit": "",
        "source": source,
    },
    "q_95": {"value": 3.25, "unit": "", "source": source},
    "f_ni": {"value": 0, "unit": "", "source": source},
}

problem_settings = {
    "amin": plasmod_params["R_0"]["value"] / plasmod_params["A"]["value"],
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
    "directory": get_bluemira_path("", subfolder="generated_data"),
}

plasmod_solver = transport_code_solver(
    params=plasmod_params,
    build_config=plasmod_build_config,
    module="PLASMOD",
)

# %%[markdown]
# Initialise the FEM problem

# %%

fem_GS_fixed_boundary = FemGradShafranovFixedBoundary(
    p_order=2,
    max_iter=30,
    iter_err_max=1e-4,
    relaxation=0.05,
)

# %%[markdown]
# Solve

# %%

solve_transport_fixed_boundary(
    johner_parameterisation,
    plasmod_solver,
    fem_GS_fixed_boundary,
    kappa95_t=1.652,  # Target kappa_95
    delta95_t=0.333,  # Target delta_95
    lcar_mesh=0.3,
    max_iter=15,
    iter_err_max=1e-4,
    relaxation=0.0,
    plot=False,
)
