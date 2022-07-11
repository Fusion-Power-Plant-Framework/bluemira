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

from bluemira.base.config import Configuration
from bluemira.base.file import get_bluemira_root
from bluemira.base.logs import set_log_level
from bluemira.builders.plasma import MakeParameterisedPlasma
from bluemira.equilibria.fem_fixed_boundary.equilibrium import (
    solve_plasmod_fixed_boundary,
)

# set_log_level("DEBUG")

# %%
main_params = {
    "R_0": 8.9830e00,
    "A": 3.1,
    "kappa_u": 1.6,
    "kappa_l": 1.75,
    "delta_u": 0.33,
    "delta_l": 0.45,
    "I_p": 19e6,
    "B_0": 5.31,
}

build_config = {
    "name": "Plasma",
    "class": "MakeParameterisedPlasma",
    "param_class": "bluemira.equilibria.shapes::JohnerLCFS",
    "variables_map": {
        "r_0": "R_0",
        "a": "A",
        "kappa_u": "kappa_u",
        "kappa_l": "kappa_l",
        "delta_u": "delta_u",
        "delta_l": "delta_l",
    },
}

Configuration.set_template_parameters(
    [
        ["kappa_u", "kappa_u", "", "dimensionless"],
        ["kappa_l", "kappa_l", "", "dimensionless"],
        ["delta_u", "delta_u", "", "dimensionless"],
        ["delta_l", "delta_l", "", "dimensionless"],
        ["kappa_95", "kappa_95", "", "dimensionless"],
        ["delta_95", "delta_95", "", "dimensionless"],
    ]
)

builder_plasma = MakeParameterisedPlasma(main_params, build_config)

new_params = {
    "A": main_params["A"],
    "R_0": main_params["R_0"],
    "I_p": main_params["I_p"] / 1e6,
    "B_0": main_params["B_0"],
    "V_p": -2500,
    "v_burn": -1.0e6,
    "kappa_95": 1.652,
    "delta_95": 0.333,
    "delta": (main_params["delta_l"] + main_params["delta_u"]) / 2,
    "kappa": (main_params["kappa_l"] + main_params["kappa_u"]) / 2,
    "q_95": 3.25,
    "f_ni": 0,
}

# plasmod options
PLASMOD_PATH = "/home/ivan/Desktop/bluemira_project/plasmod/bin/"
PLASMOD_PATH = PLASMOD_PATH = os.path.join(
    os.path.split(get_bluemira_root())[:-1][0], "plasmod/bin"
)
binary = os.path.join(PLASMOD_PATH, "plasmod")

plasmod_params = Configuration(new_params)

# Add parameter source
for param_name in plasmod_params.keys():
    if param_name in new_params:
        param = plasmod_params.get_param(param_name)
        param.source = "Plasmod Example"

problem_settings = {
    "amin": new_params["R_0"] / new_params["A"],
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

plasmod_options = {"params": plasmod_params, "build_config": plasmod_build_config}
gs_options = {"p_order": 2, "tol": 1e-5, "max_iter": 30, "verbose_plot": False}

# target values
delta95_t = 0.333
kappa95_t = 1.652

solve_plasmod_fixed_boundary(
    builder_plasma,
    plasmod_options,
    gs_options,
    delta95_t,
    kappa95_t,
    lcar_coarse=0.3,
    lcar_fine=0.05,
    niter_max=15,
    iter_err_max=7e-3,
    relaxation=0.0,
    gs_relaxation=0.05,
    plot=False,
    verbose=False,
)
