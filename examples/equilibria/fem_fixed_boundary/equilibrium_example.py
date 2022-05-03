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
This example try to reproduce the fixed boundary equilibrium problem as solved
in mira implemented in matlab (i.e. with Plasmod coupling + Grad-Shafranov)
"""

import numpy as np

from bluemira.builders.plasma import MakeParameterisedPlasma
from bluemira.base.config import Configuration
from bluemira.base.logs import set_log_level
from bluemira.equilibria.fem_fixed_boundary.equilibrium import (
    solve_plasmod_fixed_boundary,
)

# ------------------------------------------------------------------------------
set_log_level("INFO")
# ------------------------------------------------------------------------------

# R0 = 8.9830e00
# A = 3.1000e00
# V_in = 2.4170e03
# Bt = 5.3100e00
# deltaX = 4.9141e-01
# delta95 = 3.3300e-01
# kappaX = 1.7962e00
# kappa95 = 1.6520e00
# q95 = 3.2500e00
# Ip_in = 1.9000e01
tol = 1.0000e-06
dtmin = 3.0000e-02
dtmax = 1.0000e-01
dgy = 1.0000e-05
nx = 4.1000e01
nxt = 1.1000e01
# isiccir = 0.0000e00
test = 1.0000e04
fuelmix = 5.0000e-01
fP2E = 5.0000e00
fP2EAr = 5.0000e00
cW = 5.0000e-05
Tesep = 1.0000e-01
Teped = 5.2397e00
rho_T = 9.4000e-01
rho_n = 9.4000e-01
Psep_PLH_min = 1.1000e00
Psep_PLH_max = 1.2000e00
PsepBt_qAR_max = 9.2000e00
Psep_R0_max = 2.3000e01
qdivt_max = 1.0000e01
f_gw = 9.0000e-01
f_gws = 5.0000e-01
H = 9.0676e-01
nbcdeff = 3.0000e-01
fpion = 5.0000e-01
nbi_energy = 1.0000e03
x_control = 0.0000e00
dx_control = 2.0000e-01
x_cd = 0.0000e00
dx_cd = 2.0000e-01
x_fus = 0.0000e00
dx_fus = 2.0000e-01
x_heat = 0.0000e00
dx_heat = 2.0000e-01
# Pfus_req = 2.0000e03
# q_control = 5.0000e01
# f_ni = 0.0000e00
# Pheat_max = 1.0000e02
# i_impmodel = 1.0000e00
# i_modeltype = 1.1100e02
# i_equiltype = 1.0000e00
# i_pedestal = 2.0000e00


main_params = {
    "R_0": 8.9830e00,
    "A": 3.1,
    "kappa_u": 1.65,
    "kappa_l": 1.85,
    "delta_u": 0.6,
    "delta_l": 0.55,
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

Configuration.set_template_parameters([["kappa_u", "kappa_u", "", "dimensionless"]])
Configuration.set_template_parameters([["kappa_l", "kappa_l", "", "dimensionless"]])
Configuration.set_template_parameters([["delta_u", "delta_u", "", "dimensionless"]])
Configuration.set_template_parameters([["delta_l", "delta_l", "", "dimensionless"]])
Configuration.set_template_parameters([["kappa_95", "kappa_95", "", "dimensionless"]])
Configuration.set_template_parameters([["delta_95", "delta_95", "", "dimensionless"]])

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
    "delta": (main_params["delta_l"] + main_params["delta_u"])/2,
    "kappa": (main_params["kappa_l"] + main_params["kappa_u"])/2,
    "q_95": 3.25,
    "f_ni": 0,
}

# plasmod options
PLASMOD_PATH = "/home/ivan/Desktop/bluemira_project/plasmod/bin/"
binary = f"{PLASMOD_PATH}plasmod"

plasmod_params = Configuration(new_params)

# Add parameter source
for param_name in plasmod_params.keys():
    if param_name in new_params:
        param = plasmod_params.get_param(param_name)
        param.source = "Plasmod Example"

problem_settings = {
    "amin": new_params['R_0']/new_params['A'],
    "pfus_req": 0, #2000.0,
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
gs_options = {"p_order": 2, "tol": 1e-4, "max_iter": 30, "verbose_plot": True}

# target values
delta95_t = 0.333
kappa95_t = 1.652

solve_plasmod_fixed_boundary(
    builder_plasma,
    plasmod_options,
    gs_options,
    delta95_t,
    kappa95_t,
    lcar_coarse=0.1,
    lcar_fine=0.01,
    niter_max=5,
    iter_err_max=1e-5,
    theta=0.8,
    gs_i_theta=1,
)
