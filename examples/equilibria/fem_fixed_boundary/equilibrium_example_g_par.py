# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
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

import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

from bluemira.base.file import get_bluemira_path, get_bluemira_root
from bluemira.base.logs import set_log_level
from bluemira.codes import transport_code_solver
from bluemira.equilibria.fem_fixed_boundary.equilibrium import (
    solve_transport_fixed_boundary,
)
from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemGradShafranovFixedBoundary,
)
from bluemira.equilibria.fem_fixed_boundary.file import save_fixed_boundary_to_file
from bluemira.equilibria.shapes import JohnerLCFS, ZakharovLCFS, flux_surface_zakharov

set_log_level("NOTSET")

# %% [markdown]
#
# # Fixed Boundary Equilibrium
# Setup the Plasma shape parameterisation variables

# %%
johner_parameterisation = JohnerLCFS(
    {
        "r_0": {"value": 8.9830},
        "a": {"value": 2.9075846464},
        "kappa_u": {"value": 1.6},
        "kappa_l": {"value": 1.75},
        "delta_u": {"value": 0.33},
        "delta_l": {"value": 0.45},
    }
)

from bluemira.geometry.tools import interpolate_bspline
from bluemira.utilities.opt_variables import BoundedVariable, OptVariables


class ModZakharovLCFS(ZakharovLCFS):
    def __init__(self, var_dict=None):
        variables = OptVariables(
            [
                BoundedVariable(
                    "r_0", 9, lower_bound=0, upper_bound=np.inf, descr="Major radius"
                ),
                BoundedVariable(
                    "z_0",
                    0,
                    lower_bound=-np.inf,
                    upper_bound=np.inf,
                    descr="Vertical coordinate at geometry centroid",
                ),
                BoundedVariable(
                    "a", 3, lower_bound=0, upper_bound=np.inf, descr="Minor radius"
                ),
                BoundedVariable(
                    "kappa", 1.5, lower_bound=1.0, upper_bound=np.inf, descr="Elongation"
                ),
                BoundedVariable(
                    "delta",
                    0.4,
                    lower_bound=0.0,
                    upper_bound=1.0,
                    descr="Triangularity",
                ),
                BoundedVariable(
                    "kappa_l",
                    1.5,
                    lower_bound=1.0,
                    upper_bound=np.inf,
                    descr="Elongation",
                ),
                BoundedVariable(
                    "kappa_u",
                    1.5,
                    lower_bound=1.0,
                    upper_bound=np.inf,
                    descr="Elongation",
                ),
                BoundedVariable(
                    "delta_l",
                    0.33,
                    lower_bound=0.0,
                    upper_bound=1.0,
                    descr="Elongation",
                ),
                BoundedVariable(
                    "delta_u",
                    0.33,
                    lower_bound=0.0,
                    upper_bound=1.0,
                    descr="Elongation",
                ),
            ],
            frozen=True,
        )
        variables.adjust_variables(var_dict, strict_bounds=False)
        variables.adjust_variable("kappa_u", value=variables.kappa)
        variables.adjust_variable("kappa_l", value=variables.kappa)
        variables.adjust_variable("delta_u", value=variables.delta)
        variables.adjust_variable("delta_u", value=variables.delta)
        self.variables = variables

    def adjust_variable(self, name, value=None, lower_bound=None, upper_bound=None):
        if name in ["kappa_u", "kappa_l"]:
            name = "kappa"
        if name in ["delta_u", "delta_l"]:
            name = "delta"
        return super().adjust_variable(name, value, lower_bound, upper_bound)

    def create_shape(self, label="LCFS", n_points=52):
        coordinates = flux_surface_zakharov(*self.variables.values[:5], n=n_points)
        return interpolate_bspline(coordinates.xyz, closed=True, label=label)


johner_parameterisation = ModZakharovLCFS(
    {
        "r_0": {"value": 8.9830e00},
        "a": {"value": 0.28911660139e0001},
        "kappa": {"value": 0.17464842648e0001},
        "delta": {"value": 0.42406696656e0000},
    }
)

from bluemira.equilibria.flux_surfaces import ClosedFluxSurface

V = ClosedFluxSurface(
    johner_parameterisation.create_shape(n_points=1000).discretize(1000)
).volume
V = -2500

# %% [markdown]
# Initialise the transport solver in this case PLASMOD is used

# %%
if plasmod_binary := shutil.which("plasmod"):
    PLASMOD_PATH = os.path.dirname(plasmod_binary)
else:
    PLASMOD_PATH = os.path.join(os.path.dirname(get_bluemira_root()), "plasmod/bin")
binary = os.path.join(PLASMOD_PATH, "plasmod")


source = "Plasmod Example"
plasmod_params = {
    "A": {"value": 3.1, "unit": "", "source": source},
    "R_0": {
        "value": johner_parameterisation.variables.r_0,
        "unit": "m",
        "source": source,
    },
    "I_p": {"value": 19e6, "unit": "A", "source": source},
    "B_0": {"value": 5.31, "unit": "T", "source": source},
    "V_p": {"value": V, "unit": "m^3", "source": source},
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
    "amin": johner_parameterisation.variables.a,
    "pfus_req": 2000.0,
    "pheat_max": 100.0,
    "q_control": 50.0,
    "i_impmodel": "PED_FIXED",
    "i_modeltype": "GYROBOHM_2",
    "i_equiltype": "q95_sawtooth",
    "i_pedestal": "SAARELMA",
    "isawt": "FULLY_RELAXED",
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

# %% [markdown]
# Initialise the FEM problem

# %%
fem_GS_fixed_boundary = FemGradShafranovFixedBoundary(
    p_order=2,
    max_iter=30,
    iter_err_max=1e-4,
    relaxation=0.0,
)

# %% [markdown]
# Solve

# %%
equilibrium = solve_transport_fixed_boundary(
    johner_parameterisation,
    plasmod_solver,
    fem_GS_fixed_boundary,
    kappa95_t=1.652,  # Target kappa_95
    delta95_t=0.333,  # Target delta_95
    lcar_mesh=0.2,
    max_iter=1,
    iter_err_max=1e-1,
    relaxation=0.0,
    plot=False,
    debug=False,
    gif=False,
)

# %% [markdown]
# Calculate g param

# %%
gs_solver = fem_GS_fixed_boundary
transport_solver = plasmod_solver

from bluemira.equilibria.fem_fixed_boundary.equilibrium import calc_metric_coefficients
from bluemira.equilibria.fem_fixed_boundary.utilities import get_flux_surfaces_from_mesh

x1d, flux_surfaces = get_flux_surfaces_from_mesh(
    gs_solver.mesh, gs_solver.psi_norm_2d, transport_solver.get_profile("x")
)

x1Dn, Vn, g1n, g2n, g3n = calc_metric_coefficients(
    flux_surfaces,
    gs_solver.psi,
    gs_solver.psi_norm_2d,
    x1d,
)

x_plasmod = transport_solver.get_profile("x")
v_plasmod = transport_solver.get_profile("V")
g2_plasmod = transport_solver.get_profile("g2")
g3_plasmod = transport_solver.get_profile("g3")

f, ax = plt.subplots(1, 3)

ax[0].plot(x1Dn, Vn, "b-", label="bluemira")
ax[0].plot(x_plasmod, v_plasmod, "b-", label="bluemira")
ax[0].grid()
ax[0].set_xlabel("psi_norm")
ax[0].set_title("$V$")
ax[0].legend()


ax[1].plot(x_plasmod, g2_plasmod, "ro", label="PLASMOD")
ax[1].plot(x1Dn, g2n, "b-", label="bluemira")
ax[1].grid()
ax[1].set_title("g2")
ax[1].set_xlabel("psi_norm")
ax[1].legend()

ax[2].plot(x_plasmod, g3_plasmod, "ro", label="PLASMOD")
ax[2].plot(x1Dn, g3n, "b-", label="bluemira")
ax[2].grid()
ax[2].set_title("g3")
ax[2].set_xlabel("psi_norm")
ax[2].legend()
plt.show()

# second step: calculate H
from bluemira.equilibria.fem_fixed_boundary.equilibrium import calc_curr_dens_profiles

q_plasmod = transport_solver.get_profile("q")
p_plasmod = transport_solver.get_profile("pressure")
psi_plasmod = transport_solver.get_profile("psi")
ffprime_plasmod = transport_solver.get_profile("ffprime")
pprime_plasmod = transport_solver.get_profile("pprime")
B_0 = plasmod_params["B_0"]["value"]
R_0 = plasmod_params["R_0"]["value"]
I_p = plasmod_params["I_p"]["value"]
psi_ax = psi_plasmod[-1]
psi_b = psi_plasmod[0]
Ip, Phi1D, Psi1D, pprime_psi1D_data, F, FFprime = calc_curr_dens_profiles(
    x_plasmod,
    p_plasmod,
    q_plasmod,
    g2_plasmod,
    g3_plasmod,
    v_plasmod,
    0,  # I_p,
    B_0,
    R_0,
    psi_ax,
    psi_b,
)


f, ax = plt.subplots(1, 2)

ax[0].plot(x_plasmod, pprime_plasmod, label="PLASMOD")
ax[0].plot(x_plasmod, pprime_psi1D_data, label="bluemira")
ax[0].set_title("p'")
ax[0].legend()

ax[1].plot(x_plasmod, ffprime_plasmod, label="PLASMOD")
ax[1].plot(x_plasmod, FFprime, label="bluemira")
ax[1].set_title("FF'")
ax[1].legend()
plt.show()
# g2_fun = interp1d(x1D, g2, fill_value="extrapolate")
# grad_g2_x1D = nd.Gradient(g2_fun)

# g3_fun = interp1d(x1D, g3, fill_value="extrapolate")
# grad_g3_x1D = nd.Gradient(g3_fun)

# q_fun = interp1d(x_plasmod, q_plasmod, fill_value="extrapolate")
# p_fun = interp1d(x_plasmod, p_plasmod, fill_value="extrapolate")
# grad_p_x1D = nd.Gradient(p_fun)


# def q2_g32(x):
#     return q_fun(x) ** 2 / g3_fun(x) ** 2


# def func_A(x):
#     return (grad_g2_x1D(x) + 8 * np.pi**4 * nd.Gradient(q2_g32)(x)) / (
#         g2_fun(x) / 2 + 8 * np.pi**4 * q_fun(x) ** 2 / g3_fun(x) ** 2
#     )


# from bluemira.base.constants import MU_0


# def func_P(x):
#     return (
#         4
#         * np.pi**2
#         * MU_0
#         * grad_p_x1D(x)
#         / (g2_fun(x) / 2 + 8 * np.pi**4 * q_fun(x) ** 2 / g3_fun(x) ** 2)
#     )
