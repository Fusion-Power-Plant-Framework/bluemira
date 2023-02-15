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
from bluemira.equilibria.shapes import JohnerLCFS

set_log_level("NOTSET")

# %% [markdown]
#
# # Fixed Boundary Equilibrium
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
    relaxation=0.05,
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
    plot=True,
    debug=False,
    gif=False,
)

# %% [markdown]
# Save to a file

# %%
data = save_fixed_boundary_to_file(
    os.sep.join(
        [get_bluemira_path("", subfolder="generated_data"), "fixed_boundary_data.json"]
    ),
    "something",
    equilibrium,
    100,
    110,
    formatt="json",
)

# %% [markdown]
# Inspect the final converged equilibrum

# %%
xx, zz = np.meshgrid(data.x, data.z, indexing="ij")
f, ax = plt.subplots()
ax.contour(xx, zz, data.psi)
ax.plot(data.xbdry, data.zbdry)
ax.set_aspect("equal")

f, ax = plt.subplots(2, 2)
ax[0, 0].plot(data.psinorm, data.pprime, label="p'")
ax[0, 1].plot(data.psinorm, data.ffprime, label="FF'")
ax[1, 1].plot(data.psinorm, data.fpol, label="F")
ax[1, 0].plot(data.psinorm, data.pressure, label="p")
for axi in ax.flat:
    axi.legend()

plt.show()

# %% [markdown]
# Calculate g param

# %%
gs_solver = fem_GS_fixed_boundary
transport_solver = plasmod_solver

from bluemira.geometry.coordinates import Coordinates

x1D = np.linspace(0.1, 1, 100)
#x1D = transport_solver.get_profile("x")

x2D = gs_solver.psi_norm_2d
mesh = gs_solver.mesh
nx = x1D.size

from bluemira.equilibria.fem_fixed_boundary import utilities

mesh_points = gs_solver.mesh.coordinates()
x = mesh_points[:, 0]
z = mesh_points[:, 1]

x2D_data = np.array([x2D(p) for p in mesh_points])

ax, cntr, cntrf = utilities.plot_scalar_field(x, z, x2D_data, levels=x1D)
plt.title("psi_norm")
plt.show()

index = []
FS = []
for i in range(nx):
    if x1D[i] == 1:
        from bluemira.equilibria.fem_fixed_boundary import file
        path = np.transpose(file._get_mesh_boundary(mesh))
        FS.append(path)
    else:
        path = cntr.collections[i].get_paths()
        if len(path):
            FS.append(path[0].vertices)
        else:
            print(f"Cannot calculate volume for psi_norm = {x1D[i]}")
            index.append(i)

n = len(index)
for i in range(n):
    x1D = np.delete(x1D, index[i])

nx = x1D.size

g1 = np.zeros((nx, 1))
g2 = np.zeros((nx, 1))
g3 = np.zeros((nx, 1))
V = np.zeros((nx, 1))

# calculate volume
from bluemira.geometry.tools import make_polygon, BluemiraFace
from bluemira.base.components import Component, PhysicalComponent

FS_pol = []
root = Component("root")
for i in range(nx):
    points = Coordinates({"x": FS[i][:, 0], "z": FS[i][:, 1]})
    pgonPol = make_polygon(points, closed=True)
    PhysicalComponent(f"FS{i}", pgonPol, parent=root)
    pgonFac = BluemiraFace(pgonPol)
    V[i] = 2 * np.pi * pgonFac.center_of_mass[0] * pgonFac.area
    FS_pol.append(pgonPol)

root.plot_2d(show=False)
plt.title("Flux surfaces for g2,g3 calculation")
plt.show()

V = V.reshape(-1)

from scipy.interpolate import interp1d

V_fun = interp1d(x1D, V, fill_value="extrapolate")

# dVdx_data = np.gradient(V,x1D)
# dVdx = interp1d(x1D, dVdx_data, fill_value="extrapolate")

import numdifftools as nd

gradV_x1D = nd.Gradient(V_fun)
gradV_x1D_data = np.array([gradV_x1D(x) for x in x1D])

plt.plot(x1D, V, label='V')
# plt.plot(x1D, dVdx_data, 'ro')
plt.plot(x1D, gradV_x1D_data, "g-", label="gradV")
plt.xlabel("psi_norm")
plt.legend()
plt.show()

grad_x2D = nd.Gradient(x2D)


def grad_x2D_norm(x):
    return np.sqrt(np.sum(np.abs(grad_x2D(x)) ** 2))


grad_x2D_data = np.array([grad_x2D(p) for p in mesh_points])
grad_x2D_norm_data = np.sqrt(np.sum(np.abs(grad_x2D_data) ** 2, axis=-1))

ax, cntr, cntrf = utilities.plot_scalar_field(x, z, grad_x2D_norm_data)
plt.title("|grad psi_norm|")
plt.show()

psi_data = np.array([gs_solver.psi(p) for p in mesh_points])
grad_psi = nd.Gradient(gs_solver.psi)
grad_psi_data = np.array([grad_psi(p) for p in mesh_points])


def grad_psi_norm(x):
    return np.sqrt(np.sum(np.abs(grad_psi(x)) ** 2))


grad_psi_norm_data = np.array([grad_psi_norm(p) for p in mesh_points])

ax, cntr, cntrf = utilities.plot_scalar_field(x, z, psi_data)
plt.title("psi")
plt.show()

ax, cntr, cntrf = utilities.plot_scalar_field(x, z, grad_psi_norm_data)
plt.title("|grad psi|")
plt.show()

r2D = mesh_points[:, 0]

def gradV_norm(x):
    """gradV norm"""
    return grad_x2D_norm(x) * gradV_x1D(x2D(x))

gradV_norm_data = np.array([gradV_norm(p) for p in mesh_points])

ax, cntr, cntrf = utilities.plot_scalar_field(x, z, gradV_norm_data)
plt.title("|grad V|")
plt.show()


def Bp(x):
    """Bp"""
    return np.divide(grad_psi_norm(x), x[0]) / (2 * np.pi * x[0])


Bp_data = np.array([Bp(p) for p in mesh_points])

dlp = []
lp = []
for i in range(nx):
    dlp.append(
        np.concatenate(
            (np.array([0]), np.array([edge.length for edge in FS_pol[i].edges]))
        )
    )
    lp.append(np.cumsum(dlp[i]))

for i in range(nx):
    print(f"integrating over FS[{i}]")
    points = FS_pol[i].vertexes[[0, 2], :]
    points = np.transpose(points)
    points = np.concatenate((points, [points[0]]))

    y0_data = np.array([1 / Bp(p) for p in points])
    y1_data = np.array([gradV_norm(p) ** 2 / Bp(p) for p in points])
    y2_data = np.array([gradV_norm(p) ** 2 / Bp(p) / p[0] ** 2 for p in points])
    y3_data = np.array([1 / Bp(p) / p[0] ** 2 for p in points])
    x_data = lp[i]
    g1[i] = np.trapz(y1_data, x_data) / np.trapz(y0_data, x_data)
    g2[i] = np.trapz(y2_data, x_data) / np.trapz(y0_data, x_data)
    g3[i] = np.trapz(y3_data, x_data) / np.trapz(y0_data, x_data)

g1 = g1.flatten()
g2 = g2.flatten()
g3 = g3.flatten()

x1D = np.insert(x1D, 0, 0)
g1 = np.insert(g1, 0, 0)
g2 = np.insert(g2, 0, 0)
g3 = np.insert(g3, 0, 0)

g2_temp = interp1d(x1D[1:-1], g2[1:-1], fill_value="extrapolate")
g2[-1] = g2_temp(x1D[-1])

g3_temp = interp1d(x1D[1:-1], g3[1:-1], fill_value="extrapolate")
g3[0] = g3_temp(x1D[0])
g3[-1] = g3_temp(x1D[-1])

x_plasmod = transport_solver.get_profile("x")
g2_plasmod = transport_solver.get_profile("g2")
g3_plasmod = transport_solver.get_profile("g3")

plt.plot(x_plasmod, g2_plasmod, "ro", label="PLASMOD")
plt.plot(x1D, g2, "b-", label="bluemira")
plt.grid()
plt.title("g2")
plt.xlabel("psi_norm")
plt.legend()
plt.show()

plt.plot(x_plasmod, g3_plasmod, "ro", label="PLASMOD")
plt.plot(x1D, g3, "b-", label="bluemira")
plt.grid()
plt.title("g3")
plt.xlabel("psi_norm")
plt.legend()
plt.show()

# second step: calculate H
x_plasmod = transport_solver.get_profile("x")
q_plasmod = transport_solver.get_profile("q")
p_plasmod = transport_solver.get_profile("pressure")
B_0 = plasmod_params["B_0"]["value"]
R_0 = plasmod_params["R_0"]["value"]

g2_fun = interp1d(x1D, g2, fill_value="extrapolate")
grad_g2_x1D = nd.Gradient(g2_fun)

g3_fun = interp1d(x1D, g3, fill_value="extrapolate")
grad_g3_x1D = nd.Gradient(g3_fun)

q_fun = interp1d(x_plasmod, q_plasmod, fill_value="extrapolate")
p_fun = interp1d(x_plasmod, p_plasmod, fill_value="extrapolate")
grad_p_x1D = nd.Gradient(p_fun)

def q2_g32(x):
    return q_fun(x) ** 2 / g3_fun(x) ** 2

def func_A(x):
    return (grad_g2_x1D(x) + 8 * np.pi ** 4 * nd.Gradient(q2_g32)(x)) / (
                g2_fun(x) / 2 + 8 * np.pi ** 4 * q_fun(x) ** 2 / g3_fun(x) ** 2)

from bluemira.base.constants import MU_0
def func_P(x):
    return 4 * np.pi ** 2 * MU_0 * grad_p_x1D(x) / (g2_fun(x) / 2 + 8 * np.pi ** 4 * q_fun(x) ** 2 / g3_fun(x) ** 2)


from scipy.integrate import quad
def c1(x):
    return quad(func_A, x, 1)

