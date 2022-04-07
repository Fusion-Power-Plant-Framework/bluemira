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
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf

import dolfin
import matplotlib.pyplot as plt

from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemGradShafranovFixedBoundary,
)
import time

from bluemira.base.logs import set_log_level
from bluemira.equilibria.fem_fixed_boundary.transport_solver import (
    PlasmodTransportSolver,
)
from bluemira.equilibria.fem_fixed_boundary.utilities import (
    plot_scalar_field,
    plot_profile,
)

# ------------------------------------------------------------------------------
set_log_level("DEBUG")
# ------------------------------------------------------------------------------
params = {
    "R_0": 8.983,
    "A": 3.1,
    "kappa_u": 1.6,
    "kappa_l": 1.8,
    "delta_u": 0.5,
    "delta_l": 0.5,
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

builder_plasma = MakeParameterisedPlasma(params, build_config)
plasma = builder_plasma.build_xz().get_component("xz").get_component("LCFS")

print(builder_plasma._shape.variables)
# ------------------------------------------------------------------------------

PLASMOD_PATH = "/home/ivan/Desktop/bluemira_project/plasmod-master/bin/"
binary = f"{PLASMOD_PATH}plasmod"


new_params = {
    "A": 3.1,
    "R_0": 9.002,
    "I_p": 17.75,
    "B_0": 5.855,
    "V_p": -2500,
    "v_burn": -1.0e6,
    "kappa_95": 1.652,
    "delta_95": 0.333,
    "delta": 0.38491934960310104,
    "kappa": 1.6969830041844367,
}

params = Configuration(new_params)

# Add parameter source
for param_name in params.keys():
    if param_name in new_params:
        param = params.get_param(param_name)
        param.source = "Plasmod Example"

problem_settings = {
    "pfus_req": 0.0,
    "pheat_max": 0.0,
    "q_control": 0.0,
    "i_modeltype": "GYROBOHM_2",
    "i_equiltype": "q95_sawtooth",
    "i_pedestal": "SAARELMA",
}

build_config = {
    "problem_settings": problem_settings,
    "mode": "run",
    "binary": binary,
}

plasmod_solver = PlasmodTransportSolver(
    params=params,
    build_config=build_config,
)

plot_profile(plasmod_solver.x, plasmod_solver.pprime(plasmod_solver.x), "pprime", "-")
plot_profile(plasmod_solver.x, plasmod_solver.ffprime(plasmod_solver.x), "ffrime", "-")


# ------------------------------------------------------------------------------
plasma.shape.boundary[0].mesh_options = {"lcar": 0.5, "physical_group": "lcfs"}
plasma.shape.mesh_options = {"lcar": 0.5, "physical_group": "plasma_face"}

m = meshing.Mesh()
buffer = m(plasma)

msh_to_xdmf("Mesh.msh", dimensions=(0, 2), directory=".", verbose=True)

mesh, boundaries, subdomains, labels = import_mesh(
    "Mesh",
    directory=".",
    subdomains=True,
)
dolfin.plot(mesh)
plt.show()

# ------------------------------------------------------------------------------
# initialize the Grad-Shafranov solver
p = 5
gs_solver = FemGradShafranovFixedBoundary(mesh, p_order=p)

print("\nSolving...")

# solve the Grad-Shafranov equation
solve_start = time.time()  # compute the time it takes to solve
psi = gs_solver.solve(
    plasmod_solver.pprime,
    plasmod_solver.ffprime,
    plasmod_solver.I_p,
    tol=1e-3,
    max_iter=50,
)
solve_end = time.time()

points = mesh.coordinates()
psi_data = np.array([gs_solver.psi(x) for x in points])

levels = np.linspace(0.0, gs_solver._psi_ax, 25)

axis, cntr, _ = plot_scalar_field(
    points[:, 0], points[:, 1], psi_data, levels=levels, axis=None, tofill=True
)
plt.show()

axis, cntr, _ = plot_scalar_field(
    points[:, 0],
    points[:, 1],
    psi_data,
    levels=[gs_solver._psi_ax * 0.05],
    axis=None,
    tofill=False,
)
plt.show()
