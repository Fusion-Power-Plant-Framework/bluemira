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
    calculate_plasma_shape_params,
)

from bluemira.geometry.tools import make_bspline
from bluemira.geometry.coordinates import Coordinates

# ------------------------------------------------------------------------------
set_log_level("INFO")
# ------------------------------------------------------------------------------

main_params = {
    "R_0": 8.983,
    "A": 3.1,
    "kappa_u": 1.65,
    "kappa_l": 1.85,
    "delta_u": 0.6,
    "delta_l": 0.55,
    "I_p": 17e6,
    "B_0": 4.96,
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

PLASMOD_PATH = "/home/ivan/Desktop/bluemira_project/plasmod/bin/"
binary = f"{PLASMOD_PATH}plasmod"

theta = 0.5
iter_err_max = 1e-5
niter = 1
niter_max = 30
delta95_t = 0.333
kappa95_t = 1.652

new_params = {
    "A": main_params["A"],
    "R_0": main_params["R_0"],
    "I_p": main_params["I_p"] / 1e6,
    "B_0": main_params["B_0"],
    "V_p": -2500,
    "v_burn": -1.0e6,
    "kappa_95": 1.652,
    "delta_95": 0.333,
    "delta": main_params["delta_l"],
    "kappa": main_params["kappa_l"],
}

verbose_plot = False

while niter <= niter_max:

    builder_plasma = MakeParameterisedPlasma(main_params, build_config)
    plasma = builder_plasma.build_xz().get_component("xz").get_component("LCFS")

    plasma_volume = (
        builder_plasma.build_xyz()
        .get_component("xyz")
        .get_component("LCFS")
        .shape.volume
    )

    print(f"plasma shape: {plasma._shape}")

    # print(builder_plasma._shape.variables)
    # ------------------------------------------------------------------------------
    new_params['V_p'] = plasma_volume
    plasmod_params = Configuration(new_params)

    # Add parameter source
    for param_name in plasmod_params.keys():
        if param_name in new_params:
            param = plasmod_params.get_param(param_name)
            param.source = "Plasmod Example"

    problem_settings = {
        "amin": 2.903871,
        "pfus_req": 0.0,
        "pheat_max": 0.0,
        "q_control": 0.0,
        "i_modeltype": "GYROBOHM_2",
        "i_equiltype": "q95_sawtooth",
        "i_pedestal": "SAARELMA",
    }

    plasmod_build_config = {
        "problem_settings": problem_settings,
        "mode": "run",
        "binary": binary,
    }

    plasmod_solver = PlasmodTransportSolver(
        params=plasmod_params,
        build_config=plasmod_build_config,
    )

    plot_profile(
        plasmod_solver.x, plasmod_solver.pprime(plasmod_solver.x), "pprime", "-"
    )
    plot_profile(
        plasmod_solver.x, plasmod_solver.ffprime(plasmod_solver.x), "ffrime", "-"
    )

    # ------------------------------------------------------------------------------
    plasma.shape.boundary[0].mesh_options = {"lcar": 0.25, "physical_group": "lcfs"}
    plasma.shape.mesh_options = {"lcar": 0.25, "physical_group": "plasma_face"}

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
    p = 3
    gs_solver = FemGradShafranovFixedBoundary(mesh, p_order=p)

    print(f"\nSolving iteration {niter}...")

    # solve the Grad-Shafranov equation
    solve_start = time.time()  # compute the time it takes to solve
    psi = gs_solver.solve(
        plasmod_solver.pprime,
        plasmod_solver.ffprime,
        plasmod_solver.I_p,
        tol=1e-4,
        max_iter=50,
        verbose_plot=verbose_plot
    )
    solve_end = time.time()

    plasma.shape.boundary[0].mesh_options = {"lcar": 0.15, "physical_group": "lcfs"}
    plasma.shape.mesh_options = {"lcar": 0.15, "physical_group": "plasma_face"}

    m = meshing.Mesh()
    buffer = m(plasma)

    msh_to_xdmf("Mesh.msh", dimensions=(0, 2), directory=".", verbose=True)

    mesh, boundaries, subdomains, labels = import_mesh(
        "Mesh",
        directory=".",
        subdomains=True,
    )

    new_mesh = dolfin.refine(mesh)

    if verbose_plot:
        dolfin.plot(mesh)
        plt.show()

        dolfin.plot(new_mesh)
        plt.title("Refined")
        plt.show()

    points = new_mesh.coordinates()
    psi_data = np.array([gs_solver.psi(x) for x in points])

    if verbose_plot:
        levels = np.linspace(0.0, gs_solver.psi_ax, 25)

        axis, cntr, _ = plot_scalar_field(
            points[:, 0], points[:, 1], psi_data, levels=levels, axis=None, tofill=True
        )
        plt.show()

    ###################################################
    R_geo, kappa_95, delta_95 = calculate_plasma_shape_params(
        points, psi_data, [gs_solver.psi_ax * 0.05]
    )

    R_geo, kappa_95, delta_95 = R_geo[0], kappa_95[0], delta_95[0]
    new_params['kappa_95'] = kappa_95
    new_params['delta_95'] = delta_95

    err_delta = abs(delta_95 - delta95_t) / delta95_t
    err_kappa = abs(kappa_95 - kappa95_t) / kappa95_t
    iter_err = max(err_delta, err_kappa)

    print("previous shape parameters")
    print(f"kappa_u: {main_params['kappa_u']}, delta_u: {main_params['delta_u']}")

    main_params["kappa_u"] = (
        theta * main_params["kappa_u"] * (kappa95_t / kappa_95)
        + (1 - theta) * main_params["kappa_u"]
    )
    main_params["delta_u"] = (
        theta * main_params["delta_u"] * (delta95_t / delta_95)
        + (1 - theta) * main_params["delta_u"]
    )

    print("recalculated shape parameters")


    print(f"kappa_u: {main_params['kappa_u']}, delta_u: {main_params['delta_u']}")

    print(" ")
    print(f"MIRA delta95 = {delta_95}")
    print(f"target delta95 = {delta95_t}")

    print(f"|Target - MIRA|/Target = {((delta_95 - delta95_t) / delta95_t)}")

    print(" ")
    print(f"MIRA kappa95 = {kappa_95}")
    print(f"target kappa95 = {kappa95_t}")

    print(f"|Target - MIRA|/Target = {((kappa_95 - kappa95_t) / kappa95_t)}")

    print(f"iter_err: {iter_err}, iter_err_max: {iter_err_max}")

    if iter_err <= iter_err_max:
       break

    niter += 1
