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
Example on the application of the GS solver for a Johner plasma parametrization
"""

import dolfin
import matplotlib.pyplot as plt
import numpy as np

# %%
from bluemira.base.design import Design
from bluemira.builders.plasma import MakeParameterisedPlasma
from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemGradShafranovFixedBoundary,
    FemMagnetostatic2d,
)
from bluemira.equilibria.fem_fixed_boundary.utilities import (
    ScalarSubFunc,
    plot_scalar_field,
)
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf

# %%[markdown]
# # Create a plasma shape

# %%

params = {
    "R_0": (9.0, "Input"),
    "A": (3.5, "Input"),
}
build_config = {
    "name": "Plasma",
    "class": "MakeParameterisedPlasma",
    "param_class": "bluemira.equilibria.shapes::JohnerLCFS",
    "variables_map": {
        "r_0": "R_0",
        "a": "A",
    },
}
builder = MakeParameterisedPlasma(params, build_config)
plasma = builder().get_component("xz").get_component("LCFS")

plasma.shape.mesh_options = {"lcar": 0.3, "physical_group": "plasma"}
plasma.shape.boundary[0].mesh_options = {"lcar": 0.3, "physical_group": "lcfs"}

# %%[markdown]

# Initialize and create the mesh

# %%

m = meshing.Mesh()
buffer = m(plasma)

# %%[markdown]

# # Convert to xdmf

# %%

msh_to_xdmf("Mesh.msh", dimensions=(0, 2), directory=".", verbose=True)

mesh, boundaries, subdomains, labels = import_mesh(
    "Mesh",
    directory=".",
    subdomains=True,
)
dolfin.plot(mesh)
plt.show()

Ic = 18e6

gs_solver = FemGradShafranovFixedBoundary(mesh)
gs_solver.solve(1, 0, Ic, max_iter=20)

points = mesh.coordinates()
psi_data = np.array([gs_solver.psi(x) for x in points])
# solovev.psi.set_allow_extrapolation(True)

levels = np.linspace(0.0, gs_solver._psi_ax, 25)

axis, cntr, _ = plot_scalar_field(
    points[:, 0], points[:, 1], psi_data, levels=levels, axis=None, tofill=True
)
plt.show()
