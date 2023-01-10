# %% nbsphinx="hidden"
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
Example on the application of the GS solver for a Johner plasma parametrization
"""

# %%
import os

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.components import PhysicalComponent
from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemGradShafranovFixedBoundary,
)
from bluemira.equilibria.fem_fixed_boundary.utilities import plot_scalar_field
from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.geometry.face import BluemiraFace
from bluemira.mesh import meshing
from bluemira.mesh.tools import import_mesh, msh_to_xdmf

import dolfin  # isort:skip

# %% [markdown]
# # Create a plasma shape


# %%
var_dict = {"r_0": {"value": 9.0}, "a": {"value": 3.5}}
plasma = PhysicalComponent(
    "Plasma", shape=BluemiraFace(JohnerLCFS(var_dict).create_shape())
)

plasma.shape.mesh_options = {"lcar": 0.3, "physical_group": "plasma"}
plasma.shape.boundary[0].mesh_options = {"lcar": 0.3, "physical_group": "lcfs"}

# %% [markdown]
#
# Initialize and create the mesh

# %%
directory = get_bluemira_path("", subfolder="generated_data")
meshfiles = [os.path.join(directory, p) for p in ["Mesh.geo_unrolled", "Mesh.msh"]]
meshing.Mesh(meshfile=meshfiles)(plasma)

# %% [markdown]
#
# # Convert to xdmf

# %%
msh_to_xdmf("Mesh.msh", dimensions=(0, 2), directory=directory)

mesh, boundaries, subdomains, labels = import_mesh(
    "Mesh",
    directory=directory,
    subdomains=True,
)
dolfin.plot(mesh)
plt.show()

Ic = 18e6

gs_solver = FemGradShafranovFixedBoundary(max_iter=20)
gs_solver.set_mesh(mesh)
gs_solver.set_profiles(np.ones(2), np.zeros(2), Ic)
gs_solver.solve(plot=True)

points = mesh.coordinates()
psi_data = np.array([gs_solver.psi(x) for x in points])

levels = np.linspace(0.0, gs_solver.psi_ax, 25)

axis, _, _ = plot_scalar_field(
    points[:, 0], points[:, 1], psi_data, levels=levels, axis=None, tofill=True
)
plt.show()

axis, _, _ = plot_scalar_field(
    points[:, 0],
    points[:, 1],
    psi_data,
    levels=[gs_solver.psi_ax * 0.05],
    axis=None,
    tofill=False,
)
plt.show()
