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
# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Fixed boundary equilibrium example
"""

# %% [markdown]
#
# # Fixed boundary equilibrium example
#
# Imports

# %%
from datetime import datetime

import dolfin
import matplotlib.pyplot as plt

from bluemira.base.components import PhysicalComponent
from bluemira.equilibria.fem_fixed_boundary.fem_magnetostatic_2D import (
    FemGradShafranovFixedBoundary,
)
from bluemira.equilibria.fem_fixed_boundary.file import save_fixed_boundary_to_file
from bluemira.equilibria.fem_fixed_boundary.utilities import create_mesh
from bluemira.equilibria.profiles import DoublePowerFunc, LaoPolynomialFunc
from bluemira.equilibria.shapes import KuiroukidisLCFS
from bluemira.geometry.face import BluemiraFace

# %% [markdown]
# Define some important values

# %%
R_0 = 9  # [m]
I_p = 17e6  # [A]
B_0 = 5  # [T]

# %% [markdown]
# Let's define a boundary shape for the fixed boundary equilibrium

# %%
parameterisation = KuiroukidisLCFS({
    "kappa_u": {"value": 1.7},
    "kappa_l": {"value": 1.8},
    "delta_u": {"value": 0.33},
    "delta_l": {"value": 0.4},
})

lcfs_shape = parameterisation.create_shape("LCFS", n_points=100)
lcfs_face = BluemiraFace(lcfs_shape)

# %% [markdown]
# Next we need to mesh this geometry

# %%
plasma = PhysicalComponent("plasma", lcfs_face)
plasma.shape.mesh_options = {"lcar": 0.3, "physical_group": "plasma_face"}
plasma.shape.boundary[0].mesh_options = {"lcar": 0.3, "physical_group": "lcfs"}

mesh = create_mesh(plasma, ".", "fixed_boundary_example", "fixed_boundary_example.msh")

dolfin.plot(mesh)
plt.show()
# %% [markdown]
# Now we define some profile functions for p' and FF'.
# We'll use some typical functional forms for this, but you are free to specify
# the flux functions using whichever callable you like.

# %%

p_prime = LaoPolynomialFunc([2, 3, 1])
ff_prime = DoublePowerFunc([1.5, 2])


# %% [markdown]
# Set up the solver and run it

# %%

solver = FemGradShafranovFixedBoundary(
    p_prime,
    ff_prime,
    mesh,
    I_p=I_p,
    R_0=R_0,
    B_0=B_0,
    p_order=2,
    max_iter=30,
    iter_err_max=1e-4,
    relaxation=0.05,
)
equilibrium = solver.solve(plot=True)


# %% [markdown]
# We can also update the flux functions and/or the mesh with new entities
# if we we wish to do so:

# %%
solver.set_profiles(
    p_prime=DoublePowerFunc([2.0, 1.0]), ff_prime=DoublePowerFunc([1.5, 2])
)
solver.solve(plot=True)

# %%
plasma.shape.mesh_options = {"lcar": 0.15, "physical_group": "plasma_face"}
plasma.shape.boundary[0].mesh_options = {"lcar": 0.15, "physical_group": "lcfs"}

mesh = create_mesh(plasma, ".", "fixed_boundary_example", "fixed_boundary_example.msh")
dolfin.plot(mesh)
plt.show()

solver.set_mesh(mesh)
solver.solve()

# %% [markdown]
# Save the result to a file

# %%
save_fixed_boundary_to_file(
    "my_fixed_boundary eqdsk.json",
    f"{datetime.now().strftime('%m/%d/%Y, %H:%M:%S')}",
    equilibrium,
    nx=100,
    nz=150,
)
