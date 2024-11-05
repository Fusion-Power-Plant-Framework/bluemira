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
Simple HelmholzCage example with different current sources.
"""

# %% [markdown]
# # Simple HelmholtzCage example
# ## Introduction
#
# In this example we will build some HelmholtzCages with different types of current
# sources.

# %%
import matplotlib.pyplot as plt  # noqa: F401
import numpy as np  # noqa: F401

from bluemira.geometry.parameterisations import TripleArc
from bluemira.geometry.tools import make_circle
from bluemira.magnetostatics.baseclass import SourceGroup
from bluemira.magnetostatics.circuits import (
    ArbitraryPlanarRectangularXSCircuit,
    HelmholtzCage,
)

# %% [markdown]
# Set up some geometry and key parameters

# %%
n_TF = 12
current = 20e6
breadth = 0.5
depth = 1.0
radius = 6
x_c = 9
z_c = 0

centreline = TripleArc({
    "x1": {"value": 3.746477, "lower_bound": 3, "upper_bound": 5, "fixed": True},
    "dz": {
        "value": -0.1853300309672189,
        "lower_bound": -1,
        "upper_bound": 1,
        "fixed": False,
    },
    "sl": {"value": 5.0, "lower_bound": 5, "upper_bound": 10, "fixed": False},
    "f1": {
        "value": 7.647219147792518,
        "lower_bound": 2,
        "upper_bound": 12,
        "fixed": False,
    },
    "f2": {
        "value": 3.4287056384774584,
        "lower_bound": 2,
        "upper_bound": 12,
        "fixed": False,
    },
    "a1": {
        "value": 43.66109481422152,
        "lower_bound": 5,
        "upper_bound": 120,
        "fixed": False,
    },
    "a2": {
        "value": 46.07769446619384,
        "lower_bound": 10,
        "upper_bound": 120,
        "fixed": False,
    },
}).create_shape()


# %% [markdown]
# Make an analytical circuit with a rectangular cross-section comprised
# of several trapezoidal prism elements

# %%
coordinates = centreline.discretise(ndiscr=100, byedges=True)
analytical_circuit1 = ArbitraryPlanarRectangularXSCircuit(
    coordinates, breadth=breadth, depth=depth, current=current
)


# %% [markdown]
# Pattern the circuit into HelmholtzCages for TF

# %%
analytical_tf_cage1 = HelmholtzCage(analytical_circuit1, n_TF=n_TF)

# %% [markdown]
# PF coils


PF_breadth = [
    0.33886403,
    0.26956804,
    0.35292703,
    0.44555113,
    0.57497131,
    0.74353211,
    0.355065,
    0.355065,
    0.355065,
    0.355065,
    0.355065,
]
PF_depth = [
    0.33886403,
    0.26956804,
    0.35292703,
    0.44555113,
    0.57497131,
    0.74353211,
    1.53881955,
    1.53881955,
    3.0776391,
    1.53881955,
    1.53881955,
]
PF_current = [
    -5741441.40305201,
    -3134079.54660759,
    -5850113.49072429,
    -9017874.46902416,
    -15358326.84239624,
    25068239.68867189,
    14382098.23881778,
    -36061143.57026877,
    -52332056.79435857,
    -29852223.19724484,
    -36061143.57026877,
]
PF_radius = [
    5.5057532,
    14.97472545,
    17.55905918,
    18.12441602,
    11.60092321,
    7.91553047,
    2.530965,
    2.530965,
    2.530965,
    2.530965,
    2.530965,
]

PF_z = [
    10.1879471,
    7.76496629,
    4.55091839,
    -3.73274898,
    -10.12808809,
    -10.66741308,
    7.65440824,
    4.47676914,
    -0.23968951,
    -4.95614816,
    -8.13378725,
]
pf = []

for breadth, depth, current, radius, height in zip(
    PF_breadth, PF_depth, PF_current, PF_radius, PF_z, strict=False
):
    circle = make_circle(radius, center=(0, 0, height), axis=(0, 0, 1)).discretise(
        ndiscr=100, byedges=True
    )

    pf.append(
        ArbitraryPlanarRectangularXSCircuit(
            circle, breadth=breadth, depth=depth, current=current
        )
    )

full_cage = SourceGroup([analytical_tf_cage1, *pf])

# full_cage.plot()
# plt.show()

# %% [markdown]
# Calculate fields at given locations

# nx, nz = 50, 50
# x = np.linspace(0, 18, nx)
# z = np.linspace(0, 14, nz)
# xx, zz = np.meshgrid(x, z, indexing="ij")

# analytical_xz_fields = full_cage.field(xx, np.zeros_like(xx), zz)

# analytical_xz_fields = np.sqrt(np.sum(analytical_xz_fields**2, axis=0))
