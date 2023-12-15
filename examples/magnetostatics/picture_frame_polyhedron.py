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
# # Simple PictureFrame polyhedral cross-section example

# %%
import matplotlib.pyplot as plt
import numpy as np

from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.parameterisations import PictureFrame
from bluemira.magnetostatics.baseclass import SourceGroup
from bluemira.magnetostatics.biot_savart import BiotSavartFilament
from bluemira.magnetostatics.circuits import ArbitraryPlanarPolyhedralXSCircuit
from bluemira.magnetostatics.polyhedral_prism import PolyhedralPrismCurrentSource

parameterisation = PictureFrame(
    {
        "x1": {"value": 3.5},
        "x2": {"value": 10.0},
        "z1": {"value": 9},
        "z2": {"value": -9},
        "ri": {"value": 0.75},
        "ro": {"value": 2.0},
    }
)
wire = parameterisation.create_shape()


current = 10.0e6


tf_xs = Coordinates({"x": [-0.5, 0.5, 0.5, -0.5], "z": [-0.3, -0.6, 0.6, 0.3]})
tf_xs.close()

sources = []
for i, edge in enumerate(wire.edges):
    if i % 2 == 0:
        start = edge.end_point()
        end = edge.start_point()
        origin = 0.5 * (start.xyz.T[0] + end.xyz.T[0])
        ds = end.xyz.T[0] - start.xyz.T[0]
        normal = np.array([0, -1, 0])
        t_vec = np.cross(normal, ds / np.linalg.norm(ds))
        source = PolyhedralPrismCurrentSource(
            origin=origin,
            ds=ds,
            normal=normal,
            t_vec=t_vec,
            xs_coordinates=tf_xs,
            alpha=0,
            beta=0,
            current=current,
        )
        sources.append(source)
    else:
        coords = edge.discretize(5)
        # Careful, not sure
        coords.set_ccw([0, -1, 0])
        source = ArbitraryPlanarPolyhedralXSCircuit(coords, tf_xs, current=-current)
        sources.append(source)


f, (ax1, ax2) = plt.subplots(1, 2, subplot_kw={"projection": "3d"})
source = SourceGroup(sources)
source.plot(ax1)

x, z = np.linspace(0, 12), np.linspace(-10, 10)
xx, zz = np.meshgrid(x, z)
yy = np.zeros_like(xx)

Bx, By, Bz = source.field(xx, yy, zz)
B = np.sqrt(Bx**2 + By**2 + Bz**2)
ax1.contourf(xx, B, zz, zdir="y", offset=0)

bs = BiotSavartFilament(wire.discretize(100, byedges=True), radius=1)
bs.plot(ax2)
Bx, By, Bz = source.field(xx, yy, zz)
B = np.sqrt(Bx**2 + By**2 + Bz**2)
ax2.contourf(xx, B, zz, zdir="y", offset=0)

plt.show()
