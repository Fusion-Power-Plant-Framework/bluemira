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
ST generated from spherical harmonics equilibrium attempt
Based off of work by O. Bardsley
"""

# %%[markdown]

# # Script to demonstrate an initial equilbrium calculated using a JETTO eqdsk and spherical harmonics

# # Imports

# Import necessary Equilbrium module definitions.

# %%

import json
from cmath import pi

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from scipy.special import lpmv

from bluemira.base.constants import MU_0
from bluemira.display.auto_config import plot_defaults
from bluemira.equilibria.coils import Coil, CoilSet, PlasmaCoil, SymmetricCircuit
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.file import EQDSKInterface
from bluemira.equilibria.find import find_OX_points

# Clean up and make plots look good
plt.close("all")
plot_defaults()

# %%[markdown]

# # Input Definitions

# %%

# JSON
file = open("../bluemira/data/equilibria/DN-DEMO_eqref.json")
data = json.load(file)

# EQDSK
file_path = "../bluemira/data/eqdsk/jetto.eqdsk_out"
reader = EQDSKInterface()
fixed_boundary_eqdsk = reader.read(str(file_path))


# %%[markdown]

# ## Plasma Profiles and vacuum psi

# %%

eqdsk_eq = Equilibrium.from_eqdsk(file_path)
profiles = eqdsk_eq.profiles

eqdsk_psi = fixed_boundary_eqdsk["psi"]
o_points, x_points = find_OX_points(eqdsk_eq.grid.x, eqdsk_eq.grid.z, eqdsk_psi)

# There's an issue creating the mask on which to calculate / present jtor. It struggles
# to calculate the LCFS somewhere, even though this can be calcualted directly from
# the eqdsk xbdry and zbdry.
jtor = profiles.jtor(
    eqdsk_eq.grid.x, eqdsk_eq.grid.z, eqdsk_psi, o_points=o_points, x_points=x_points
)

plasma = PlasmaCoil(jtor, eqdsk_eq.grid)

plasma_psi = plasma.psi(eqdsk_eq.grid.x, eqdsk_eq.grid.z)

vacuum_psi = eqdsk_psi - plasma_psi


# %%[markdown]

# ## Create the set of collocation points for the harmonics.

# %%

n = 7  # Number of taragets without extrema
x_bdry = np.array(fixed_boundary_eqdsk["xbdry"])
z_bdry = np.array(fixed_boundary_eqdsk["zbdry"])
r_bdry = np.sqrt(x_bdry**2 + z_bdry**2)
theta_bdry = np.arctan2(x_bdry, z_bdry)
collocation_theta = np.linspace(np.amin(theta_bdry), np.amax(theta_bdry), n + 2)
collocation_theta = collocation_theta[1:-1]
collocation_r = 0.9 * np.amax(r_bdry) * np.ones(n)
collocation_x = collocation_r * np.sin(collocation_theta)
collocation_z = collocation_r * np.cos(collocation_theta)

d = 0.1
extrema_x = np.array(
    [
        np.amin(x_bdry) + d,
        np.amax(x_bdry) - d,
        x_bdry[np.argmax(z_bdry)],
        x_bdry[np.argmin(z_bdry)],
    ]
)
extrema_z = np.array([0, 0, np.amax(z_bdry) - d, np.amin(z_bdry) + d])
collocation_x = np.concatenate([collocation_x, extrema_x])
collocation_z = np.concatenate([collocation_z, extrema_z])
n_collocation = np.size(collocation_x)
collocation_r = np.sqrt(collocation_x**2 + collocation_z**2)
collocation_theta = np.arctan2(collocation_x, collocation_z)
collocation_r, collocation_theta, n_collocation


# %%[markdown]

# ## Initial CoilSet

# %%

# Potentially need to be wary of difference in how dx, dz are defined between
# bluemira and json file

# Create the initial coilset from JSON file.

# coil_x = np.array(data["xc"])
# coil_z = np.array(data["zc"])
# coil_dx = np.array(data["dxc"])
# coil_dz = np.array(data["dzc"])
# currents = np.array(data["Ic"])
# coils = []
# for i in range(len(coil_x)):
#     coil = Coil(
#         coil_x[i],
#         coil_z[i],
#         current=currents[i],
#         dx=coil_dx[i] / 2,
#         dz=coil_dz[i] / 2,
#         ctype="PF",
#     )
#     coils.append(coil)
# coilset = CoilSet(coils)

# Create initial coilset for the eqdsk (could need fiddling with)

coil_x = [1.05, 6.85, 6.85, 1.05, 3.2, 5.7, 5.3]
coil_z = [7.85, 4.75, 3.35, 6.0, 8.0, 7.8, 5.50]
coil_dx = [0.45, 0.5, 0.5, 0.3, 0.6, 0.5, 0.25]
coil_dz = [0.5, 0.8, 0.8, 0.8, 0.5, 0.5, 0.5]
currents = [0, 0, 0, 0, 0, 0, 0]

circuits = []
for i in range(len(coil_x)):
    coil = Coil(
        coil_x[i],
        coil_z[i],
        dx=coil_dx[i] / 2,
        dz=coil_dz[i] / 2,
        current=currents[i],
        ctype="PF",
    )
    circuit = SymmetricCircuit(coil)
    circuits.append(circuit)
coilset = CoilSet(circuits)


# %%[markdown]

# ## Generating matrices for spherical harmonics and current calculation

# %%

# Calculate flux function at collocation points

points = np.array((eqdsk_eq.grid.x.flatten(), eqdsk_eq.grid.z.flatten())).T
values = vacuum_psi.flatten()
collocation_psivac = griddata(points, values, (collocation_x, collocation_z))

# Typical lengthscale
r_t = np.amax(x_bdry)

max_degree = 11
r_f = np.sqrt(coil_x**2 + coil_z**2)
theta_f = np.arctan2(coil_x, coil_z)

psi_vac = np.array([])
harmonics2collocation = np.zeros(n_collocation, max_degree)
harmonics2collocation[:, 0] = 1
# Oli sets n_harmonics (max degree) to be x_p - 1
for degree in np.arange(1, max_degree):
    harmonics2collocation[:, degree] = (
        collocation_r ** (degree + 1)
        * np.sin(collocation_theta)
        * lpmv(1, degree, np.cos(collocation_theta))
        / ((r_t**degree) * np.sqrt(degree * (degree + 1)))
    )

currents2harmonics = np.zeros((max_degree, np.size(coil_x)))
for degree in np.arange(max_degree):
    currents2harmonics[degree, :] = (
        0.5
        * MU_0
        * (r_t / r_f) ** degree
        * np.sin(theta_f)
        * lpmv(1, degree, np.cos(theta_f))
        / np.sqrt(degree * (degree + 1))
    )


harmonic_ampltidues = np.linalg.lstsq(harmonics2collocation, collocation_psivac)

# May need messing around with harmonic_amplitude array indexing here and multiple of 2pi
currents = np.linalg.lstsq(currents2harmonics, harmonic_ampltidues[1:]) / (2 * pi)

print(currents)

# JSON coil currents
# print(data["Ic"])

# Plots of equilibrium
