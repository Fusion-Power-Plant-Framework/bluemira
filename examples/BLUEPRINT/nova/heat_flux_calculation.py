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
Heat flux calculation example
"""
# %%[markdown]
# # Heat Flux Calculation
# %%
import os
import matplotlib.pyplot as plt

from BLUEPRINT.base.file import get_BP_path
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.equilibria.equilibrium import Equilibrium

from BLUEPRINT.systems.firstwall import FirstWall
from time import time

# %%[markdown]
# Loading an equilibrium file and a first wall profile

t = time()

read_path = get_BP_path("equilibria", subfolder="data/BLUEPRINT")
eq_name = "EU-DEMO_EOF.json"
eq_name = os.sep.join([read_path, eq_name])
eq = Equilibrium.from_eqdsk(eq_name, load_large_file=True)
profile = Loop.from_file("first_wall.json")

# %%[markdown]
# Calling the First Wall Class
# First Wall Class will create a set of flux surfaces between the LCFS and the FW
# In this case we are going to calculate the heat fluxes onto the provided profile
# Alternatively we can also call the class only providing an equilibrium
# In this case a "preliminary first wall" profile will be designed

fw = FirstWall(FirstWall.default_params, {"equilibrium": eq, "profile": profile})

# %%[markdown]
# We are going to define the key parameters for the flux surfaces

(
    lfs_first_intersection,
    hfs_first_intersection,
    qpar_omp,
    qpar_local_lfs,
    qpar_local_hfs,
    glancing_angle_lfs,
    glancing_angle_hfs,
    f_lfs,
    f_hfs,
) = fw.define_flux_surfaces_parameters()
# %%[markdown]
# We are going to calculate the heat flux onto the FW

x, z, hf, hf_lfs, hf_hfs, th = fw.calculate_heat_flux_lfs_hfs(
    lfs_first_intersection,
    hfs_first_intersection,
    qpar_omp,
    qpar_local_lfs,
    qpar_local_hfs,
    glancing_angle_lfs,
    glancing_angle_hfs,
)

# %%[markdown]
# First wall, separatrix and flux surfaces
f, ax = plt.subplots()
fw.lcfs.plot(ax, fill=False, facecolor="b", linewidth=0.1)
fw.separatrix.plot(ax, fill=False, facecolor="b", linewidth=0.5)
fw.profile.plot(ax, fill=False, facecolor="b", linewidth=0.1)
for fs in fw.flux_surfaces:
    fs.loop.plot(ax, fill=False, facecolor="r", linewidth=0.1)

plt.show()

# %%[markdown]
# First wall shape, hit points and heat flux values
fig, ax = plt.subplots()
fw.profile.plot(ax=ax)
cs = ax.scatter(x, z, c=hf, cmap="viridis", zorder=100)
bar = fig.colorbar(cs, ax=ax)
bar.set_label("Heat Flux [MW/m^2]")

plt.show()

# %%[markdown]
# Heat flux values against poloidal location

plt.style.use("seaborn")
fig, ax = plt.subplots()
ax.scatter(th, hf, c=hf, cmap="viridis", s=100)
ax.legend()
ax.set_title("Heat flux on the wall", fontsize=24)
ax.set_xlabel("Theta", fontsize=14)
ax.set_ylabel("HF (MW/m^2)", fontsize=14)
ax.tick_params(axis="both", which="major", labelsize=14)
plt.show()

print(f"{time()-t:.2f} seconds")
