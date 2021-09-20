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
An example of how to use FirstWallDN from firstwall.py
to calculate the heat flux due to charged particles onto
the first wall and optimise the shape design.
"""

# %%[markdown]
# Heat Flux Calculation and first wall shaping

# %%
import os
import matplotlib.pyplot as plt
from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from BLUEPRINT.systems.firstwall import FirstWallDN
from BLUEPRINT.geometry.loop import Loop
from time import time

t = time()

# %%[markdown]
# Loading an equilibrium file

# %%
read_path = get_bluemira_path("bluemira/equilibria/test_data", subfolder="tests")
eq_name = "DN-DEMO_eqref.json"
eq_name = os.sep.join([read_path, eq_name])
eq = Equilibrium.from_eqdsk(eq_name, load_large_file=True)
x_box = [4, 15, 15, 4, 4]
z_box = [-11, -11, 11, 11, -11]
vv_box = Loop(x=x_box, z=z_box)

# %%[markdown]
# Calling the First Wall Class the create a
# preliminary first wall profile devoid of the divertor.
# In input, besides the equilibrium, we need to provide a
# a vacuum vessel profile.
# Adding the divertor to the above mentioned first wall.

# %%
fw = FirstWallDN(
    FirstWallDN.default_params, {"equilibrium": eq, "vv_inner": vv_box, "DEMO_DN": True}
)
divertor_loops = fw.make_divertor(fw.profile)
fw_diverted = fw.attach_divertor(fw.profile, divertor_loops)

# %%[markdown]
# Plotting the initial first wall profile

# %%
f, ax = plt.subplots()
fw_diverted.plot(ax=ax, fill=False, facecolor="b", linewidth=0.5)

# %%[markdown]
# Setting the while loop to iterate the first wall optimisation
# until the heat flux is within the limit
# In the first iteration, the preliminary profile made above is used

# %%
profile = fw.profile
hf_wall_max = 1

while hf_wall_max > 0.5:
    fw_opt = FirstWallDN(
        FirstWallDN.default_params,
        {"equilibrium": eq, "profile": profile, "vv_inner": vv_box, "DEMO_DN": True},
    )

    # Calculate the parallel contribution of the heat flux
    # at the outer and inner mid-plane
    qpar_omp, qpar_imp = fw_opt.q_parallel_calculation()

    # Find the first intersection between each flux surface
    # and the first wall profile at the lfs and hfs
    lfs_hfs_intersections = fw_opt.find_intersections()
    lfs_hfs_first_int = fw_opt.find_first_intersections(*lfs_hfs_intersections)

    # At each intersection, calculate the local parallel
    # contribution of the heat flux, the incident angle between
    # fs and fw, and the flux expansion
    (
        qpar_local_lfs_hfs,
        incindent_angle_lfs_hfs,
        f_list_lfs_hfs,
    ) = fw_opt.define_flux_surfaces_parameters_to_calculate_heat_flux(
        qpar_omp,
        qpar_imp,
        *lfs_hfs_first_int,
    )

    # Associate to each intersection the x and z coordinates
    # and the heat flux value
    x, z, hf = fw_opt.calculate_heat_flux(
        *lfs_hfs_first_int,
        *qpar_local_lfs_hfs[0],
        *qpar_local_lfs_hfs[1],
        *incindent_angle_lfs_hfs[0],
        *incindent_angle_lfs_hfs[1],
    )

    # As only the first wall is optimised, isolate the heat fluxes on
    # the first wall from the whole profile (which includes divertor)
    x_wall = []
    z_wall = []
    hf_wall = []
    for list_x, list_z, list_hf in zip(x, z, hf):
        for x, z, hf in zip(list_x, list_z, list_hf):
            if z < fw.points["x_point"]["z_up"] and z > fw.points["x_point"]["z_low"]:
                x_wall.append(x)
                z_wall.append(z)
                hf_wall.append(hf)

    # Optimise the first wall shape and verify that for the final iteration
    # the heat flux is below the limit
    optimised_profile = profile
    for x, z, hf in zip(x_wall, z_wall, hf_wall):
        optimised_profile = fw_opt.modify_fw_profile(optimised_profile, x, z, hf)
    hf_wall_max = max(hf_wall)
    profile = optimised_profile
    print(hf_wall_max)

# %%[markdown]
# Close the profile by adding the divertor to the final fw

# %%
divertor_loops = fw_opt.make_divertor(profile)
fw_opt_diverted = fw_opt.attach_divertor(fw_opt.profile, divertor_loops)

# %%[markdown]
# Plotting First wall, separatrix and flux surfaces

# %%
f, ax = plt.subplots()
fw_opt.separatrix.plot(ax, fill=False, facecolor="b", linewidth=0.1)
for fs in lfs_hfs_first_int[0]:
    if len(fs) != 0:
        fs[2].plot(ax, fill=False, facecolor="r", linewidth=0.1)
for fs in lfs_hfs_first_int[1]:
    if len(fs) != 0:
        fs[2].plot(ax, fill=False, facecolor="r", linewidth=0.1)
for fs in lfs_hfs_first_int[2]:
    if len(fs) != 0:
        fs[2].plot(ax, fill=False, facecolor="r", linewidth=0.1)
for fs in lfs_hfs_first_int[3]:
    if len(fs) != 0:
        fs[2].plot(ax, fill=False, facecolor="r", linewidth=0.1)
fw_opt_diverted.plot(ax=ax, fill=False, facecolor="b", linewidth=0.5)

# %%[markdown]
# Plotting First wall shape, intersection points and heat flux values
# on the first wall (no divertor)

# %%
fig, ax = plt.subplots()
fw_opt_diverted.plot(ax=ax, fill=False)
cs = ax.scatter(x_wall, z_wall, s=25, c=hf_wall, cmap="viridis", zorder=100)
bar = fig.colorbar(cs, ax=ax)
bar.set_label("Heat Flux [MW/m^2]")

print(f"{time()-t:.2f} seconds")
