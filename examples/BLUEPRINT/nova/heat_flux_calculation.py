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

from bluemira.base.file import get_bluemira_path
from BLUEPRINT.geometry.loop import Loop
from bluemira.equilibria.equilibrium import Equilibrium

from BLUEPRINT.systems.firstwall import FirstWallSN
from time import time

# %%[markdown]
# Loading an equilibrium file and a first wall profile

t = time()

read_path = get_bluemira_path("equilibria", subfolder="data")
eq_name = "EU-DEMO_EOF.json"
eq_name = os.sep.join([read_path, eq_name])
eq = Equilibrium.from_eqdsk(eq_name, load_large_file=True)
x_box = [4, 15, 15, 4, 4]
z_box = [-11, -11, 11, 11, -11]
vv_box = Loop(x=x_box, z=z_box)

# %%[markdown]
# Calling the First Wall Class which will run
# the whole first wall optimisation being everything
# defined in the relevant __init__ function.
# Some basic inputs need to be specified.
# Particularly, if the user is dealing with a configuration
# different than STEP DN, this needs to be specified.
# At the moment, options alternative to STEP DN are
# "DEMO_DN" and "SN".

# %%
fw = FirstWallSN(
    FirstWallSN.default_params,
    {
        "equilibrium": eq,
        "vv_inner": vv_box,
        "SN": True,
        "div_vertical_outer_target": True,
        "div_vertical_inner_target": False,
    },
)

# %%[markdown]
# The funtion "plot_hf" gives a summary plot of
# optimised wall, heat flux and flux surfaces.

# %%

fig, ax = plt.subplots()
fw.plot_hf()
plt.show()


print(f"{time()-t:.2f} seconds")
