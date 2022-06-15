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
from time import time

import matplotlib.pyplot as plt

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.geometry._deprecated_loop import Loop
from BLUEPRINT.systems.firstwall import FirstWallSN

# %%[markdown]
# Loading an equilibrium file and a first wall profile

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
t = time()
fw = FirstWallSN(
    FirstWallSN.default_params,
    {
        "equilibrium": eq,
        "vv_inner": vv_box,
        "SN": True,
        "DEMO_like_divertor": True,
        "div_vertical_outer_target": True,
        "div_vertical_inner_target": False,
        # Can't quite replicate the extremely spaced values from the above... but it's
        # the same for all intents and purposes. I couldn't find much on the spacing for
        # the single null case, except for step_size = 0.02 in line 2340 of firstwall.py
        # This 0.02 is then doubled in line 2352, but if I set 0.04 I don't get the same
        # This is I think because of the offset from the LCFS which is taken for reasons
        # I don't understand, so I increase from 0.04 a bit to get roughly the same
        # number of flux surfaces.
        # NOTE: Such low resolutions give fairly meaningless results...
        "dx_mp": 0.05,
    },
)
fw.build()
print(f"{time()-t:.2f} seconds")

# %%[markdown]
# The funtion "plot_hf" gives a summary plot of
# optimised wall, heat flux and flux surfaces.

# %%

fig, ax = plt.subplots()
fw.plot_hf()
