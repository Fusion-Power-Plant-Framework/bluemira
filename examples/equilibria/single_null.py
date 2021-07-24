# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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
SN pulsed equilibrium example
"""

# %%[markdown]
# # Single Null example pulsed equilibrium problem

# %%
from IPython import get_ipython
import os
import numpy as np
import matplotlib.pyplot as plt
from BLUEPRINT.base.file import get_BP_path
from bluemira.base.look_and_feel import plot_defaults
from bluemira.geometry._deprecated_loop import Loop
from bluemira.equilibria.run import AbInitioEquilibriumProblem
from bluemira.equilibria.profiles import DoublePowerFunc

try:
    get_ipython().run_line_magic("matplotlib", "qt")
except AttributeError:
    pass

# %%[markdown]
# Set some plotting defaults

# %%
plt.close("all")
plot_defaults()

# %%[markdown]
# Make a TF coil shape and use it as an exclusion zone object

# %%
fp = get_BP_path("Geometry", subfolder="data")
TF = Loop.from_file(os.sep.join([fp, "TFreference.json"]))
TF = TF.offset(2.4)
clip = np.where(TF.x >= 3.5)
TF = Loop(TF.x[clip], z=TF.z[clip])
TF.interpolate(200)

# %%[markdown]
# Choose a flux function parameterisation (with some initial values)
# The shape function parameters will be optimised to meet the integral plasma
# parameters we specify later on

# %%
p = DoublePowerFunc([2, 2])

# %%[markdown]
# Set up an equilibrium problem (a typical sized pulsed EU-DEMO like machine)

# %%
SN = AbInitioEquilibriumProblem(
    R_0=9,  # [m]
    B_0=5.8,  # [T]
    A=3.1,
    Ip=19e6,  # [A]
    betap=1.3,
    li=0.8,
    kappa_u=1.65,
    kappa_l=1.8,
    delta_u=0.4,
    delta_l=0.4,
    psi_u_neg=180,
    psi_u_pos=0,
    psi_l_neg=-120,
    psi_l_pos=30,
    div_l_ib=1.0,
    div_l_ob=1.45,
    r_cs=2.85,  # [m]
    tk_cs=0.3,  # [m]
    tfbnd=TF,
    n_PF=6,
    n_CS=5,
    eqtype="SN",
    rtype="Normal",
    profile=p,
    psi=None,
)

# %%[markdown]
# Get an initial unconstrained solution

# %%
eqref = SN.solve()
