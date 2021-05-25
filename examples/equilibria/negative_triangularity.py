# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.
"""
NT pulsed equilibrium example
"""

# %%[markdown]
# # Negative Triangularity example pulsed equilibrium problem

# %%
from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt
from BLUEPRINT.base.lookandfeel import plot_defaults
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.parameterisations import PictureFrame
from BLUEPRINT.equilibria.run import AbInitioEquilibriumProblem
from BLUEPRINT.equilibria.profiles import DoublePowerFunc

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
TF = PictureFrame()
TF.xo["x1"] = {"value": 3.5, "lb": 3.5, "ub": 5}  # inner leg
TF.xo["x2"] = {"value": 16.2, "lb": 14, "ub": 20}  # outer leg
TF.xo["r"] = {"value": 2, "lb": 1.999, "ub": 2.001}  # Corner radius
TF.xo["z1"] = {"value": 9, "lb": 5, "ub": 15}  # Vertical height
TF.xo["z2"] = {"value": -9, "lb": -15, "ub": -3}  # vertical
TF = Loop(**TF.draw())
clip = np.where(TF.x >= 3.6)
TF = Loop(TF.x[clip], z=TF.z[clip])

# %%[markdown]
# Choose a flux function parameterisation (with some initial values)
# The shape function parameters will be optimised to meet the integral plasma
# parameters we specify later on

# %%
profile = DoublePowerFunc([2, 2])

# %%[markdown]
# Set up an equilibrium problem (a typical sized pulsed EU-DEMO like machine)
#
# eps BD opt 1e-3
# gamma FBI 1e-15

# %%
NTT = AbInitioEquilibriumProblem(
    R_0=8,
    B_0=5.2,
    A=3.2,
    Ip=15e6,
    betap=1.2,
    li=0.8,
    kappa=1.655,
    delta=-0.3,
    r_cs=2.6,
    tk_cs=0.32,
    tfbnd=TF,
    n_PF=5,
    n_CS=6,
    eqtype="SN",
    rtype="Normal",
    profile=profile,
)

# %%[markdown]
# Do an initial solve

# %%
eqref = NTT.solve(plot=True)

# %%[markdown]
# Now let's optimise:
# *  positions of the PF coils
# *  currents of the PF and CS coils
#
# constraining:
# *  plasma shape
# *  plasma integral values (I_p, beta_p, l_i)
# *  coil positions         (L)
# *  coil currents          (I)
# *  coil forces            (F)
# *  field at coils         (B)
# *  pulse length           (tau_flattop)
#
# Define some exclusion zones

# %%
UP = Loop(x=[6, 12, 12, 6, 6], z=[3, 3, 14.5, 14.5, 3])
LP = Loop(x=[10, 10, 12, 22, 22, 10], z=[-6, -6, -11, -11, -6, -6])
EQ = Loop(x=[14, 22, 22, 14, 14], z=[-1.4, -1.4, 1.4, 1.4, -1.4])

# %%[markdown]
# Run the optimisation

# %%
NTT.optimise_positions(
    max_PF_current=1.5 * 15e6,
    PF_Fz_max=350e6,
    CS_Fz_sum=300e6,
    CS_Fz_sep=250e6,
    tau_flattop=1.1 * 3600,
    v_burn=0.04,
    psi_bd=None,  # will auto-calculate breakdown flux
    pfcoiltrack=TF,
    pf_exclusions=[LP, EQ, UP],
    CS=False,
    plot=True,
    gif=False,
)

# %%[markdown]
# Generate a summary plot

# %%
NTT.plot_summary()

# %%[markdown]
# Let's have a look at the plasma over the pulse

# %%
print(NTT.report())

# %%[markdown]
# It doesn't look very good... we probably need some more PF coils in there.
#
# Or actually..! Fix the breakdown optimiser which I think is not doing a
# great job here :(
#
# Let's see what the optimisers have done

# %%
NTT.opt_report()
