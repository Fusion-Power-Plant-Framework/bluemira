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
Double null example pulsed equilibrium problem
"""

# %%[markdown]
# # Double Null example pulsed equilibrium problem

# %%
from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt
from BLUEPRINT.base.lookandfeel import plot_defaults
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.parameterisations import flatD
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
# Build a basis function along which to optimise the positions of the PF coils

# %%
TF = flatD(2.2, 14.2, 0.0, npoints=500)
TF = Loop(**{"x": TF[0], "z": TF[1]})
TF.interpolate(200)
clip = np.where(TF.x >= 2.6)
TF = Loop(TF.x[clip], z=TF.z[clip])

# %%[markdown]
# Choose a plasma profile parameterisation (with initial values)

# %%
profile = DoublePowerFunc([2, 2])

# %%[markdown]
# Set up the equilibrium problem

# %%
DN = AbInitioEquilibriumProblem(
    R_0=6.8,
    B_0=5.2,
    A=2,
    Ip=14e6,
    betap=0.7,
    li=0.7,
    kappa=1.7,
    delta=0.3,
    r_cs=1.555,
    tk_cs=0.3,
    tfbnd=TF,
    n_PF=6,
    n_CS=7,
    eqtype="DN",
    rtype="Normal",
    profile=profile,
)

# %%[markdown]
# Let's assign some materials to the PF and CS coils

# %%
DN.coilset.assign_coil_materials("PF", "NbTi")
DN.coilset.assign_coil_materials("CS", "Nb3Sn")

# %%[markdown]
# Do an initial solve with unconstrained coils

# %%
eqref = DN.solve()

# %%[markdown]
# Carry out an initial breakdown optimisation (to determine the flux we can
# have at the start of flat-top)
#
# The maximum fields are implicitly specified in the coil materials we assigned
# The maximum currents are specified for the CS coils by their size, and for
# the PF coils it doesn't matter right now: we need to determine the breakdown
# flux.
#
# We do however need to specify some force constraints...

# %%
PF_Fz_max = 300e6
CS_Fz_sum = 250e6
CS_Fz_sep = 200e6

DN.breakdown(PF_Fz_max=PF_Fz_max, CS_Fz_sum=CS_Fz_sum, CS_Fz_sep=CS_Fz_sep)

psi_at_breakdown = DN.psi_bd

# %%[markdown]
# Alright now we want to optimise the positions of the PF coils.. over a given
# pulse length, and with some position constraints
#
# Set up some exclusions zones for the PF coils
#
# ## Upper port

# %%
UP = Loop(x=[4, 9.4, 9.4, 4, 4], z=[3, 3, 14.5, 14.5, 3])

# %%[markdown]
# ## Lower port

# %%
LP = Loop(x=[6, 6, 14, 14, 6], z=[-4, -6.6, -6.6, -4, -4])
LP.translate([0, 0, -1])

# %%[markdown]
# ## Equatorial port

# %%
EQ = Loop(x=[4, 16, 16, 4, 4], z=[-1.5, -1.5, 1.5, 1.5, -1.5])

# %%[markdown]
# Run the optimisation

# %%
DN.optimise_positions(
    max_PF_current=1.5 * 14e6,
    PF_Fz_max=PF_Fz_max,
    CS_Fz_sum=CS_Fz_sum,
    CS_Fz_sep=CS_Fz_sep,
    tau_flattop=0.5 * 3600,  # [s]
    v_burn=0.015,  # [V]
    psi_bd=psi_at_breakdown,
    pfcoiltrack=TF,
    pf_exclusions=[LP, EQ, UP],
    CS=False,
    plot=True,
    gif=False,
)

# %%[markdown]
# Generate a summary plot

# %%
DN.plot_summary()
plt.show()
