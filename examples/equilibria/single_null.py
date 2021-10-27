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
SN pulsed equilibrium example
"""

# %%[markdown]
# # Single Null example pulsed equilibrium problem

# %%
from IPython import get_ipython
import os
import numpy as np
import matplotlib.pyplot as plt

from bluemira.base.file import get_bluemira_path
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
fp = get_bluemira_path("BLUEPRINT/Geometry", subfolder="data")
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

# %%[markdown]
# Make all coils use by making regions for coils

# %%
pf_coilregions = {}
region_coils = {
    1: {"x": 6.01, "z": 10.09},
    3: {"x": 19.00, "z": 3.67},
    5: {"x": 10.92, "z": -10.77},
}
for coil in SN.coilset.coils.values():
    coil_number = int(coil.name.split("_")[-1])
    if coil.ctype != "PF" or coil_number not in region_coils:
        continue
    dx, dz = coil.dx * 4, coil.dz * 4  # Arbitrarily sized region
    pf_coilregions[coil.name] = Loop(
        x=region_coils[coil_number]["x"] + np.array([-dx, dx, dx, -dx, -dx]),
        z=region_coils[coil_number]["z"] + np.array([-dz, -dz, dz, dz, -dz]),
    )

# %%[markdown]
# Let's look at the coilset on its own

# %%
SN.coilset.plot()

# %%[markdown]
# Define some exclusion zones for the PF coils

# %%
UP = Loop(x=[7.5, 14, 14, 7.5, 7.5], z=[3, 3, 14.5, 14.5, 3])
LP = Loop(x=[10, 10, 15, 22, 22, 15, 10], z=[-6, -10, -13, -13, -8, -8, -6])
EQ = Loop(x=[14, 22, 22, 14, 14], z=[-1.4, -1.4, 1.4, 1.4, -1.4])

# %%[markdown]
# Look at the "track" for the PF coil locations, and the exclusion zones:

# %%
f, ax = plt.subplots()

TF.plot(ax, fill=False)
UP.plot(ax, edgecolor="r", facecolor="r", alpha=0.5)
LP.plot(ax, edgecolor="r", facecolor="r", alpha=0.5)
EQ.plot(ax, edgecolor="r", facecolor="r", alpha=0.5)

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
# The resulting equilbria will automatically be converged once the coil sizes
# have been fixed at their maximum
# (sometimes problematic for end of flattop)

# The following method will:
# *  calculate the breakdown flux for this reactor
# *  optimise the coil positions for the start and end of flat-top
# *  converge the resulting SOF and EOF equilibria

# %%
SN.optimise_positions(
    max_PF_current=25e6,  # [A]
    PF_Fz_max=400e6,  # [N]
    CS_Fz_sum=300e6,  # [N]
    CS_Fz_sep=250e6,  # [N]
    tau_flattop=1.5 * 3600,  # [s]
    v_burn=0.04,  # [V]
    psi_bd=None,
    pfcoiltrack=TF,
    pf_exclusions=[LP, EQ, UP],
    pf_coilregions=pf_coilregions,
    CS=False,
    plot=True,
    gif=False,
)
