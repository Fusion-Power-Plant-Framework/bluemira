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
Double null example pulsed equilibrium problem
"""

# %%[markdown]
# # Double Null example pulsed equilibrium problem

# %%
from IPython import get_ipython
import numpy as np
import matplotlib.pyplot as plt
from bluemira.base.look_and_feel import plot_defaults
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.parameterisations import flatD
from BLUEPRINT.equilibria.run import AbInitioEquilibriumProblem
from BLUEPRINT.equilibria.profiles import DoublePowerFunc
from BLUEPRINT.equilibria.positioner import CoilPositioner
from BLUEPRINT.equilibria.coils import (
    CoilSet,
    SymmetricCircuit,
    PF_COIL_NAME,
    CS_COIL_NAME,
)

try:
    get_ipython().run_line_magic("matplotlib", "qt")
except AttributeError:
    pass

# %%[markdown]
# Set some plotting defaults, the Circuit run flag and the number of coils

# %%
plt.close("all")
plot_defaults()

# %%[markdown]
# Set some problem defaults

# %%
USE_CIRCUITS = True

n_PF = 6
n_CS = 7

R_0 = 6.8
kappa = 1.7
delta = 0.3
r_cs = 1.555
tk_cs = 0.3

B_0 = 5.2
A = 2
Ip = 14e6
betap = 0.7
li = 0.7

# %%[markdown]
# Build a basis function along which to optimise the positions of the PF coils

# %%
TF = flatD(2.2, 14.2, 0.0, npoints=500)
TF = Loop(**{"x": TF[0], "z": TF[1]})
TF.interpolate(200)
clip = np.where(TF.x >= 2.6)
TF = Loop(TF.x[clip], z=TF.z[clip])

# %%[markdown]
# Make coilset for circuits if required

# %%


def build_circuit_coilset(coilset, odd):
    """
    Build circuit based coilset
    """
    old_coils, pf_coils, cs_coils = [], [], []
    central = None
    for i in reversed(range(1, coilset.n_PF + 1)):
        coil_name = PF_COIL_NAME.format(i)
        if i <= coilset.n_PF // 2:
            old_coils.append(coilset.coils[coil_name])

    for i in reversed(range(1, coilset.n_CS + 1)):
        coil_name = CS_COIL_NAME.format(i)
        if i <= coilset.n_CS // 2:
            old_coils.append(coilset.coils[coil_name])
        elif odd and i == coilset.n_CS // 2 + 1:
            central = coilset.coils[coil_name]

    for old_coil in old_coils:
        coil = SymmetricCircuit(
            old_coil.x,
            old_coil.z,
            dx=old_coil.dx,
            dz=old_coil.dz,
            current=old_coil.current,
            ctype=old_coil.ctype,
            name=old_coil.name,
            control=old_coil.control,
            j_max=old_coil.j_max,
            b_max=old_coil.b_max,
        )
        if coil.ctype == "PF":
            pf_coils.append(coil)
        elif coil.ctype == "CS":
            cs_coils.append(coil)

    if central is not None:
        cs_coils.append(central)

    return pf_coils, cs_coils


if USE_CIRCUITS:

    cs = CoilPositioner(
        R_0=R_0,
        A=A,
        kappa=kappa,
        delta=delta,
        x_cs=r_cs,
        tk_cs=tk_cs,
        track=TF,
        n_PF=n_PF,
        n_CS=n_CS,
        rtype="Normal",
    )
    pf_coils, cs_coils = build_circuit_coilset(cs.make_coilset(), odd=cs.n_CS % 2 == 1)

    coilset = CoilSet(pf_coils + cs_coils, R_0=R_0)
    n_PF, n_CS = None, None
else:
    coilset = None


# %%[markdown]
# Choose a plasma profile parameterisation (with initial values)

# %%
profile = DoublePowerFunc([2, 2])

# %%[markdown]
# Set up the equilibrium problem

# %%
DN = AbInitioEquilibriumProblem(
    R_0=R_0,
    B_0=B_0,
    A=A,
    Ip=Ip,
    betap=betap,
    li=li,
    kappa=kappa,
    delta=delta,
    r_cs=r_cs,
    tk_cs=tk_cs,
    tfbnd=TF,
    n_PF=n_PF,
    n_CS=n_CS,
    eqtype="DN",
    rtype="Normal",
    profile=profile,
    coilset=coilset,
)

print(DN.coilset)

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
