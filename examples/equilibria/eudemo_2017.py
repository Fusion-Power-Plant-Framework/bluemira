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


# %%[markdown]

# # EU-DEMO 2017 reference breakdown and equilibrium benchmark

# %%

import os
import numpy as np
import matplotlib.pyplot as plt
from bluemira.base.file import get_bluemira_path
from bluemira.base.look_and_feel import bluemira_print, plot_defaults
from bluemira.equilibria.file import EQDSKInterface
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.coils import Coil, CoilSet
from bluemira.equilibria.equilibrium import Equilibrium, Breakdown
from bluemira.equilibria.constraints import AutoConstraints
from bluemira.equilibria.profiles import CustomProfile, BetaIpProfile, DoublePowerFunc
from bluemira.equilibria.optimiser import FBIOptimiser, BreakdownOptimiser
from bluemira.equilibria.physics import calc_psib
from bluemira.equilibria.solve import PicardLiAbsIterator, PicardAbsIterator

# %%[markdown]

# Load the reference equilibria from EFDA_D_2MUW9R

# %%

plot_defaults()

PATH = get_bluemira_path("eqdsk/EUROfusion_DEMO_2017_equilibria", subfolder="data")
SOF = "Equil_2017_PMI_baseline_SOF_150Vs_final_newFW.eqdsk"
EOF = "Equil_2017_PMI_baseline_EOF_final_newFW.eqdsk"
sof_filename = os.sep.join([PATH, SOF])
eof_filename = os.sep.join([PATH, EOF])

reader = EQDSKInterface()

sof_dict = reader.read(sof_filename)
eof_dict = reader.read(eof_filename)

# %%[markdown]

# Make the same CoilSet as CREATE

# %%
x = [5.4, 14, 17.75, 17.75, 14.0, 7, 2.77, 2.77, 2.77, 2.77, 2.77]
z = [9.26, 7.9, 2.5, -2.5, -7.9, -10.5, 7.07, 4.08, -0.4, -4.88, -7.86]
dx = [0.6, 0.7, 0.5, 0.5, 0.7, 1.0, 0.4, 0.4, 0.4, 0.4, 0.4]
dz = [0.6, 0.7, 0.5, 0.5, 0.7, 1.0, 2.99 / 2, 2.99 / 2, 5.97 / 2, 2.99 / 2, 2.99 / 2]

coils = []
for i, (xi, zi, dxi, dzi) in enumerate(zip(x, z, dx, dz)):
    ctype = "PF" if i < 6 else "CS"
    coil = Coil(xi, zi, current=0, dx=dxi, dz=dzi, ctype=ctype, name=f"{ctype}_{i+1}", control=True)
    coils.append(coil)

coilset = CoilSet(coils)

# Assign current density and peak field constraints
coilset.assign_coil_materials("CS", "Nb3Sn")
coilset.assign_coil_materials("PF", "NbTi")
coilset.fix_sizes()
coilset.mesh_coils(0.4)

coilset.plot()

# %%[markdown]

# Define parameters

# %%

# Machine parameters
I_p = 19.07e6  # A
beta_p = 1.141
l_i = 0.8
R_0 = 8.938
Z_0 = 0.027454
B_0 = 4.8901  # ???
A = 3.1
kappa_95 = 1.65
delta_95 = 0.33
tau_flattop = 2 * 3600
v_burn = 4.220e-2  # V
c_ejima = 0.3

# Breakdown constraints (I can't quite get it with 3mT.. this was the closest I get to 310)
# This is quite a sensitive optimisation, and is possibly a multi-modal space
# May want to think about optimising with a stochastic optimiser, and including
# a parametric location of the breakdown point...
x_zone = 9.6  # ??
z_zone = 0.0  # ??
r_zone = 1.0  # ??
b_zone_max = 0.003  # T

# Coil constraints
PF_Fz_max = 450e6
CS_Fz_sum = 300e6
CS_Fz_sep = 350e6

# %%[markdown]
# Use the same grid as CREATE (but less discretised):

# %%

sof_dict["nx"] = 100
sof_dict["nz"] = 100
grid = Grid.from_eqdict(sof_dict)

# %%[markdown]

# Set up the Breakdown object

# %%

max_currents = coilset.get_max_currents(0)
coilset.set_control_currents(max_currents, update_size=False)


breakdown = Breakdown(coilset.copy(), grid, R_0=R_0)
breakdown.set_breakdown_point(x_zone, z_zone)

optimiser = BreakdownOptimiser(
    x_zone,
    z_zone,
    r_zone,
    b_zone_max,
    max_currents,
    coilset.get_max_fields(),
    PF_Fz_max=PF_Fz_max,
    CS_Fz_sum=CS_Fz_sum,
    CS_Fz_sep=CS_Fz_sep,
)

currents = optimiser(breakdown)
breakdown.coilset.set_control_currents(currents)

bluemira_print(f"Breakdown psi: {breakdown.breakdown_psi*2*np.pi:.2f} V.s")

# %%[markdown]

# Calculate SOF and EOF plasma boundary fluxes

# %%
psi_sof = calc_psib(breakdown.breakdown_psi * 2 * np.pi, R_0, I_p, l_i, c_ejima)
psi_eof = psi_sof - tau_flattop * v_burn

# CREATE then knocked off an extra 10 V.s for misc plasma stuff I didnt look into

psi_sof -= 10
psi_eof -= 10

# %%[markdown]

# Set up a parameterised profile

# %%
shape = DoublePowerFunc([2, 3])
profile = BetaIpProfile(beta_p * 1.2, I_p, R_0, B_0, shape=shape)


# %%[markdown]
# Solve the SOF and EOF equilibria

# %%

sof = Equilibrium(
    coilset.copy(),
    grid,
    Ip=I_p / 1e6,
    li=l_i,
    profiles=None,
    RB0=[R_0, B_0],
)
eof = Equilibrium(
    coilset.copy(),
    grid,
    Ip=I_p / 1e6,
    li=l_i,
    profiles=None,
    RB0=[R_0, B_0],
)

# Make a set of magnetic constraints for the equilibria... I got lazy here,
# this is just:
#   * LCFS boundary fluxes
#   * Field null at lower X-point
#   * divertor legs are not treated, but could easily be added

sof_constraints = AutoConstraints(
    sof_dict["xbdry"], sof_dict["zbdry"], psi_sof / 2 / np.pi, n=100
)
eof_constraints = AutoConstraints(
    eof_dict["xbdry"], eof_dict["zbdry"], psi_eof / 2 / np.pi, n=100
)


optimiser = FBIOptimiser(
    coilset.get_max_fields(),
    PF_Fz_max,
    CS_Fz_sum,
    CS_Fz_sep,
)
optimiser.update_current_constraint(coilset.get_max_currents(0))

solver = PicardLiAbsIterator

iterator = solver(sof, profile, sof_constraints, optimiser, plot=True)
iterator()


iterator = solver(eof, profile, eof_constraints, optimiser, plot=False)
iterator()

# %%[markdown]
# Plot the results

# %%
f, ax = plt.subplots(1, 3)
breakdown.plot(ax[0])
breakdown.coilset.plot(ax[0])
sof.plot(ax[1])
sof.coilset.plot(ax[1])
eof.plot(ax[2])
eof.coilset.plot(ax[2])

sof_psi = 2 * np.pi * sof.psi(*sof._x_points[0][:2])[0][0]
eof_psi = 2 * np.pi * eof.psi(*eof._x_points[0][:2])[0][0]
ax[1].set_title("$\\psi_{b}$ = " + f"{sof_psi:.2f} V.s")
ax[2].set_title("$\\psi_{b}$ = " + f"{eof_psi:.2f} V.s")


bluemira_print("SOF:\n" f"beta_p: {sof.calc_beta_p():.2f}\n" f"l_i: {sof.calc_li():.2f}")


bluemira_print("EOF:\n" f"beta_p: {eof.calc_beta_p():.2f}\n" f"l_i: {eof.calc_li():.2f}")


# %%[markdown]

# Can also fit a Johner parameterisation to the CREATE separatrix..

# %%

from BLUEPRINT.equilibria.shapes import flux_surface_johner
from BLUEPRINT.geometry.loop import Loop

sep_loop = Loop(x=sof_dict["xbdry"], z=sof_dict["zbdry"])
sep_loop.close()
sep_loop.interpolate(150)
sep_loop.sort_bottom()

z_min = min(sep_loop.z)
arg_min = np.argmin(sep_loop.z)
z_max = max(sep_loop.z)
arg_max = np.argmax(sep_loop.z)
delta_z_bot = Z_0 - z_min
delta_z_top = z_max - Z_0
delta_x_bot = R_0 - sep_loop.x[arg_min]
delta_x_top = R_0 - sep_loop.x[arg_max]
a = R_0 / A
kappa_l = delta_z_bot / a
kappa_u = delta_z_top / a
delta_l = delta_x_bot / a
delta_u = delta_x_top / a


def fitter(x):
    a1, a2, a3, a4 = x
    a2 = abs(a1 - 190)
    a4 = 180 - a3
    loop = flux_surface_johner(
        R_0, Z_0, a, kappa_u, kappa_l, delta_u, delta_l, a1, a2, a3, a4, n=150
    )
    loop.close()
    loop.interpolate(150)
    loop.sort_bottom()
    return np.sum(np.sqrt((sep_loop.x - loop.x) ** 2 + (sep_loop.z - loop.z) ** 2))


import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

result = differential_evolution(
    fitter,  # x0=np.array([190, 10, -120, 30]),
    bounds=[[160, 190], [5, 20], [-120, -120], [20, 33]],
    #'method="SLSQP", options={"eps": .5}
)
new = flux_surface_johner(R_0, 0, R_0 / A, kappa_u, kappa_l, delta_u, delta_l, *result.x)
f, ax = plt.subplots()
ax.plot(sof_dict["xbdry"], sof_dict["zbdry"], color="g")
new.plot(fill=False, edgecolor="r")
