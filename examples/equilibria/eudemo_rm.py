# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2022 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
Attempt at recreating the EU-DEMO 2017 reference equilibria from a known coilset.
"""

# %%[markdown]

# # EU-DEMO 2017 reference breakdown and equilibrium benchmark

# with
# - 300mw out
# - 2 hr flat top

# %%
import json
import os
import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython import get_ipython

from bluemira.base.constants import raw_uc
from bluemira.base.file import get_bluemira_path
from bluemira.base.look_and_feel import bluemira_print
from bluemira.display import plot_2d, plot_defaults
from bluemira.equilibria.coils import Coil, CoilSet
from bluemira.equilibria.equilibrium import Breakdown, Equilibrium
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.opt_constraints import (
    CoilFieldConstraints,
    CoilForceConstraints,
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
    PsiBoundaryConstraint,
)
from bluemira.equilibria.opt_problems import (
    BreakdownCOP,
    MinimalCurrentCOP,
    OutboardBreakdownZoneStrategy,
    UnconstrainedTikhonovCurrentGradientCOP,
)
from bluemira.equilibria.physics import calc_psib
from bluemira.equilibria.profiles import CustomProfile
from bluemira.equilibria.run import PulsedNestedPositionCOP
from bluemira.equilibria.solve import PicardIterator
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import make_polygon, offset_wire, slice_shape, split_wire
from bluemira.utilities.optimiser import Optimiser
from bluemira.utilities.positioning import PathInterpolator, PositionMapper
from eudemo.pf_coils.tools import make_coil_mapper, make_pf_coil_path

# %%[markdown]

# Load the reference equilibria from EFDA_D_2MUW9R

# %%

plot_defaults()

try:
    get_ipython().run_line_magic("matplotlib", "qt")
except AttributeError:
    pass

path = get_bluemira_path("equilibria", subfolder="examples")
name = "EUDEMO_2017_CREATE_SOF_separatrix.json"
filename = os.sep.join([path, name])
with open(filename, "r") as file:
    data = json.load(file)

sof_xbdry = data["xbdry"]
sof_zbdry = data["zbdry"]


# %%[markdown]

# Import keep out zones

# %%
xl_dict = pd.read_excel(sys.argv[1], None)

koz_LP_k = koz_UP_k = TF_inner_k = TF_outer_k = None
for key in xl_dict.keys():
    key_low = key.lower()
    if "keep" in key_low:
        if "lp" in key_low:
            koz_LP_k = key
        elif "up" in key_low:
            koz_UP_k = key
    if "tf" in key_low:
        if "inner" in key_low:
            TF_inner_k = key
        elif "outer" in key_low:
            TF_outer_k = key


def get_unit(column_name):
    try:
        return column_name.split("(")[1].strip(") ")
    except IndexError:
        return "mm"


# Asummes the units are the same for both columns
koz_LP_raw = xl_dict[koz_LP_k]
koz_UP_raw = xl_dict[koz_UP_k]

TF_inner_raw = xl_dict[TF_inner_k]
TF_outer_raw = xl_dict[TF_outer_k]

koz_LP = raw_uc(
    koz_LP_raw.to_numpy()[:, :2].astype(float), get_unit(koz_LP_raw.columns[0]), "m"
).T

koz_UP = raw_uc(
    koz_UP_raw.to_numpy()[:, :2].astype(float), get_unit(koz_UP_raw.columns[0]), "m"
).T

TF_inner = raw_uc(
    TF_inner_raw.to_numpy()[:, :2].astype(float), get_unit(TF_inner_raw.columns[0]), "m"
).T

TF_outer = raw_uc(
    TF_outer_raw.to_numpy()[:, :2].astype(float), get_unit(TF_outer_raw.columns[0]), "m"
).T

# %%[markdown]

# Make the same CoilSet as CREATE

# %%
x = [5.4, 14, 17.75, 17.75, 14.0, 7, 2.77, 2.77, 2.77, 2.77, 2.77]
z = [9.26, 7.9, 2.5, -2.5, -7.9, -10.5, 7.07, 4.08, -0.4, -4.88, -7.86]
dx = [0.6, 0.7, 0.5, 0.5, 0.7, 1.0, 0.4, 0.4, 0.4, 0.4, 0.4]
dz = [0.6, 0.7, 0.5, 0.5, 0.7, 1.0, 2.99 / 2, 2.99 / 2, 5.97 / 2, 2.99 / 2, 2.99 / 2]

# # crude movement of coils
# x[0] = min(x[0], min(koz_UP[0])) - dx[0]
# x[1] = max(x[1], max(koz_UP[0])) + dx[1]
# x[4] = x[4] + 2 * dx[4]
# x[5] = min(x[5], min(koz_LP[0]))

# z[1] = z[1] - dz[1]
# z[4] = z[4] + 2 * dz[4]

# # Matti's positions
# x[:6] = np.array([4, 14.54, 17.75, 17.75, 15.4, 7.0])
# z[:6] = np.array([9.26, 7.25, 2.5, -2.5, -6.55, -10.5])

# create coilset
coils = []
j = 1
for i, (xi, zi, dxi, dzi) in enumerate(zip(x, z, dx, dz)):
    if j > 6:
        j = 1
    ctype = "PF" if i < 6 else "CS"
    coil = Coil(
        xi,
        zi,
        current=0,
        dx=dxi,
        dz=dzi,
        ctype=ctype,
        name=f"{ctype}_{j}",
    )
    coils.append(coil)
    j += 1

coilset = CoilSet(*coils)

# Assign current density and peak field constraints
coilset.assign_material("CS", j_max=16.5e6, b_max=13)
coilset.assign_material("PF", j_max=12.5e6, b_max=11)
coilset.fix_sizes()
coilset.discretisation = 0.3

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
tau_flattop = 2 * 3600  # 2hrs
v_burn = 4.220e-2  # V
c_ejima = 0.3

# Breakdown constraints (I can't quite get it with 3mT..) I've gotten close to 305 V.s,
# but only using a smaller low-field region.
# This is quite a sensitive optimisation, and is possibly a multi-modal space
# May want to think about optimising with a stochastic optimiser, and including
# a parametric location of the breakdown point...
x_zone = 9.84  # ??
z_zone = 0.0  # ??
r_zone = 2.0  # ??
b_zone_max = 0.003  # T

# Coil constraints
PF_Fz_max = 450
CS_Fz_sum = 300
CS_Fz_sep = 350

# %%[markdown]
# Use the same grid as CREATE (but less discretised):

# %%

grid = Grid(2, 16.0, -9.0, 9.0, 100, 100)

# %%[markdown]

# Set up the Breakdown object

# %%

field_constraints = CoilFieldConstraints(coilset, coilset.b_max, tolerance=1e-6)
force_constraints = CoilForceConstraints(
    coilset, PF_Fz_max, CS_Fz_sum, CS_Fz_sep, tolerance=1e-6
)

max_currents = coilset.get_max_current(0)
max_CS_currents = coilset.get_coiltype("CS").get_max_current(0)
coilset.get_coiltype("CS").current = max_CS_currents


breakdown = Breakdown(deepcopy(coilset), grid)

bd_opt_problem = BreakdownCOP(
    breakdown.coilset,
    breakdown,
    OutboardBreakdownZoneStrategy(R_0, A, 0.225),
    optimiser=Optimiser("COBYLA", opt_conditions={"max_eval": 3000, "ftol_rel": 1e-6}),
    max_currents=max_currents,
    B_stray_max=1e-3,
    B_stray_con_tol=1e-6,
    n_B_stray_points=10,
    constraints=[
        field_constraints,
        force_constraints,
    ],
)

coilset = bd_opt_problem.optimise(max_currents)
bluemira_print(f"Breakdown psi: {breakdown.breakdown_psi*2*np.pi:.2f} V.s")

# force breakdown flux to 320 Vs
breakdown_flux = 320  # breakdown.breakdown_psi*2*np.pi

# %%[markdown]

# Calculate SOF and EOF plasma boundary fluxes

# %%
psi_sof = calc_psib(breakdown_flux, R_0, I_p, l_i, c_ejima)
psi_eof = psi_sof - tau_flattop * v_burn

# CREATE then knocked off an extra 10 V.s for misc plasma stuff I didnt look into

psi_sof -= 10
psi_eof -= 10

# %%[markdown]

# Set up a parameterised profile

# %%
profiles = CustomProfile(
    np.array([86856, 86506, 84731, 80784, 74159, 64576, 52030, 36918, 20314, 4807, 0.0]),
    -np.array(
        [0.125, 0.124, 0.122, 0.116, 0.106, 0.093, 0.074, 0.053, 0.029, 0.007, 0.0]
    ),
    R_0=R_0,
    B_0=B_0,
    I_p=I_p,
)
# profile = BetaIpProfile(beta_p, I_p, R_0, B_0, shape=shape)


# %%[markdown]
# Solve the SOF and EOF equilibria

# %%

reference_eq = Equilibrium(
    deepcopy(coilset),
    grid,
    profiles,
)

# Make a set of magnetic constraints for the equilibria... I got lazy here,
# this is just:
#   * LCFS boundary fluxes
#   * Field null at lower X-point
#   * divertor legs are not treated, but could easily be added

isoflux = IsofluxConstraint(
    np.array(sof_xbdry)[::10],
    np.array(sof_zbdry)[::10],
    sof_xbdry[0],
    sof_zbdry[0],
    tolerance=1e-3,
    constraint_value=0.25,  # Difficult to choose...
)
xp_idx = np.argmin(sof_zbdry)
x_point = FieldNullConstraint(
    sof_xbdry[xp_idx],
    sof_zbdry[xp_idx],
    tolerance=1e-3,
)

ref_opt_problem = UnconstrainedTikhonovCurrentGradientCOP(
    reference_eq.coilset,
    reference_eq,
    MagneticConstraintSet([isoflux, x_point]),
    gamma=1e-12,
)

program = PicardIterator(reference_eq, ref_opt_problem, fixed_coils=True, relaxation=0.2)
program()

opt_problems = []
eqs = []
for psi in [psi_sof, psi_eof]:
    optimiser = Optimiser("SLSQP", opt_conditions={"max_eval": 1000, "ftol_rel": 1e-6})

    psi_boundary = PsiBoundaryConstraint(
        np.array(sof_xbdry)[::5],
        np.array(sof_zbdry)[::5],
        psi / (2 * np.pi),
        tolerance=np.sqrt(1.5 * len(sof_xbdry[::5])),
    )

    ft_eq = deepcopy(reference_eq)

    for name in ft_eq.coilset.get_coiltype("PF").name:
        ft_eq.coilset[name]._flag_sizefix = False

    eqs.append(ft_eq)
    opt_problems.append(
        MinimalCurrentCOP(
            ft_eq.coilset,
            ft_eq,
            optimiser=optimiser,
            max_currents=max_currents,
            constraints=[psi_boundary, x_point, deepcopy(field_constraints)],
        )
    )


# %%

# We'll store these so that we can look at them again later
old_coilset = deepcopy(coilset)
old_eq = deepcopy(reference_eq)

offset_val = np.max([dx[:6], dz[:6]])
tf_outer = TF_outer[:, np.where(TF_outer[1] == np.max(TF_outer[1]))[0][0] :]
tf_outer = tf_outer[:, : np.where(tf_outer[0] == np.min(tf_outer[0]))[0][2]]
t_outer = np.array([tf_outer[0], np.zeros(len(tf_outer[0])), tf_outer[1]])

face_koz_UP = BluemiraFace(
    make_polygon(np.array([koz_UP[0], np.zeros_like(koz_UP[0]), koz_UP[1]]))
)
face_koz_LP = BluemiraFace(
    make_polygon(np.array([koz_LP[0], np.zeros_like(koz_LP[0]), koz_LP[1]]))
)

from bluemira.geometry.coordinates import Coordinates

pf_coil_path = make_pf_coil_path(
    make_polygon(Coordinates({"x": TF_outer[0], "z": TF_outer[1]})), offset_val
)

position_mapper = make_coil_mapper(
    pf_coil_path,
    [face_koz_UP, face_koz_LP],
    coilset.get_coiltype("PF")._coils,
)

# %%

position_opt_problem = PulsedNestedPositionCOP(
    coilset,
    position_mapper,
    sub_opt_problems=opt_problems,
    optimiser=Optimiser("COBYLA", opt_conditions={"max_eval": 200, "ftol_rel": 1e-6}),
    debug=False,
)
optimised_coilset = position_opt_problem.optimise(verbose=True)

# %%


max_pf_currents = np.max(
    np.abs([eq.coilset.get_coiltype("PF").get_control_coils().current for eq in eqs]),
    axis=0,
)

pf_coil_names = optimised_coilset.get_coiltype("PF").name

max_cs_currents = optimised_coilset.get_coiltype("CS").get_max_current()

max_currents = np.concatenate([max_pf_currents, max_cs_currents])

for problem in opt_problems:
    for pf_name, max_current in zip(pf_coil_names, max_pf_currents):
        problem.eq.coilset[pf_name].resize(max_current)
        problem.eq.coilset[pf_name].fix_size()
        problem.eq.coilset[pf_name].discretisation = 0.3
    problem.set_current_bounds(max_currents)

# %%
for eq, problem in zip(eqs, opt_problems):
    PicardIterator(eq, problem, plot=True, relaxation=0.2, fixed_coils=True)()

# %%[markdown]
# Plot the results

# %%

plt.close("all")
f, ax = plt.subplots(1, 3)

for name, _ax, eq in zip(["Breakdown", "SOF", "EOF"], ax, [breakdown] + eqs):
    eq.plot(_ax)
    eq.coilset.plot(_ax, label=True)

    if isinstance(eq, Equilibrium):
        psi = 2 * np.pi * eq.get_OX_psis()[1]
    else:
        psi = 2 * np.pi * eq.breakdown_psi

    _ax.set_title(f"{name}" " $\\psi_{b}$ = " + f"{psi:.2f} V.s")
plt.show()
