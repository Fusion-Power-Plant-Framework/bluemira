# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Test MAST-U.
"""

# %%
import pickle
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np

from bluemira.display.auto_config import plot_defaults
from bluemira.equilibria.coils import (
    Circuit,
    Coil,
    CoilSet,
    symmetrise_coilset,
)
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.optimisation.constraints import (
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
)
from bluemira.equilibria.optimisation.problem import (
    TikhonovCurrentCOP,
)
from bluemira.equilibria.profiles import BetaIpProfile, DoublePowerFunc
from bluemira.equilibria.solve import (
    DudsonConvergence,
    PicardIterator,
)

plot_defaults()

# %%
# Get MAST-U coils used by FreeGSNK and covert to
# BM coilset of circuits.
with open("MAST-U_like_active_coils.pickle", "rb") as f:
    coil_dict = pickle.load(f)

circuits = []
count = []
for n, d in coil_dict.items():
    coils = []
    if n == "Solenoid":
        i = 0
        count.append(len(d["R"]))
        for r, z in zip(d["R"], d["Z"], strict=False):
            coil = Coil(
                r,
                z,
                current=5000,
                dx=d["dR"] / 2,
                dz=d["dZ"] / 2,
                ctype="CS",
                name=n + "_" + str(i),
            )
            coils.append(coil)
            i += 1
    else:
        i = 0
        count.append(len(d["1"]["R"]))
        for r1, z1, r2, z2 in zip(
            d["1"]["R"],
            d["1"]["Z"],
            d["2"]["R"],
            d["2"]["Z"],
            strict=False,
        ):
            coil_up = Coil(
                r1,
                z1,
                current=0,
                dx=d["1"]["dR"] / 2,
                dz=d["1"]["dZ"] / 2,
                ctype="PF",
                name=n + "U_" + str(i),
            )
            coil_low = Coil(
                r2,
                z2,
                current=0,
                dx=d["2"]["dR"] / 2,
                dz=d["2"]["dZ"] / 2,
                ctype="PF",
                name=n + "L_" + str(i),
            )
            coils.append(coil_up)
            coils.append(coil_low)
            i += 1
    circuit = Circuit(*coils)
    circuits.append(circuit)
full_coilset = CoilSet(*circuits)

full_coilset.control = [n for n in coilset.name if "Solenoid" not in n]
full_coilset.plot()
plt.show()

# %%
# How many BM coils are there?
np.sum(count[1:]) * 2 + count[0]

# %%
# Create BM Coilset of single coils
masty = {
    "xc": [
        0.19475,
        0.24849975,
        0.42625037,
        0.60125023,
        0.8432501,
        0.96224999,
        1.92205048,
        1.32175004,
        1.55675006,
        1.56500018,
        1.71500003,
        1.35444999,
        0.24849975,
        0.42625037,
        0.60125023,
        0.8432501,
        0.96224999,
        1.92205048,
        1.32175004,
        1.55675006,
        1.56500018,
        1.71500003,
        1.35444999,
    ],
    "zc": [
        0.0,
        1.22402805,
        1.57249993,
        1.735000015,
        1.982000055,
        1.4936999649999998,
        1.9499999849999998,
        1.467700005,
        1.4676999450000001,
        1.0956499,
        0.352150035,
        0.943414985,
        -1.22402805,
        -1.57249993,
        -1.735000015,
        -1.982000055,
        -1.4936999649999998,
        -1.9499999849999998,
        -1.467700005,
        -1.4676999450000001,
        -1.0956499,
        -0.352150035,
        -0.943414985,
    ],
    "dxc": [
        0.0,
        0.007000,
        0.036750,
        0.036750,
        0.036750,
        0.036750,
        0.022050,
        0.036750,
        0.036750,
        0.065000,
        0.065000,
        0.036750,
        0.007000,
        0.036750,
        0.036750,
        0.036750,
        0.036750,
        0.022050,
        0.036750,
        0.036750,
        0.065000,
        0.065000,
        0.036750,
    ],
    "dzc": [
        1.581,
        0.192378,
        0.036750,
        0.022050,
        0.022050,
        0.022050,
        0.044100,
        0.022050,
        0.022050,
        0.058500,
        0.058500,
        0.052750,
        0.192378,
        0.036750,
        0.022050,
        0.022050,
        0.022050,
        0.044100,
        0.022050,
        0.022050,
        0.058500,
        0.058500,
        0.052750,
    ],
    "coil_names": [
        "Solenoid",
        "PXU",
        "D1U",
        "D2U",
        "D3U",
        "DpU",
        "D5U",
        "D6U",
        "D7U",
        "P4U",
        "P5U",
        "P6U",
        "PXL",
        "D1L",
        "D2L",
        "D3L",
        "DpL",
        "D5L",
        "D6L",
        "D7L",
        "P4L",
        "P5L",
        "P6L",
    ],
    "coil_types": [
        "CS",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
        "PF",
    ],
    "n_turns": [
        234,
        42,
        35,
        23,
        23,
        23,
        24,
        27,
        23,
        23,
        23,
        23,
        42,
        35,
        23,
        23,
        23,
        24,
        27,
        23,
        23,
        23,
        23,
    ],
}

coils = []
for (
    xi,
    zi,
    dxi,
    dzi,
    typei,
    namei,
) in zip(  # turni,
    masty["xc"],
    masty["zc"],
    masty["dxc"],
    masty["dzc"],
    masty["coil_types"],
    masty["coil_names"],
    # masty["n_turns"],
    strict=False,
):
    coil = Coil(
        xi,
        zi,
        current=0,
        dx=dxi,
        dz=dzi,
        ctype=typei,
        name=namei,
        # discretisation=0.01, #0.05, # This is the minimum allowed in BM
        # n_turns=turni,
    )
    coils.append(coil)
masty_coilset = CoilSet(*coils)


# Set current value in solenoid to match FreeGSNKE and remove from control coils
masty_coilset["Solenoid"].current = 5000
masty_coilset = symmetrise_coilset(masty_coilset)
masty_coilset.control = [
    "PXU",
    "D1U",
    "D2U",
    "D3U",
    "DpU",
    "D5U",
    "D6U",
    "D7U",
    "P4U",
    "P5U",
    "P6U",
    "PXL",
    "D1L",
    "D2L",
    "D3L",
    "DpL",
    "D5L",
    "D6L",
    "D7L",
    "P4L",
    "P5L",
    "P6L",
]

masty_coilset.plot()
plt.show()

# %%
# Grid
grid = Grid(0.1, 2.0, -2.2, 2.2, 65, 129)

# Profile params
I_p = 6e5  # A
R_0 = 0.85  # m
A = 1.3
B_0 = 0.588  # T, in FreeGSNKE fvac = R * B_0 = 0.5
betap = 0.3

# Use DoublePowerFunc to match FreeGSNKE
profiles = BetaIpProfile(
    betap=betap,
    I_p=I_p,
    R_0=R_0,
    B_0=B_0,
    shape=DoublePowerFunc([1.8, 1.2]),  # FreeGSNKE: alpha_m=1.8, alpha_n=1.2
)

# Equilibrium wioth single filament coils
masty_eq = Equilibrium(masty_coilset, grid, profiles, psi=None)

# %%
# Constraints

# x-point constraints
x_xp = 0.6
z_xp = 1.1
x_point_u = FieldNullConstraint(x_xp, z_xp, tolerance=1e-6)  # FreeGSNKE target tolerence
x_point_l = FieldNullConstraint(x_xp, -z_xp, tolerance=1e-6)

# # isoflux constraints
# x_omp = 1.4
# x_imp = 0.35
# x_ref_iso = x_imp
# z_ref_iso = 0.0
# x_iso = [
#     x_omp, #x_xp, x_xp,
#     1.2, 1.2, 0.45, 0.45,
#     0.85, 0.85, 0.75, 0.75,
#     x_imp, x_imp, x_imp, x_imp,
#     # Add extra
#     #1.33, 1.33, 0.85, 0.85

# ]
# z_iso = [
#     0.0, #z_xp, -z_xp,
#     0.7, -0.7, 1.8, -1.8,
#     1.7, -1.7, 1.6, -1.6,
#     0.2, -0.2, 0.1, -0.1,
#     # Add extra
#     #0.4, -0.4, 1.0, 1.0
# ]

# isoflux constraints
# set up in pairs (psi_p2 - psi_p1)
x_omp = 1.4
x_imp = 0.35
z_imp = 0.0
z_omp = 0.0

# x_p2 = [
#     x_xp, x_imp, x_omp,
#     1.2, 1.2,
#     0.85, 0.75, x_imp, x_imp, x_imp,
#     x_imp, 0.85, 0.75, 0.45, 0.45
# ]
# z_p2 = [
#     z_xp, z_imp, z_omp,
#     0.7, -0.7,
#     1.7, 1.6, 0.2, 0.1, -0.1,
#     -0.2, -1.7, -1.6, -1.8, 1.8
# ]

# x_p1 = [
#     x_xp, x_omp, x_xp,
#     x_omp, x_omp,
#     x_xp, x_xp, x_xp, x_xp, x_xp,
#     x_xp, x_xp, x_xp, x_xp, x_xp
# ]
# z_p1 = [
#     -z_xp, z_omp, z_xp,
#     z_omp, z_omp,
#     z_xp, z_xp, z_xp, z_xp, z_xp,
#     z_xp, z_xp, z_xp, z_xp, z_xp
# ]

x_p2 = [
    x_xp,
    x_imp,
    x_omp,
    1.2,
    1.2,
    0.85,
    0.75,
    x_imp,
    x_imp,
    x_imp,
    x_imp,
    0.85,
    0.75,
    0.45,
    0.45,
]
z_p2 = [
    -z_xp,
    z_imp,
    z_omp,
    0.7,
    -0.7,
    1.7,
    1.6,
    0.2,
    0.1,
    -0.1,
    -0.2,
    -1.7,
    -1.6,
    -1.8,
    1.8,
]

x_p1 = x_xp
z_p1 = z_xp

isoflux = IsofluxConstraint(
    x_p2,
    z_p2,
    x_p1,
    z_p1,
    tolerance=1e-6,
    constraint_value=0.0,
)

isoflux_set = MagneticConstraintSet([isoflux])
constraints_list = [isoflux, x_point_u, x_point_l]
constraints_set = MagneticConstraintSet(constraints_list)

f, ax = plt.subplots()
masty_eq.plot(ax=ax)
constraints_set.plot(ax=ax)
masty_eq.coilset.plot(ax=ax)
plt.show()

# %%
# # %%time
# unconstrained_eq = deepcopy(masty_eq)
# current_opt_problem = UnconstrainedTikhonovCurrentGradientCOP(
#     unconstrained_eq,
#     constraints_set_hi_tol,
#     gamma=5e-7 # Gamma to match FreeGSNKE
# )

# program = PicardIterator(
#     unconstrained_eq,
#     current_opt_problem,
#     fixed_coils=True,
#     convergence=DudsonConvergence(1e-3, 1e-10), # Matches FreeGSNKE rtol=1e-3, but they also have an or atol condition - psi_maxchange < 1e-10
#     relaxation=0.0, # FreeGSNKE blend = 0.0
#     maxiter=50,
# )

# program()

# f, ax = plt.subplots()
# unconstrained_eq.plot(ax=ax)
# constraints_set.plot(ax=ax)
# coilset.plot(ax=ax)
# plt.show()

# %%
# %%time
# constrained_eq = deepcopy(unconstrained_eq)
constrained_eq = deepcopy(masty_eq)
# FreeGSNKE uses least squares problem with Tikhonov regularisation term
# I think that SLSQP is aapropriate algo, constrained least squares
current_opt_problem = TikhonovCurrentCOP(
    constrained_eq,
    targets=constraints_set,
    gamma=5e-7,
    opt_algorithm="SLSQP",
    opt_conditions={"max_eval": 2000, "ftol_rel": 1e-6},
    opt_parameters={},
    max_currents=1e6,
    constraints=constraints_list,
)

program = PicardIterator(
    constrained_eq,
    current_opt_problem,
    fixed_coils=True,
    # Matches FreeGSNKE rtol=1e-3,
    # but they also have an or atol condition: psi_maxchange < 1e-10.
    convergence=DudsonConvergence(1.0e-3, 1e-10),
    relaxation=0.0,  # FreeGSNKE blend = 0.0
    maxiter=30,
)
_ = program()
f, ax = plt.subplots()
constrained_eq.plot(ax=ax)
constraints_set.plot(ax=ax)
constrained_eq.coilset.plot(ax=ax)
plt.show()

# %%
# Try setting currents using previous opt
full_coilset._opt_currents = constrained_eq.coilset.current[1::2] / count[1:]

# Grid
grid = Grid(0.1, 2.0, -2.2, 2.2, 65, 129)

# Profile params
I_p = 6e5  # A
R_0 = 0.85  # m
A = 1.3
B_0 = 0.588  # T, in FreeGSNKE fvac = R * B_0 = 0.5
betap = 0.3

# Use DoublePowerFunc to match FreeGSNKE
profiles = BetaIpProfile(
    betap=betap,
    I_p=I_p,
    R_0=R_0,
    B_0=B_0,
    shape=DoublePowerFunc([1.8, 1.2]),  # FreeGSNKE: alpha_m=1.8, alpha_n=1.2
)

# Equilibrium
eq = Equilibrium(full_coilset, grid, profiles, psi=None)

# %%
# Constraints

# x-point constraints
x_xp = 0.6
z_xp = 1.1
x_point_u = FieldNullConstraint(x_xp, z_xp, tolerance=1e-6)  # FreeGSNKE target tolerence
x_point_l = FieldNullConstraint(x_xp, -z_xp, tolerance=1e-6)

# # isoflux constraints
# x_omp = 1.4
# x_imp = 0.35
# x_ref_iso = x_imp
# z_ref_iso = 0.0
# x_iso = [
#     x_omp, #x_xp, x_xp,
#     1.2, 1.2, 0.45, 0.45,
#     0.85, 0.85, 0.75, 0.75,
#     x_imp, x_imp, x_imp, x_imp,
#     # Add extra
#     #1.33, 1.33, 0.85, 0.85

# ]
# z_iso = [
#     0.0, #z_xp, -z_xp,
#     0.7, -0.7, 1.8, -1.8,
#     1.7, -1.7, 1.6, -1.6,
#     0.2, -0.2, 0.1, -0.1,
#     # Add extra
#     #0.4, -0.4, 1.0, 1.0
# ]

# isoflux constraints
# set up in pairs (psi_p2 - psi_p1)
x_omp = 1.4
x_imp = 0.35
z_imp = 0.0
z_omp = 0.0

# x_p2 = [
#     x_xp, x_imp, x_omp,
#     1.2, 1.2,
#     0.85, 0.75, x_imp, x_imp, x_imp,
#     x_imp, 0.85, 0.75, 0.45, 0.45
# ]
# z_p2 = [
#     z_xp, z_imp, z_omp,
#     0.7, -0.7,
#     1.7, 1.6, 0.2, 0.1, -0.1,
#     -0.2, -1.7, -1.6, -1.8, 1.8
# ]

# x_p1 = [
#     x_xp, x_omp, x_xp,
#     x_omp, x_omp,
#     x_xp, x_xp, x_xp, x_xp, x_xp,
#     x_xp, x_xp, x_xp, x_xp, x_xp
# ]
# z_p1 = [
#     -z_xp, z_omp, z_xp,
#     z_omp, z_omp,
#     z_xp, z_xp, z_xp, z_xp, z_xp,
#     z_xp, z_xp, z_xp, z_xp, z_xp
# ]

x_p2 = [
    x_xp,
    x_imp,
    x_omp,
    1.2,
    1.2,
    0.85,
    0.75,
    x_imp,
    x_imp,
    x_imp,
    x_imp,
    0.85,
    0.75,
    0.45,
    0.45,
]
z_p2 = [
    -z_xp,
    z_imp,
    z_omp,
    0.7,
    -0.7,
    1.7,
    1.6,
    0.2,
    0.1,
    -0.1,
    -0.2,
    -1.7,
    -1.6,
    -1.8,
    1.8,
]

x_p1 = x_xp
z_p1 = z_xp

isoflux = IsofluxConstraint(
    x_p2,
    z_p2,
    x_p1,
    z_p1,
    tolerance=1e-6,
    constraint_value=0.0,
)

isoflux_set = MagneticConstraintSet([isoflux])
constraints_list = [isoflux, x_point_u, x_point_l]
constraints_set = MagneticConstraintSet(constraints_list)

f, ax = plt.subplots()
eq.plot(ax=ax)
constraints_set.plot(ax=ax)
eq.coilset.plot(ax=ax)
plt.show()

# %%
eq.coilset._opt_currents

# %%
# # %%time
# unconstrained_eq = deepcopy(eq)
# current_opt_problem = UnconstrainedTikhonovCurrentGradientCOP(
#     unconstrained_eq,
#     constraints_set,
#     gamma=5e-7 # Gamma to match FreeGSNKE
# )

# program = PicardIterator(
#     unconstrained_eq,
#     current_opt_problem,
#     fixed_coils=True,
#     convergence=DudsonConvergence(1e-2, 1e-10), # Matches FreeGSNKE rtol=1e-3, but they also have an or atol condition - psi_maxchange < 1e-10
#     relaxation=0.0, # FreeGSNKE blend = 0.0
#     maxiter=50,
# )

# program()

# f, ax = plt.subplots()
# unconstrained_eq.plot(ax=ax)
# constraints_set.plot(ax=ax)
# coilset.plot(ax=ax)
# plt.show()

# %%
# %%time
# constrained_eq = deepcopy(eq)
constrained_eq = deepcopy(eq)
# FreeGSNKE uses least squares problem with Tikhonov regularisation term
# I think that SLSQP is aapropriate algo, constrained least squares
current_opt_problem = TikhonovCurrentCOP(
    constrained_eq,
    targets=constraints_set,
    gamma=5e-7,
    opt_algorithm="SLSQP",
    opt_conditions={"max_eval": 2000, "ftol_rel": 1e-6},
    opt_parameters={},
    max_currents=1e6,
    constraints=constraints_list,
)

program = PicardIterator(
    constrained_eq,
    current_opt_problem,
    fixed_coils=True,
    convergence=DudsonConvergence(
        1.0e-3, 1e-10
    ),  # DudsonConvergence(1.0e-3, 1e-10), # Matches FreeGSNKE rtol=1e-3, but they also have an or atol condition - psi_maxchange < 1e-10
    relaxation=0.0,  # FreeGSNKE blend = 0.0
    maxiter=30,
)
_ = program()
f, ax = plt.subplots()
constrained_eq.plot(ax=ax)
constraints_set.plot(ax=ax)
constrained_eq.coilset.plot(ax=ax)
plt.show()

# %%
import pickle

with open("freeGSNK_picard.pk", "rb") as f:
    picard_dict = pickle.load(f)

# %%
with open("freeGSNK_newton.pk", "rb") as f:
    newton_dict = pickle.load(f)

# %%
psi_diff_newton = np.abs(constrained_eq.psi() - newton_dict["psi"]) / np.max(
    np.abs(constrained_eq.psi())
)
psi_diff_picard = np.abs(constrained_eq.psi() - picard_dict["psi"]) / np.max(
    np.abs(constrained_eq.psi())
)
psi_diff_free = np.abs(picard_dict["psi"] - newton_dict["psi"]) / np.max(
    np.abs(picard_dict["psi"])
)

# %%
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from bluemira.equilibria.find import in_plasma

# %%
mask = in_plasma(
    constrained_eq.x,
    constrained_eq.z,
    constrained_eq.psi(),
    o_points=constrained_eq._o_points,
    x_points=constrained_eq._x_points,
)


# %%
def plot_diff(ax, grid, psi_diff):
    # psi_diff *= mask
    levels = np.linspace(np.amin(psi_diff), np.amax(psi_diff), 50)
    x, z = grid.x, grid.z
    vmin = np.amin(psi_diff)
    vmax = np.amax(psi_diff)
    cax = make_axes_locatable(ax).append_axes("right", size="5%", pad="2%")
    im = ax.contourf(
        x,
        z,
        psi_diff,
        levels=levels,
        cmap="plasma",
        zorder=8,
        vmin=vmin,
        vmax=vmax,
    )
    plt.colorbar(mappable=im, cax=cax, ticks=np.linspace(vmin, vmax, 10))
    plt.tight_layout()
    plt.show()


# %%
_, ax = plt.subplots()
plot_diff(ax, constrained_eq.grid, psi_diff_newton)

# %%
_, ax = plt.subplots()
plot_diff(ax, constrained_eq.grid, psi_diff_picard)

# %%
_, ax = plt.subplots()
plot_diff(ax, constrained_eq.grid, psi_diff_free)
