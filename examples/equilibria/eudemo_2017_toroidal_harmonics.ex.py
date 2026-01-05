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
Attempt at recreating the EU-DEMO 2017 reference equilibria from a known coilset.
"""

# %% [markdown]
#
# # EU-DEMO 2017 reference breakdown and equilibrium benchmark

# %%
import json
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.file import get_bluemira_path
from bluemira.display import plot_defaults
from bluemira.equilibria.coils import Coil, CoilSet
from bluemira.equilibria.equilibrium import Breakdown, Equilibrium
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.optimisation.constraints import (
    CoilFieldConstraints,
    CoilForceConstraints,
    FieldNullConstraint,
    MagneticConstraintSet,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_constraints import (
    ToroidalHarmonicConstraint,
)
from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import (
    TauLimit,
    plot_toroidal_harmonic_approximation,
    toroidal_harmonic_approximation,
    toroidal_harmonic_grid_and_coil_setup,
    toroidal_harmonics_to_positions,
)
from bluemira.equilibria.optimisation.problem import (
    BreakdownCOP,
    OutboardBreakdownZoneStrategy,
    TikhonovCurrentCOP,
)
from bluemira.equilibria.physics import calc_psib
from bluemira.equilibria.plotting import PLOT_DEFAULTS
from bluemira.equilibria.solve import DudsonConvergence, PicardIterator

# %%
plot_defaults()

# uncomment for pop out plots
# with contextlib.suppress(AttributeError):
#     get_ipython().run_line_magic("matplotlib", "qt")

# %% [markdown]
# Load up all the eqs etc. we will use from the eudemo_2017 example.

# %%
# CREATE reference data
path = get_bluemira_path("equilibria", subfolder="examples")
name = "EUDEMO_2017_CREATE_SOF_separatrix.json"
with open(Path(path, name)) as file:
    data = json.load(file)

sof_xbdry = data["xbdry"]
sof_zbdry = data["zbdry"]

# %%
# Breakdown - remaking here as to_eqdsk needs updating

# Make the same CoilSet as CREATE
x = [5.4, 14, 17.75, 17.75, 14.0, 7, 2.77, 2.77, 2.77, 2.77, 2.77]
z = [9.26, 7.9, 2.5, -2.5, -7.9, -10.5, 7.07, 4.08, -0.4, -4.88, -7.86]
dx = [0.6, 0.7, 0.5, 0.5, 0.7, 1.0, 0.4, 0.4, 0.4, 0.4, 0.4]
dz = [0.6, 0.7, 0.5, 0.5, 0.7, 1.0, 2.99 / 2, 2.99 / 2, 5.97 / 2, 2.99 / 2, 2.99 / 2]
coils = []
j = 1
for i, (xi, zi, dxi, dzi) in enumerate(zip(x, z, dx, dz, strict=False)):
    if j > 6:  # noqa: PLR2004
        j = 1
    ctype = "PF" if i < 6 else "CS"  # noqa: PLR2004
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
coilset.assign_material("CS", j_max=16.5e6, b_max=12.5)
coilset.assign_material("PF", j_max=12.5e6, b_max=11)
coilset.fix_sizes()
coilset.discretisation = 0.3

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
PF_Fz_max = 450e6  # [N]
CS_Fz_sum = 300e6  # [N]
CS_Fz_sep = 350e6  # [N]

# Use the same grid as CREATE (but less discretised)
grid = Grid(2, 16.0, -9.0, 9.0, 100, 100)

# Set up the Breakdown object
field_constraints = CoilFieldConstraints(coilset, coilset.b_max, tolerance=1e-6)
force_constraints = CoilForceConstraints(
    coilset, PF_Fz_max, CS_Fz_sum, CS_Fz_sep, tolerance=1e-6
)

max_currents = coilset.get_max_current(0.0)
breakdown = Breakdown(deepcopy(coilset), grid)

bd_opt_problem = BreakdownCOP(
    breakdown,
    OutboardBreakdownZoneStrategy(R_0, A, 0.225),
    opt_algorithm="COBYLA",
    opt_conditions={"max_eval": 3000, "ftol_rel": 1e-6},
    max_currents=max_currents,
    B_stray_max=1e-3,
    B_stray_con_tol=1e-6,
    n_B_stray_points=10,
    constraints=[field_constraints, force_constraints],
)

coilset = bd_opt_problem.optimise(x0=max_currents).coilset
# bluemira_print(f"Breakdown psi: {breakdown.breakdown_psi * 2 * np.pi:.2f} V.s")
breakdown.plot()
plt.show()

# %%
# Equilbria obtained from initial unconstrained optimisation of CREATE Reference,
# of the three profile classes availble for use, we have selected CustomProfile.
file_path = Path(
    get_bluemira_path("equilibria/data", subfolder="examples"),
    "unconstrained_reference_eq.json",
)

unconstrained_eq = Equilibrium.from_eqdsk(file_path, from_cocos=3, qpsi_positive=False)
unconstrained_eq.plot()
plt.show()

# %%
# Reference equilibrium
# (not from eudemo_2017 example but used unconstrained_eq as input),
# obtained from running an TikhonovCurrentCOP with the following non-default inputs:
# 1. targets set to eudemo_2017 isoflux
# 2. constraints are eudemo_2017 x-point, field & force
# 3. gamma=0.0, opt_algorithm="SLSQP",
# opt_conditions={"max_eval": 2000, "ftol_rel": 1e-6}
# 4. max_currents are set using get_max_current
# 5. Picard iterator used Dudson Convergence of 1e-4 and relaxation=0.1.

file_path = Path(
    get_bluemira_path("equilibria/data", subfolder="examples"),
    "constrained_reference_eq.json",
)

reference_eq = Equilibrium.from_eqdsk(file_path, from_cocos=3, qpsi_positive=False)
reference_eq.plot()
plt.show()

# %%
# SOF from eudemo_2017 example
file_path = Path(get_bluemira_path("equilibria/data", subfolder="examples"), "SOF.json")

sof = Equilibrium.from_eqdsk(file_path, from_cocos=3, qpsi_positive=False)
sof.plot()
plt.show()

# %%
# EOF from eudemo_2017 example
file_path = Path(get_bluemira_path("equilibria/data", subfolder="examples"), "EOF.json")

eof = Equilibrium.from_eqdsk(file_path, from_cocos=3, qpsi_positive=False)
eof.plot()
plt.show()

# %%
# Calcualte psi at boudary for SOF and EOF
psi_sof = calc_psib(breakdown.breakdown_psi * 2 * np.pi, R_0, I_p, l_i, c_ejima)
psi_eof = psi_sof - tau_flattop * v_burn

# CREATE then knocked off an extra 10 V.s for misc plasma stuff I didn't look into
psi_sof -= 10
psi_eof -= 10

# Get psi at ref eq boundary
ref_lcfs = reference_eq.get_LCFS()
ind = np.argmax(ref_lcfs.x)
psi_at_ref_lcfs = reference_eq.psi(ref_lcfs.x[ind], ref_lcfs.z[ind])

print(f"{psi_at_ref_lcfs=}")
print(f"{psi_sof=}")
print(f"{psi_eof=}")


# %% [markdown]
# ### Test 1: Selected modes and amps
#
# Look at the modes and amps chosen by TH approximation function for the reference
# equlibrium and the optimised SOF and EOF from the EU DEMO 2017 NB.


# %%
def th_approx_and_info(eq, setup):
    """Get the TH approx, plot and return the result"""
    R_0, Z_0 = eq.effective_centre()
    th_params_ref = toroidal_harmonic_grid_and_coil_setup(
        eq=eq, R_0=R_0, Z_0=Z_0, tau_limit=setup["tau_limit"]
    )

    result = toroidal_harmonic_approximation(
        eq=eq,
        th_params=th_params_ref,
        psi_norm=setup["psi_norm"],
        n_degrees_of_freedom=setup["n_degrees_of_freedom"],
        max_harmonic_mode=setup["max_harmonic_mode"],
        plasma_mask=setup["plasma_mask"],
    )

    _f, ax = plot_toroidal_harmonic_approximation(
        eq=eq, th_params=th_params_ref, result=result, psi_norm=setup["psi_norm"]
    )
    eq.coilset.plot(ax=ax)
    ax.set_title("Comparison of bluemira coilset psi to TH approx.")
    plt.show()

    print("Modes")
    print("------")
    print(f"sin m = {result.sin_m}")
    print(f"cos m = {result.cos_m}")
    print(" ")
    print("Mode Amplitudes")
    print("------")
    print(f"sin amps  = {result.sin_amplitudes}")
    print(f"cos amps = {result.cos_amplitudes}")

    return result


# raise EquilibriaError
# %%
# Same set up norm for all eq
setup = {
    "tau_limit": TauLimit.COIL,
    "psi_norm": 0.95,
    "n_degrees_of_freedom": 6,
    "max_harmonic_mode": 5,
    "plasma_mask": True,
}

# %%
ref_result = th_approx_and_info(reference_eq, setup)

# %%
sof_result = th_approx_and_info(sof, setup)

# %%
eof_result = th_approx_and_info(eof, setup)

# %% [markdown]
# ### Test 2 - ref to sof/eof
#
# Set up toroidal harmonics approximation for reference equilibrium
# and modify harmonic amplitude values for use as SOF and EOF constraints.

# %%
# Don't forget factor of 2 pi
# Note: there was an np.abs used on psi_sof and psi_eof before
sof_factor = psi_sof / (2 * np.pi * psi_at_ref_lcfs)
eof_factor = psi_eof / (2 * np.pi * psi_at_ref_lcfs)

print(f"{psi_sof=}")
print(f"{psi_eof=}")
print(f"{sof_factor=}")
print(f"{eof_factor=}")

# %%
harm_cos_term, harm_sin_term = toroidal_harmonics_to_positions(
    th_params=ref_result.th_params, n_allowed=setup["n_degrees_of_freedom"]
)

harm_cos_term = harm_cos_term[ref_result.cos_m, :]
harm_sin_term = harm_sin_term[ref_result.sin_m, :]

# cos mat * cos amps + sin mat * sin amps
psi_calc_sof = harm_cos_term.T @ (
    ref_result.cos_amplitudes * sof_factor
) + harm_sin_term.T @ (ref_result.sin_amplitudes * sof_factor)


psi_calc_eof = harm_cos_term.T @ (
    ref_result.cos_amplitudes * eof_factor
) + harm_sin_term.T @ (ref_result.sin_amplitudes * eof_factor)


psi_calc_ref = harm_cos_term.T @ (ref_result.cos_amplitudes) + harm_sin_term.T @ (
    ref_result.sin_amplitudes
)


# %%
nlevels = PLOT_DEFAULTS["psi"]["nlevels"]
cmap = PLOT_DEFAULTS["psi"]["cmap"]
f, axs = plt.subplots(1, 5)

axs[0].contourf(
    ref_result.th_params.R,
    ref_result.th_params.Z,
    psi_calc_sof.T,
    levels=nlevels,
    cmap=cmap,
)
axs[0].set_title("SOF")


axs[1].contourf(
    ref_result.th_params.R,
    ref_result.th_params.Z,
    psi_calc_eof.T,
    levels=nlevels,
    cmap=cmap,
)
axs[1].set_title("EOF")


axs[2].contourf(
    ref_result.th_params.R,
    ref_result.th_params.Z,
    psi_calc_ref.T,
    levels=nlevels,
    cmap=cmap,
)
axs[2].set_title("Ref psi from ref harmonics")


axs[3].contourf(
    ref_result.th_params.R,
    ref_result.th_params.Z,
    ref_result.coilset_psi,
    levels=nlevels,
    cmap=cmap,
)
axs[3].set_title("Coilset psi from th approx result")


axs[4].contourf(
    ref_result.th_params.R,
    ref_result.th_params.Z,
    reference_eq.coilset.psi(ref_result.th_params.R, ref_result.th_params.Z),
    levels=nlevels,
    cmap=cmap,
)
axs[4].set_title("Bluemira coilset psi")


axs[0].set_aspect("equal")
axs[1].set_aspect("equal")
axs[2].set_aspect("equal")
axs[3].set_aspect("equal")
axs[4].set_aspect("equal")
# %%
nlevels = PLOT_DEFAULTS["psi"]["nlevels"]
cmap = PLOT_DEFAULTS["psi"]["cmap"]
f, axs = plt.subplots(1, 5)

axs[0].contourf(
    ref_result.th_params.R,
    ref_result.th_params.Z,
    psi_calc_sof.T + ref_result.fixed_psi,
    levels=nlevels,
    cmap=cmap,
)
axs[0].set_title("SOF + fixed psi from approx")


axs[1].contourf(
    ref_result.th_params.R,
    ref_result.th_params.Z,
    psi_calc_eof.T + ref_result.fixed_psi,
    levels=nlevels,
    cmap=cmap,
)
axs[1].set_title("EOF + fixed psi from approx")


axs[2].contourf(
    ref_result.th_params.R,
    ref_result.th_params.Z,
    psi_calc_ref.T + ref_result.fixed_psi,
    levels=nlevels,
    cmap=cmap,
)
axs[2].set_title("Ref psi from ref harmonics \n+ fixed psi from approx")

axs[3].contourf(
    ref_result.th_params.R,
    ref_result.th_params.Z,
    ref_result.coilset_psi + ref_result.fixed_psi,
    levels=nlevels,
    cmap=cmap,
)
axs[3].set_title("Coilset psi from th approx result \n+ fixed psi from approx")


axs[4].contourf(
    ref_result.th_params.R,
    ref_result.th_params.Z,
    reference_eq.psi(ref_result.th_params.R, ref_result.th_params.Z),
    levels=nlevels,
    cmap=cmap,
)
axs[4].set_title("Bluemira coilset psi \n+ fixed psi")


axs[0].set_aspect("equal")
axs[1].set_aspect("equal")
axs[2].set_aspect("equal")
axs[3].set_aspect("equal")
axs[4].set_aspect("equal")

# %%
# make sof and eof results - same th params
# need to change the amplitudes as these are used in the constraint
# only amps and th params are used in constraint
sof_result = deepcopy(ref_result)
sof_result.cos_amplitudes = ref_result.cos_amplitudes * sof_factor
sof_result.sin_amplitudes = ref_result.sin_amplitudes * sof_factor

eof_result = deepcopy(ref_result)
eof_result.cos_amplitudes = ref_result.cos_amplitudes * eof_factor
eof_result.sin_amplitudes = ref_result.sin_amplitudes * eof_factor


sof_constraint = ToroidalHarmonicConstraint(
    th_result=sof_result, constraint_type="inequality"
)

eof_constraint = ToroidalHarmonicConstraint(
    th_result=eof_result, constraint_type="inequality"
)
# %%
os, xs = reference_eq.get_OX_points()
x_point_constraint = FieldNullConstraint(
    xs[0].x,
    xs[0].z,
    tolerance=1e-3,
)
o_point_constraint = FieldNullConstraint(os[0].x, os[0].z, tolerance=1e-3)

# %%
# SOF OPT
sof_opt_eq = deepcopy(reference_eq)
sof_opt_eq.coilset.control = ref_result.th_params.th_coil_names

current_opt_problem = TikhonovCurrentCOP(
    sof_opt_eq,
    targets=MagneticConstraintSet([
        sof_constraint,
    ]),
    gamma=1e-12,
    opt_algorithm="SLSQP",
    # opt_conditions={"max_eval": 1000, "ftol_rel": 1e-4},
    # opt_parameters={"initial_step": 0.1},
    max_currents=3e10,
    constraints=[
        x_point_constraint,
        o_point_constraint,
        # sof_constraint,
    ],
)

# current_opt_problem = MinimalCurrentCOP(
#     sof_opt_eq,
#     opt_algorithm="COBYLA",
#     opt_conditions={"max_eval": 1000, "ftol_rel": 1e-6},
#     max_currents=3e10,
#     constraints=[sof_constraint],
# )

program = PicardIterator(
    sof_opt_eq,
    current_opt_problem,
    fixed_coils=True,
    convergence=DudsonConvergence(1e-3),
    relaxation=0.0,
    maxiter=50,
)
program()

# Plot
f, (ax_1, ax_2) = plt.subplots(1, 2)

reference_eq.plot(ax=ax_1)
ax_1.set_title("Starting Equilibrium")

sof_opt_eq.plot(ax=ax_2)
ax_2.set_title("Optimised Equilibrium SOF")
plt.show()

# %%
# EOF OPT
eof_opt_eq = deepcopy(reference_eq)
eof_opt_eq.coilset.control = ref_result.th_params.th_coil_names

current_opt_problem = TikhonovCurrentCOP(
    eof_opt_eq,
    targets=MagneticConstraintSet([
        eof_constraint,
    ]),
    gamma=1e-12,
    opt_algorithm="SLSQP",
    # opt_conditions={"max_eval": 1000, "ftol_rel": 1e-4},
    # opt_parameters={"initial_step": 0.1},
    max_currents=3e10,
    constraints=[
        x_point_constraint,
        o_point_constraint,
        # eof_constraint,
    ],
)

# current_opt_problem.optimise()
# eof_opt_eq.solve()
# f, ax = plt.subplots()
# eof_opt_eq.plot(ax=ax)

# %%
program = PicardIterator(
    eof_opt_eq,
    current_opt_problem,
    fixed_coils=True,
    convergence=DudsonConvergence(1e-3),
    relaxation=0.0,
    maxiter=50,
)
program()

# Plot
f, (ax_1, ax_2) = plt.subplots(1, 2)

reference_eq.plot(ax=ax_1)
ax_1.set_title("Starting Equilibrium")

eof_opt_eq.plot(ax=ax_2)
ax_2.set_title("Optimised Equilibrium EOF")
plt.show()

# %% [markdown]
# ### Test 3 - reduced psi
#
# Same as above but try some smaller changes in psi for SOF and EOF

# %%
# Just going to choose
psi_sof = psi_at_ref_lcfs * 1.0
psi_eof = -psi_at_ref_lcfs * 1.0

sof_factor = psi_sof / (2 * np.pi * psi_at_ref_lcfs)
eof_factor = psi_eof / (2 * np.pi * psi_at_ref_lcfs)

sof_factor = 1.0
eof_factor = -1.0

sof_result = deepcopy(ref_result)
sof_result.cos_amplitudes = ref_result.cos_amplitudes * sof_factor
sof_result.sin_amplitudes = ref_result.sin_amplitudes * sof_factor

eof_result = deepcopy(ref_result)
eof_result.cos_amplitudes = ref_result.cos_amplitudes * eof_factor
eof_result.sin_amplitudes = ref_result.sin_amplitudes * eof_factor


sof_constraint = ToroidalHarmonicConstraint(
    th_result=sof_result, constraint_type="inequality"
)

eof_constraint = ToroidalHarmonicConstraint(
    th_result=eof_result, constraint_type="inequality"
)

# %%
sof_opt_eq = deepcopy(reference_eq)
sof_opt_eq.coilset.control = ref_result.th_params.th_coil_names

current_opt_problem = TikhonovCurrentCOP(
    sof_opt_eq,
    targets=MagneticConstraintSet([
        sof_constraint,
    ]),
    gamma=1e-12,
    opt_algorithm="SLSQP",
    # opt_conditions={"max_eval": 1000, "ftol_rel": 1e-4},
    # opt_parameters={"initial_step": 0.1},
    max_currents=3e10,
    constraints=[
        x_point_constraint,
        o_point_constraint,
        # sof_constraint,
    ],
)

program = PicardIterator(
    sof_opt_eq,
    current_opt_problem,
    fixed_coils=True,
    convergence=DudsonConvergence(1e-3),
    relaxation=0.1,
    maxiter=50,
)
program()

# Plot
f, (ax_1, ax_2) = plt.subplots(1, 2)

reference_eq.plot(ax=ax_1)
ax_1.set_title("Starting Equilibrium")

sof_opt_eq.plot(ax=ax_2)
ax_2.set_title("Optimised Equilibrium SOF")
plt.show()

# %%
eof_opt_eq = deepcopy(reference_eq)
eof_opt_eq.coilset.control = ref_result.th_params.th_coil_names

current_opt_problem = TikhonovCurrentCOP(
    eof_opt_eq,
    targets=MagneticConstraintSet([
        eof_constraint,
    ]),
    gamma=1e-12,
    opt_algorithm="SLSQP",
    # opt_conditions={"max_eval": 1000, "ftol_rel": 1e-4},
    # opt_parameters={"initial_step": 0.1},
    max_currents=3e10,
    constraints=[
        x_point_constraint,
        o_point_constraint,
        # eof_constraint,
    ],
)

program = PicardIterator(
    eof_opt_eq,
    current_opt_problem,
    fixed_coils=True,
    convergence=DudsonConvergence(1e-3),
    relaxation=0.1,
    maxiter=50,
)
program()

# Plot
f, (ax_1, ax_2) = plt.subplots(1, 2)

reference_eq.plot(ax=ax_1)
ax_1.set_title("Starting Equilibrium")

eof_opt_eq.plot(ax=ax_2)
ax_2.set_title("Optimised Equilibrium EOF")
plt.show()

# %% [markdown]
# ### Test 3 - reverse from eof to ref

# %%
# Calcualte psi at boudary for SOF and EOF
psi_sof = calc_psib(breakdown.breakdown_psi * 2 * np.pi, R_0, I_p, l_i, c_ejima)
psi_eof = psi_sof - tau_flattop * v_burn
psi_eof -= 10

reverse_factor = psi_at_ref_lcfs / (2 * np.pi * psi_eof)

new_ref_result = deepcopy(eof_result)
new_ref_result.cos_amplitudes = ref_result.cos_amplitudes * reverse_factor
new_ref_result.sin_amplitudes = ref_result.sin_amplitudes * reverse_factor

reverse_constraint = ToroidalHarmonicConstraint(
    th_result=eof_result, constraint_type="inequality"
)

# %%
os, xs = eof.get_OX_points()
x_point_constraint = FieldNullConstraint(
    xs[0].x,
    xs[0].z,
    tolerance=1e-3,
)
o_point_constraint = FieldNullConstraint(os[0].x, os[0].z, tolerance=1e-3)

# %%
reverse_eq = deepcopy(eof)
reverse_eq.coilset.control = eof_result.th_params.th_coil_names

current_opt_problem = TikhonovCurrentCOP(
    reverse_eq,
    targets=MagneticConstraintSet([
        reverse_constraint,
    ]),
    gamma=1e-12,
    opt_algorithm="SLSQP",
    # opt_conditions={"max_eval": 1000, "ftol_rel": 1e-4},
    # opt_parameters={"initial_step": 0.1},
    max_currents=3e10,
    constraints=[
        x_point_constraint,
        o_point_constraint,
        # eof_constraint,
    ],
)

program = PicardIterator(
    reverse_eq,
    current_opt_problem,
    fixed_coils=True,
    convergence=DudsonConvergence(1e-3),
    relaxation=0.1,
    maxiter=50,
)
program()

# Plot
f, (ax_1, ax_2) = plt.subplots(1, 2)

eof.plot(ax=ax_1)
ax_1.set_title("Starting Equilibrium")

reverse_eq.plot(ax=ax_2)
ax_2.set_title("Optimised Equilibrium")
plt.show()
