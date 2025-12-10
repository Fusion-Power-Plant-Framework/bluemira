# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,title,-all
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


# %%
# Just trying some simple boxes, we can look up somemore sensible values after testing
from copy import deepcopy
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.find_legs import LegFlux
from bluemira.equilibria.optimisation.constraints import (
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_approx_functions import (
    fs_fit_metric,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_constraints import (
    ToroidalHarmonicConstraint,
)
from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import (
    TauLimit,
    plot_toroidal_harmonic_approximation,
    toroidal_harmonic_approximation,
    toroidal_harmonic_grid_and_coil_setup,
)
from bluemira.equilibria.optimisation.problem._nested_position import (
    NestedCoilsetPositionCOP,
    PulsedNestedPositionCOP,
)
from bluemira.equilibria.optimisation.problem._position import CoilsetPositionCOP
from bluemira.equilibria.optimisation.problem._tikhonov import TikhonovCurrentCOP
from bluemira.equilibria.plotting import PLOT_DEFAULTS
from bluemira.equilibria.solve import DudsonConvergence, PicardIterator
from bluemira.geometry.tools import make_polygon
from bluemira.optimisation._algorithm import Algorithm
from bluemira.utilities.positioning import PositionMapper, RegionInterpolator

# %%
# %% # TEST WORK
TEST_PATH = get_bluemira_path("equilibria/test_data", subfolder="tests")

eq = Equilibrium.from_eqdsk(
    Path(TEST_PATH, "eqref_OOB.json").as_posix(),
    from_cocos=7,
)

f, ax = plt.subplots()
eq.plot(ax=ax)
eq.coilset.plot(ax=ax)
# %%
psi_norm = 0.95
R_0, Z_0 = eq.effective_centre()


th_params = toroidal_harmonic_grid_and_coil_setup(
    eq=eq, R_0=R_0, Z_0=Z_0, tau_limit=TauLimit.COIL
)


result = toroidal_harmonic_approximation(
    eq=eq,
    th_params=th_params,
    psi_norm=psi_norm,
    n_degrees_of_freedom=6,
    max_harmonic_mode=5,
    plasma_mask=True,
)
f, ax = plot_toroidal_harmonic_approximation(
    eq=eq, th_params=th_params, result=result, psi_norm=psi_norm
)
eq.plot(ax=ax)
eq.coilset.plot(ax=ax)
ax.set_title("Comparison of bluemira coilset psi to TH approx.")
plt.show()


# %%
# Create a constraint
th_constraint = ToroidalHarmonicConstraint(
    th_result=result,
    constraint_type="inequality",
)
# Ensure control coils are set to those that can be used in the toroidal
# harmonic approximation
eq.coilset.control = list(th_params.th_coil_names)

# Show the constraint region
f, ax = plt.subplots()
th_constraint.plot(ax=ax)
eq.coilset.plot(ax=ax)
eq.plot(ax=ax)


# %%

# %%
# MOVING LEGS
# Create isoflux constraints for the target inner leg
ref_lcfs = eq.get_LCFS()
legs = LegFlux(eq).get_legs()
lcfs = eq.get_LCFS()
x_bdry, z_bdry = lcfs.x, lcfs.z
arg_inner = np.argmin(x_bdry)

# # Define points to use for the isoflux constraints
inner_leg_points_x = np.array([
    6.5,
    7.0,
    7.5,
])

inner_leg_points_z = (
    np.array([
        6.25,
        5.95,
        5.7,
    ])
    + 0.3  # originally 0.3
)


outer_legs_x = (
    np.array([
        9.7,
        9.89,
        10.1,
    ])
    + 1.0
)

outer_legs_z = np.array([6.5, 7.0, 7.5]) + 1.0


# outer_legs_x = np.array([
#     10.2,
#     10.8,
#     11.5,
# ])

# outer_legs_z = np.array([6.8, 8.0, 9.0])


# # Create the necessary isoflux constraints for the inner and outer legs, for
# # the upper and lower divertors.

leg_choice = "outer"

if leg_choice == "outer":
    leg_x = legs["lower_outer"][0].x
    leg_z = legs["lower_outer"][0].z
elif leg_choice == "inner":
    leg_x = legs["lower_inner"][0].x
    leg_z = legs["lower_inner"][0].z
elif leg_choice == "both":
    leg_x = np.append(legs["lower_inner"][0].x, legs["lower_outer"][0].x, axis=0)
    leg_z = np.append(legs["lower_inner"][0].z, legs["lower_outer"][0].z, axis=0)

leg_x = leg_x[0::10]
leg_z = leg_z[0::10]

arg_inner = np.argmin(ref_lcfs.x)
isofluxouter = IsofluxConstraint(
    leg_x,
    leg_z,
    ref_lcfs.x[arg_inner],
    ref_lcfs.z[arg_inner],
    constraint_value=0.0,
)

isofluxouter_upper = IsofluxConstraint(
    leg_x,
    -leg_z,
    ref_lcfs.x[arg_inner],
    ref_lcfs.z[arg_inner],
    constraint_value=0.0,
)


leg_x = legs["lower_inner"][0].x
leg_z = legs["lower_inner"][0].z
leg_x = leg_x[0::10]
leg_z = leg_z[0::10]

arg_inner = np.argmin(ref_lcfs.x)
isofluxinner = IsofluxConstraint(
    leg_x,
    leg_z,
    ref_lcfs.x[arg_inner],
    ref_lcfs.z[arg_inner],
    constraint_value=0.0,
)

isofluxinner_upper = IsofluxConstraint(
    leg_x,
    -leg_z,
    ref_lcfs.x[arg_inner],
    ref_lcfs.z[arg_inner],
    constraint_value=0.0,
)

inner_leg_points_x = (
    np.array([
        6.5,
        7.0,
        7.5,
    ])
    - 2.0
)

inner_leg_points_z = (
    np.array([
        6.25,
        5.9,
        5.6,
    ])
    + 0.3
)

# these gave a nice result:
outer_legs_x = np.array([
    9.0,
    10.0,
    10.9,
])

outer_legs_z = np.array([6.5, 7.3, 7.8])

# # 2
outer_legs_x = np.array([
    8.5,
    9.0,
    10.0,
    10.9,
])

outer_legs_z = np.array([5.9, 6.2, 6.8, 7.2])


SN_moved_outer_leg_lower = IsofluxConstraint(
    outer_legs_x,
    -outer_legs_z,
    ref_x=x_bdry[arg_inner],
    ref_z=z_bdry[arg_inner],
    tolerance=1e-3,
)


SN_moved_inner_leg_lower = IsofluxConstraint(
    inner_leg_points_x,
    -inner_leg_points_z,
    ref_x=x_bdry[arg_inner],
    ref_z=z_bdry[arg_inner],
    tolerance=1e-3,
)


# Plot the isoflux points and the starting equilibrium for reference
f, ax = plt.subplots()
eq.plot(ax=ax)
# isofluxouter.plot(ax=ax)
isofluxinner.plot(ax=ax)
# isofluxinner_upper.plot(ax=ax)
SN_moved_outer_leg_lower.plot(ax=ax)
# isofluxouter.plot(ax=ax)
# SN_moved_inner_leg_lower.plot(ax=ax)
eq.coilset.plot(ax=ax)
plt.show()


# %%
os, xs = eq.get_OX_points()
o_point = FieldNullConstraint(
    os[0].x,
    os[0].z,
    tolerance=1e-6,
)
x_point = FieldNullConstraint(
    xs[0].x,
    xs[0].z,
    tolerance=1e-6,
)

x_point_2 = FieldNullConstraint(
    xs[1].x,
    xs[1].z,
    tolerance=1e-6,
)
# %%
f, ax = plt.subplots()
size = [2, 1.8, 0.7, 2, 2]
region_interpolators = {}
pf_coils = eq.coilset.get_coiltype("PF")
pf_coils.remove_coil("PF_4")
for x, z, name, s in zip(pf_coils.x, pf_coils.z, pf_coils.name, size, strict=False):
    region = make_polygon(
        {"x": [x - s, x + s, x + s, x - s], "z": [z - s, z - s, z + s, z + s]},
        closed=True,
    )
    plot_region = region.discretise(100)
    ax.plot(plot_region.x, plot_region.z)

    region_interpolators[name] = RegionInterpolator(region)
position_mapper = PositionMapper(region_interpolators)
eq.coilset.plot(ax=ax)
th_constraint.plot(ax=ax)
plt.show()


# %%
th_current_opt_eq = deepcopy(eq)

current_opt_problem = TikhonovCurrentCOP(
    th_current_opt_eq,
    targets=MagneticConstraintSet([
        th_constraint,
        SN_moved_outer_leg_lower,
        isofluxinner,
    ]),
    gamma=1e-12,
    opt_algorithm="SLSQP",
    opt_conditions={"max_eval": 1000, "ftol_rel": 1e-4},
    opt_parameters={"initial_step": 0.1},
    max_currents=3e10,
    constraints=[o_point, x_point, x_point_2],
)

position_opt_problem = NestedCoilsetPositionCOP(
    sub_opt=current_opt_problem,
    position_mapper=position_mapper,
    opt_algorithm=Algorithm.SBPLX,
    opt_conditions={"max_eval": 1000, "ftol_rel": 1e-4},
    constraints=[],
)
optimised_coilset = position_opt_problem.optimise().coilset


# %%
# th_current_opt_eq = deepcopy(nested_opt_eq)
program = PicardIterator(
    th_current_opt_eq,
    current_opt_problem,
    fixed_coils=True,
    convergence=DudsonConvergence(1e-3),
    relaxation=0.0,
    maxiter=30,
)
program()

# %%
f, ax = plt.subplots()
th_current_opt_eq.plot(ax=ax)
th_current_opt_eq.coilset.plot(ax=ax)

# %%
original_FS = (  # noqa: N816
    eq.get_LCFS() if np.isclose(psi_norm, 1.0) else eq.get_flux_surface(psi_norm)
)
approx_FS = th_current_opt_eq.get_flux_surface(psi_norm)  # noqa: N816

total_psi_diff = np.abs(
    eq.coilset.psi(eq.grid.x, eq.grid.z)
    - th_current_opt_eq.coilset.psi(th_current_opt_eq.grid.x, th_current_opt_eq.grid.z)
) / np.max(np.abs(eq.coilset.psi(th_current_opt_eq.grid.x, th_current_opt_eq.grid.z)))

f, ax = plt.subplots()
nlevels = PLOT_DEFAULTS["psi"]["nlevels"]
cmap = PLOT_DEFAULTS["psi"]["cmap"]
ax.plot(
    approx_FS.x,
    approx_FS.z,
    color="red",
    label="Approximate FS after \noptimising using TH",
)
ax.plot(
    original_FS.x,
    original_FS.z,
    color="blue",
    label="Original equilibrium FS \nfrom Bluemira",
)
im = ax.contourf(eq.grid.x, eq.grid.z, total_psi_diff, levels=nlevels, cmap=cmap)
f.colorbar(mappable=im)
ax.set_title(
    "Absolute relative difference between coilset psi and TH approximation psi", y=1.05
)
ax.legend(bbox_to_anchor=(1.1, 1.05))
# eq.coilset.plot(ax=ax)
plt.show()

# %%
fit_metric = fs_fit_metric(original_FS, approx_FS)
print(f"fit metric = {fit_metric}")
