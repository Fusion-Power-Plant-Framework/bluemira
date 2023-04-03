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
# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
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
Finds a ST equilibrium in a double null configuration, using a constrained
optimisation method with bound constraints on the maximum coil currents
and on the position of the inboard midplane.
"""

# %% [markdown]
#
# # Introduction
#
# In this example, we will outline how to specify a CoilsetOptimisationProblem
# that specifies how an 'optimised' coilset state is found during the Free Boundary
# Equilibrium solve step.
#
# # Imports
#
# Import necessary equilbria module definitions.

# %%
import matplotlib.pyplot as plt
import numpy as np

import bluemira.equilibria.opt_constraint_funcs as opt_constraint_funcs
from bluemira.equilibria.coils import Coil, CoilSet, SymmetricCircuit
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.opt_constraints import IsofluxConstraint, MagneticConstraintSet
from bluemira.equilibria.opt_problems import (
    TikhonovCurrentCOP,
    UnconstrainedTikhonovCurrentGradientCOP,
)
from bluemira.equilibria.profiles import CustomProfile
from bluemira.equilibria.solve import DudsonConvergence, PicardIterator
from bluemira.utilities.opt_problems import OptimisationConstraint
from bluemira.utilities.optimiser import Optimiser

# %% [markdown]
#
# # OptimisationProblem
#
# The `OptimisationProblem` class is intended to be the general base class for defining
# optimisation problems across Bluemira.
#
# It is constructed from the following four objects:
# - `parameterisation`
#     - Object storing data that is updated during the optimisation to be carried out.
# - `optimiser: Optimiser`
#     - `Optimiser` object specifying the numerical algorithm used to optimise an array
#        representing the parameterisation state. Usually based on NLOpt.
# - `objective: OptimisationObjective`
#     - OptimisationObjective object, specifying objective function to be minimised
#       during optimisation.
# - `constraints: List[OptimisationConstraint]`
#     - List of `OptimisationConstraints`, specifying the set of constraints that must
#       be satisfied during the numerical optimisation.
#
#
# The goal of the `OptimisationProblem` is to be able to return an optimised
# parameterisation, judged according to the provided objective, subject to
# the set of constraints, using a numerical search algorithm defined in the optimiser.
#
# Crucially, it provides an `optimise()` method that returns this optimised
# parameterisation. Subclasses may override this `optimise()` method, but must still
# return an optimised parametrisation when `optimise()` is called
# on the OptimisationProblem.
#
# A `CoilsetOP` is a subclass of `OptimisationProblem`, intended for problems where
# the parameterisation is a `CoilSet` object, which provides some useful additional
# methods for getting and setting coilset state vectors/arrays.
#
# Subclasses of `CoilsetOP`, such as `BoundedCurrentCOP` or `NestedCoilsetPositionCOP`,
# can be made to bind a `CoilsetOP` to a specific objective function, which is often
# useful for performance reasons where some data in the OptimisationProblem
# does not need to be updated at every iteration.
#
# # Example
# We will present an example problem where we take an existing `CoilsetOP`subclass,
# `BoundedCurrentCOP`, and apply some additional constraints that
# must be held during the optimisation.
#
# As the objective function for the `BoundedCurrentCOP` OptimisationProblem is already
# specified, we only need to provide the `coilset` (parameterisation), `Optimiser`,
# `OptimisationConstraints`, and any additional arguments that are used in the
# `BoundedCurrentCOP` objective function.
#
# ### Parametrisation
#
# We first define the parameterisation to be used in the OptimisationProblem.
# This is just the starting `coilset` in this case.

# %%
coil_x = [1.05, 6.85, 6.85, 1.05, 3.2, 5.7, 5.3]
coil_z = [7.85, 4.75, 3.35, 6.0, 8.0, 7.8, 5.50]
coil_dx = [0.45, 0.5, 0.5, 0.3, 0.6, 0.5, 0.25]
coil_dz = [0.5, 0.8, 0.8, 0.8, 0.5, 0.5, 0.5]
currents = [0, 0, 0, 0, 0, 0, 0]

circuits = []
for i in range(len(coil_x)):
    coil = Coil(
        coil_x[i],
        coil_z[i],
        dx=coil_dx[i] / 2,
        dz=coil_dz[i] / 2,
        current=currents[i],
        ctype="PF",
    )
    circuit = SymmetricCircuit(coil)
    circuits.append(circuit)
coilset = CoilSet(*circuits)

# %% [markdown]
#
# ### Optimiser
#
# We next define the `Optimiser` to be used. There is no one-size-fits-all approach
# here, as the best optimiser for a given problem will depend strongly on the
# OptimisationObjective and OptimisationConstraints being applied in the problem.
#
# `Optimiser` is currently only a wrapper for NLOpt based optimisers, and as such,
# the range of algorithms that may be chosen is determined by those available in
# the NLOpt library. Not all algorithms will work for all objectives - some
# require gradient information, for example - and some require additional parameters.
# Check the NLOpt API documentation for more details.
#
# Care must also be taken to ensure the termination criteria for the optimisation
# are suitable. Otherwise, the optimisation may never stop running if tolerances
# are too tight, or may stop early and return poorly optimised states if the
# maximum number of evaluations is too low or tolerances are too large.

# %%
optimiser = Optimiser(
    algorithm_name="COBYLA",
    opt_conditions={"max_eval": 200},
    opt_parameters={"initial_step": 0.03},
)

# %% [markdown]
#
# ### Additional Parameters
#
# `BoundedCurrentCOP` requires two additional parameters for generating arguments for
# its `OptimisationObjective` during its `optimise()` call. The `OptimisationObjective`
# figure of merit for `BoundedCurrentCOP` is the regularised least-squares deviation of a
# provided Equilibrium from a set of magnetic field targets (eg. isoflux targets).
#
# We use a default set of isoflux targets here for this example.

# %%
x_lcfs = np.array([1.0, 1.67, 4.0, 1.73])
z_lcfs = np.array([0, 4.19, 0, -4.19])

lcfs_isoflux = IsofluxConstraint(
    x_lcfs, z_lcfs, ref_x=x_lcfs[2], ref_z=z_lcfs[2], constraint_value=0.1
)

x_lfs = np.array([1.86, 2.53, 3.43, 5.80, 6.70])
z_lfs = np.array([4.80, 5.84, 6.60, 6.71, 6.71])
x_hfs = np.array([1.42, 0.81, 0.62, 0.64, 0.60])
z_hfs = np.array([4.80, 5.38, 6.01, 6.82, 7.34])


x_legs = np.concatenate([x_lfs, x_lfs, x_hfs, x_hfs])
z_legs = np.concatenate([z_lfs, -z_lfs, z_hfs, -z_hfs])

legs_isoflux = IsofluxConstraint(
    x_legs, z_legs, ref_x=x_lcfs[2], ref_z=z_lcfs[2], constraint_value=0.1
)

magnetic_targets = MagneticConstraintSet([lcfs_isoflux, legs_isoflux])
magnetic_core_targets = MagneticConstraintSet([lcfs_isoflux])

# %% [markdown]
#
# We also specify an initial Equilibrium state to be used in the optimisation.
#
# For this we need a Grid and some plasma profiles

# %%
# Intialise some parameters
R_0 = 2.6
Z_0 = 0
B_t = 1.9
I_p = 16e6

r0, r1 = 0.2, 8
z0, z1 = -8, 8
nx, nz = 129, 257
grid = Grid(r0, r1, z0, z1, nx, nz)

# fmt: off
pprime = np.array(
    [
        -850951, -844143, -782311, -714610, -659676, -615987, -572963, -540556, -509991,
        -484261, -466462, -445186, -433472, -425413, -416325, -411020, -410672, -406795,
        -398001, -389309, -378528, -364607, -346119, -330297, -312817, -293764, -267515,
        -261466, -591725, -862663,
    ]
)
ffprime = np.array(
    [
        7.23, 5.89, 4.72, 3.78, 3.02, 2.39, 1.86, 1.43, 1.01, 0.62, 0.33, 0.06, -0.27,
        -0.61, -0.87, -1.07, -1.24, -1.18, -0.83, -0.51, -0.2, 0.08, 0.24, 0.17, 0.13,
        0.1, 0.07, 0.05, 0.15, 0.28,
    ]
)
# fmt: on

profiles = CustomProfile(pprime, ffprime, R_0=R_0, B_0=B_t, I_p=I_p)
eq = Equilibrium(
    coilset,
    grid,
    profiles,
    force_symmetry=True,
    vcontrol=None,
    psi=None,
)

# %% [markdown]
#
# ### Constraints
#
# We next define the list of `OptimisationConstraints` to apply.
# In this case, we wish to apply a constraint to prevent solutions where the
# plasma boundary at the inboard midplane of the plasma is prevented from moving
# inside a provided radius.
#
# We will use `bluemira.equilibria.opt_constraints.current_midplane_constraint`
# as our constraint function here to do this.
#
# ```python
# def current_midplane_constraint(
#     constraint, vector, grad, opt_problem, radius, inboard=True
# ):
#     """
#     Constraint function to constrain the inboard or outboard midplane
#     of the plasma during optimisation.
#
#     Parameters
#     ----------
#     radius: float
#         Toroidal radius at which to constrain the plasma midplane.
#     inboard: bool (default=True)
#         Boolean controlling whether to constrain the inboard (if True) or
#         outboard (if False) side of the plasma midplane.
#     """
#     coilset_state = np.concatenate((opt_problem.x0, opt_problem.z0, vector))
#     opt_problem.set_coilset_state(coilset_state)
#     lcfs = opt_problem.eq.get_LCFS()
#     if inboard:
#         constraint[:] = radius - min(lcfs.x)
#     else:
#         constraint[:] = max(lcfs.x) - radius
#     return constraint
# ```
#
# The first three arguments here are expected by NLOpt, and must always be present in
# constraint functions.
# - constraint
#     - np.array storing constraint information. During the optimisation,
#       `constraint[:]<=0` is considered to represent the constraint being satisfied,
#       and `constraint[:]>0` represents the constraint being violated.
# - vector
#     - np.array representing the state vector that is optimised during the
#       numerical optimisation.
# - grad
#     - np.array representing Jacobian for the constraint function. This must always
#       be present in the arguments, but only needs to be calculated if the `Optimiser`
#       is employing an algorithm that requires derivative information.
#
# The fourth, `opt_problem`, is optional, and provides an interface to the
# `OptimisationProblem` the constraint is applied to. This may be useful for performance
# reasons, where data needed by the `constraint` does not need to be updated every
# iteration of the optimisation. Where possible, explicit arguments should be provided to
# the `OptimisationProblem`, however.
#
# The remaining arguments are explicit arguments that can be passed to the constraint to
# control its behaviour.
#
# This constraint function can be passed to a `OptimisationConstraint` object, along with
# explicit arguments, constraint tolerances, and constraint type, that is used by
# NLOpt when applying the constraint.
#
# User specified constraints can be supplied here, if so desired.

# %%
opt_constraint_funcs = [
    OptimisationConstraint(
        f_constraint=opt_constraint_funcs.current_midplane_constraint,
        f_constraint_args={"eq": eq, "radius": 1.0, "scale": 1e6},
        tolerance=np.array([1e-4]),
        constraint_type="inequality",
    )
]

# %% [markdown]
#
# We now have all the requirements to specify our `CoilsetOP`, and can now initialise it:

# %%
opt_problem = TikhonovCurrentCOP(
    coilset,
    eq,
    magnetic_targets,
    gamma=1e-8,
    max_currents=3.0e7,
    optimiser=optimiser,
    constraints=opt_constraint_funcs,
)

# %% [markdown]
#
# # Iterators
#
# The `CoilsetOP` is only used to optimise the coilset state at fixed plasma psi;
# the Grad-Shafranov equation for the plasma is not guaranteed to still be satisfied
# after the coilset state is optimised.
#
# Picard iteration is therefore used to find a self-consistent solution, employed
# by `Iterator` objects. `PicardCoilsetIterator` specifies a scheme in which a
# Grad-Shafranov iteration is used to update the plasma psi alternate with Coilset
# optimisation at fixed plasma psi until the psi converges for the `Equilbrium`.

# %%
constrained_iterator = PicardIterator(
    eq,
    opt_problem,
    plot=False,
    relaxation=0.3,
    maxiter=400,
    convergence=DudsonConvergence(1e-4),
)

# %% [markdown]
#
# However, a poor initial `Equilibrium` (with corresponding `coilset`) will lead
# to difficulties during the Picard iteration used to find a self-consistent solution.
# It is therefore useful to perform a coarser fast pre-optimisation to try find a
# self-consistent state within the basin of convergence of the
# full constrained `OptimisationProblem`.

# %%
unconstrained_cop = UnconstrainedTikhonovCurrentGradientCOP(
    coilset, eq, magnetic_targets, gamma=1e-8
)
unconstrained_iterator = PicardIterator(
    eq,
    unconstrained_cop,
    plot=False,
    relaxation=0.3,
    convergence=DudsonConvergence(1e-2),
    maxiter=400,
)

# %% [markdown]
#
# # FBE Optimisation
# We have now initialised the necessary objects to perform the optimisation of the
# `coilset` state.
#
# We first plot the initial `Equilibrium` state - as the initial coilset for this
# example has no current in the coils, it is predictably extremely poor!

# %%
f, ax = plt.subplots()
eq.plot(ax=ax)
unconstrained_cop.targets.plot(ax=ax)

# %% [markdown]
#
# ### Pre-optimisation
#
# Constrained optimisation of this poor initial state would be difficult, as
# local optimisers would likely struggle to find the basin of convergence of
# the desired state.
#
# We therefore perform a fast pre-optimisation to get closer to a physical
# state that satisfies our constrained optimisation problem.

# %%
unconstrained_iterator()

f, ax = plt.subplots()
eq.plot(ax=ax)
magnetic_targets.plot(ax=ax)
plt.show()

# %% [markdown]
#
# ### Constrained Optimisation
#
# Now we have a better starting `Equilibrium` for our constrained optimisation

# %%
constrained_iterator()

f, ax = plt.subplots()
eq.plot(ax=ax)
magnetic_targets.plot(ax=ax)
plt.show()
