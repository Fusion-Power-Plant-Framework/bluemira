# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2022 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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
Attempt at recreating the EU-DEMO 2017 reference equilibria from a known coilset.
"""

# %%[markdown]

# An example on how to produce an equilibrium from a known coilset, profiles, and
# plasma shape.

# %%

import json
import os

import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython

from bluemira.base.file import get_bluemira_path
from bluemira.display import plot_defaults
from bluemira.equilibria.coils import Coil, CoilSet
from bluemira.equilibria.equilibrium import Equilibrium
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
    MinimalCurrentCOP,
    TikhonovCurrentCOP,
    UnconstrainedTikhonovCurrentGradientCOP,
)
from bluemira.equilibria.profiles import CustomProfile
from bluemira.equilibria.solve import DudsonConvergence, PicardIterator
from bluemira.utilities.optimiser import Optimiser

plot_defaults()

try:
    get_ipython().run_line_magic("matplotlib", "qt")
except AttributeError:
    pass

# %%[markdown]

# First let's create our coilset.

# %%
x = [5.4, 14.0, 17.75, 17.75, 14.0, 7.0, 2.77, 2.77, 2.77, 2.77, 2.77]
z = [9.26, 7.9, 2.5, -2.5, -7.9, -10.5, 7.07, 4.08, -0.4, -4.88, -7.86]
dx = [0.6, 0.7, 0.5, 0.5, 0.7, 1.0, 0.4, 0.4, 0.4, 0.4, 0.4]
dz = [0.6, 0.7, 0.5, 0.5, 0.7, 1.0, 2.99 / 2, 2.99 / 2, 5.97 / 2, 2.99 / 2, 2.99 / 2]

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
        control=True,
        name=f"{ctype}_{j}",
    )
    coils.append(coil)
    j += 1

coilset = CoilSet(coils)

# Assign current density and peak field constraints
coilset.assign_coil_materials("CS", j_max=16.5, b_max=12.5)
coilset.assign_coil_materials("PF", j_max=12.5, b_max=11.0)
coilset.fix_sizes()

# %%[markdown]

# Now, we set up our grid, equilibrium, and profiles

# %%

# Machine parameters
I_p = 19.07e6  # A
R_0 = 8.938
B_0 = 4.8901  # T

grid = Grid(3.0, 13.0, -10.0, 10.0, 65, 65)

profiles = CustomProfile(
    np.array([86856, 86506, 84731, 80784, 74159, 64576, 52030, 36918, 20314, 4807, 0.0]),
    -np.array(
        [0.125, 0.124, 0.122, 0.116, 0.106, 0.093, 0.074, 0.053, 0.029, 0.007, 0.0]
    ),
    R_0=R_0,
    B_0=B_0,
    I_p=I_p,
)

eq = Equilibrium(coilset, grid, profiles, psi=None)

# %%[markdown]

# Now we need to specify some constraints on the plasma

# We'll load up a known plasma boundary and use that to specify some constraints on the
# plasma

# %%
path = get_bluemira_path("equilibria", subfolder="examples")
name = "EUDEMO_2017_CREATE_SOF_separatrix.json"
filename = os.sep.join([path, name])
with open(filename, "r") as file:
    data = json.load(file)

sof_xbdry = data["xbdry"]
sof_zbdry = data["zbdry"]

isoflux = IsofluxConstraint(
    np.array(sof_xbdry)[::10],
    np.array(sof_zbdry)[::10],
    sof_xbdry[0],
    sof_zbdry[0],
    tolerance=1e-3,
    constraint_value=0.5,  # Difficult to choose...
)

psi_boundary = PsiBoundaryConstraint(
    np.array(sof_xbdry)[::10],
    np.array(sof_zbdry)[::10],
    100 / (2 * np.pi),
    tolerance=1.0,
)

xp_idx = np.argmin(sof_zbdry)
x_point = FieldNullConstraint(
    sof_xbdry[xp_idx], sof_zbdry[xp_idx], tolerance=1e-3, constraint_type="inequality"
)

# %%[markdown]

# It's often very useful to solve an unconstrained optimised problem in order to get
# an initial guess for the equilibrium result.

# This is done by using the magnetic constraints in a "set" for which the error is then
# minimised with an L2 norm and a Tikhonov regularisation on the currents.

# We can use this to optimise the current gradients during the solution of the
# equilibrium until convergence.

# %%

opt_problem = UnconstrainedTikhonovCurrentGradientCOP(
    coilset, eq, MagneticConstraintSet([psi_boundary, x_point]), gamma=1e-7
)

program = PicardIterator(eq, opt_problem, fixed_coils=True, relaxation=0.2)
program()


# %%[markdown]

# Now say we want to use bounds on our current vector, and that we want to solve a
# constrained optimisation problem.

# We can minimise the error on our target set with some bounds on the current vector,
# some additional constraints (e.g. on the field in the coils), and solve a new
# optimisation problem, using the previously converged equilibrium as a starting point.

# Note that here we are optimising the current vector and not the current gradient
# vector.

# %%

field_constraints = CoilFieldConstraints(
    eq.coilset, eq.coilset.get_max_fields(), tolerance=1e-6
)

PF_Fz_max = 450
CS_Fz_sum_max = 300
CS_Fz_sep_max = 250
force_constraints = CoilForceConstraints(
    eq.coilset,
    PF_Fz_max=PF_Fz_max,
    CS_Fz_sum_max=CS_Fz_sum_max,
    CS_Fz_sep_max=CS_Fz_sep_max,
    tolerance=1e-6,
)


opt_problem = TikhonovCurrentCOP(
    coilset,
    eq,
    targets=MagneticConstraintSet([psi_boundary, x_point]),
    gamma=1e-8,
    optimiser=Optimiser("SLSQP", opt_conditions={"max_eval": 2000, "ftol_rel": 1e-6}),
    max_currents=coilset.get_max_currents(0.0),
    constraints=[field_constraints, force_constraints],
)

program = PicardIterator(
    eq,
    opt_problem,
    fixed_coils=True,
    convergence=DudsonConvergence(1e-4),
    relaxation=0.3,
)
program()


# %%[markdown]

# Now let's say we don't actually want to minimise the error, but we want to minimise the
# coil currents, and use the constraints that we specified above as actual constraints
# in the optimisation problem (rather than in the objective function as above)

# %%

opt_problem = MinimalCurrentCOP(
    coilset,
    eq,
    Optimiser("SLSQP", opt_conditions={"max_eval": 2000, "ftol_rel": 1e-6}),
    max_currents=coilset.get_max_currents(0.0),
    constraints=[psi_boundary, x_point, field_constraints, force_constraints],
)

program = PicardIterator(
    eq,
    opt_problem,
    fixed_coils=True,
    convergence=DudsonConvergence(1e-4),
    relaxation=0.3,
)
program()

f, ax = plt.subplots()
eq.plot(ax=ax)
eq.coilset.plot(ax=ax)
