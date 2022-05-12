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

# # EU-DEMO 2017 reference breakdown and equilibrium benchmark

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
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
    PsiBoundaryConstraint,
    coil_field_constraints,
)
from bluemira.equilibria.opt_problems import (
    MinimalCurrentsCOP,
    MinimalErrorCOP,
    UnconstrainedMinimalErrorCOP,
)
from bluemira.equilibria.profiles import CustomProfile
from bluemira.equilibria.solve_new import DudsonConvergence, PicardIterator
from bluemira.utilities.opt_problems import OptimisationConstraint
from bluemira.utilities.optimiser import Optimiser

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

# Make the same CoilSet as CREATE

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
coilset.mesh_coils(0.3)


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
tau_flattop = 2 * 3600.0
v_burn = 4.220e-2  # V
c_ejima = 0.3


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

grid = Grid(3.0, 13.0, -10.0, 10.0, 65, 65)

profiles = CustomProfile(
    np.array([86856, 86506, 84731, 80784, 74159, 64576, 52030, 36918, 20314, 4807, 0.0]),
    -np.array(
        [0.125, 0.124, 0.122, 0.116, 0.106, 0.093, 0.074, 0.053, 0.029, 0.007, 0.0]
    ),
    R_0=R_0,
    B_0=B_0,
    Ip=I_p,
)

eq = Equilibrium(coilset, grid, psi=None, profiles=profiles, Ip=I_p, RB0=[R_0, B_0])

opt_problem = UnconstrainedMinimalErrorCOP(
    eq, MagneticConstraintSet([psi_boundary, x_point]), gamma=1e-7
)

program = PicardIterator(
    eq, profiles, opt_problem, I_not_dI=False, fixed_coils=True, relaxation=0.2
)
program()


field_constraints = OptimisationConstraint(
    coil_field_constraints,
    f_constraint_args={"eq": eq, "B_max": eq.coilset.get_max_fields(), "scale": 1e6},
    tolerance=1e-6 * np.ones(11),
)

opt_problem = MinimalCurrentsCOP(
    eq,
    Optimiser("SLSQP", opt_conditions={"max_eval": 2000, "ftol_rel": 1e-6}),
    max_currents=coilset.get_max_currents(0.0),
    constraints=[psi_boundary, x_point, field_constraints],
)

program = PicardIterator(
    eq,
    profiles,
    opt_problem,
    I_not_dI=True,
    fixed_coils=True,
    convergence=DudsonConvergence(1e-4),
    relaxation=0.3,
)
program()

f, ax = plt.subplots()
eq.plot(ax=ax)
eq.coilset.plot(ax=ax)


opt_problem = MinimalErrorCOP(
    eq,
    targets=MagneticConstraintSet([psi_boundary, x_point]),
    gamma=1e-8,
    optimiser=Optimiser("SLSQP", opt_conditions={"max_eval": 2000, "ftol_rel": 1e-6}),
    max_currents=coilset.get_max_currents(0.0),
    constraints=[field_constraints],
)

program = PicardIterator(
    eq,
    profiles,
    opt_problem,
    I_not_dI=True,
    fixed_coils=True,
    convergence=DudsonConvergence(1e-4),
    relaxation=0.3,
)
program()
