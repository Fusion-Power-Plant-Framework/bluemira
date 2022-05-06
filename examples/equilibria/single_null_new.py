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

import copy
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from IPython import get_ipython

from bluemira.base.file import get_bluemira_path
from bluemira.base.look_and_feel import bluemira_print
from bluemira.display import plot_defaults
from bluemira.equilibria.coils import Coil, CoilSet
from bluemira.equilibria.eq_constraints import AutoConstraints
from bluemira.equilibria.equilibrium import Breakdown, Equilibrium
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.opt_constraints import (
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
)
from bluemira.equilibria.opt_problems import (
    MinimalCurrentsCOP,
    UnconstrainedMinimalErrorCOP,
)
from bluemira.equilibria.optimiser import BreakdownOptimiser, FBIOptimiser
from bluemira.equilibria.physics import calc_beta_p_approx, calc_li, calc_psib
from bluemira.equilibria.profiles import CustomProfile
from bluemira.equilibria.solve_new import DudsonConvergence, PicardBaseIterator
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
# coilset.mesh_coils(0.3)


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
    np.array(sof_xbdry)[::5],
    np.array(sof_zbdry)[::5],
    sof_xbdry[0],
    sof_zbdry[0],
    tolerance=1e-3,
    target_value=0.5,
)

xp_idx = np.argmin(sof_zbdry)
x_point = FieldNullConstraint(
    sof_xbdry[xp_idx], sof_zbdry[xp_idx], tolerance=1e-3, constraint_type="inequality"
)

grid = Grid(3.0, 13.0, -10.0, 10.0, 65, 65)


def init_equilibrium(grid, coilset):
    """
    Create an initial guess for the Equilibrium state.
    Temporarily add a simple plasma coil to get a good starting guess for psi.
    """
    coilset_temp = copy.deepcopy(coilset)

    coilset_temp.add_coil(
        Coil(
            R_0 + 0.5,
            0.0,
            dx=0.5,
            dz=0.5,
            current=I_p,
            name="plasma_dummy",
            control=False,
        )
    )

    eq = Equilibrium(
        coilset_temp,
        grid,
        force_symmetry=False,
        limiter=None,
        psi=None,
        Ip=0,
        li=None,
    )
    constraint_set = MagneticConstraintSet([isoflux, x_point])
    constraint_set(eq)
    optimiser = UnconstrainedMinimalErrorCOP(eq, constraint_set, gamma=1e-7)
    coilset_temp = optimiser()

    coilset.set_control_currents(coilset_temp.get_control_currents())

    psi = coilset_temp.psi(grid.x, grid.z).copy()
    return psi


rho = np.linspace(0, 1, 30)
profiles = CustomProfile(
    np.array(
        [
            86856.15730491,
            86798.69790167,
            86506.2865987,
            85850.44673834,
            84731.28548257,
            83065.672385,
            80784.08049904,
            77829.90547563,
            74159.99618766,
            69746.01988715,
            64576.59722565,
            58660.327347,
            52030.0139464,
            44748.69633575,
            36918.64903883,
            28695.74759472,
            20314.75855847,
            12141.02112672,
            4807.4423105,
            0.0,
        ]
    ),
    np.array(
        [
            -0.12515916,
            -0.12507636,
            -0.124655,
            -0.12370994,
            -0.12209723,
            -0.1196971,
            -0.11640934,
            -0.11215239,
            -0.10686407,
            -0.10050356,
            -0.09305446,
            -0.08452915,
            -0.07497491,
            -0.06448258,
            -0.05319953,
            -0.04135039,
            -0.02927344,
            -0.01749513,
            -0.00692749,
            -0.0,
        ]
    ),
    R_0=R_0,
    B_0=B_0,
    Ip=I_p,
)

psi = init_equilibrium(grid, coilset)
eq = Equilibrium(coilset, grid, psi=psi, profiles=profiles, Ip=I_p, RB0=[R_0, B_0])

opt_problem = UnconstrainedMinimalErrorCOP(
    eq, MagneticConstraintSet([isoflux, x_point]), gamma=1e-7
)

program = PicardBaseIterator(
    eq, profiles, opt_problem, I_not_dI=False, fixed_coils=True, relaxation=0.2
)
program()

opt_problem = MinimalCurrentsCOP(
    eq,
    Optimiser("SLSQP", opt_conditions={"max_eval": 2000, "ftol_rel": 1e-6}),
    max_currents=coilset.get_max_currents(0.0),
    constraints=[isoflux, x_point],
)

program = PicardBaseIterator(
    eq,
    profiles,
    opt_problem,
    I_not_dI=True,
    fixed_coils=True,
    convergence=DudsonConvergence(1e-4),
    relaxation=0.3,
)
program()

# %%
