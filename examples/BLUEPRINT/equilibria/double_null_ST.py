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
"""
ST equilibrium attempt
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from bluemira.base.look_and_feel import plot_defaults
from BLUEPRINT.base.file import get_BP_path
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.equilibria.profiles import CustomProfile
from BLUEPRINT.equilibria.gridops import Grid
from BLUEPRINT.equilibria.constraints import (
    MagneticConstraintSet,
    IsofluxConstraint,
)
from BLUEPRINT.equilibria.coils import Coil, CoilSet, SymmetricCircuit
from BLUEPRINT.equilibria.equilibrium import Equilibrium
from BLUEPRINT.equilibria.optimiser import Norm2Tikhonov
from BLUEPRINT.equilibria.solve import PicardDeltaIterator
from BLUEPRINT.equilibria.eqdsk import EQDSKInterface

# Clean up and make plots look good
plt.close("all")
plot_defaults()

# Interactive mode allows manual step through of the solver iterations
# Run the script in ipython or with python -i
# When the script completes do as below
"""
next(program)
"""
# or
"""
program.iterate_once()
"""
# Each call of next(program) or program.iterate_once() will perform an iteration of
# the solver, while iterate_once will handle the StopIteration condition on convergence
# - the next(program) method will raise the usual StopIteration exception on convergence.
# After each iteration you can check the properties of the solver e.g. program.psi
interactive = False

# Mirroring work by Agnieszka Hudoba on STEP using FIESTA, 19/11/2019

R0 = 2.5
Z0 = 0
A = 1.6
kappa = 2.8
delta = 0.52
Bt = 1.87
betap = 2.7
Ip = 16.5e6


# Make a grid

r0, r1 = 0.2, 8
z0, z1 = -8, 8
nx, nz = 129, 257

grid = Grid(r0, r1, z0, z1, nx, nz)

# Make a plasma profile

# Import example STEP profile from eqdsk

# Import eqdsk
folder = get_BP_path("eqdsk", subfolder="data")
name = "jetto.eqdsk_out"
filename = os.sep.join([folder, name])
profile = CustomProfile.from_eqdsk(filename)

reader = EQDSKInterface()
jettoequilibria = reader.read(filename)

# Initialising LCFS

jettolcfs = Loop(x=jettoequilibria["xbdry"], z=jettoequilibria["zbdry"])

i = [np.min(jettolcfs.x), jettolcfs.z[np.argmin(jettolcfs.x)]]
o = [np.max(jettolcfs.x), jettolcfs.z[np.argmax(jettolcfs.x)]]
u = [jettolcfs.x[np.argmax(jettolcfs.z)], np.max(jettolcfs.z)]
ll = [jettolcfs.x[np.argmin(jettolcfs.z)], np.min(jettolcfs.z)]

# Make a family of constraints
constraint_set = MagneticConstraintSet()

# Trying all sorts of isoflux constraints

X = np.array([i[0], u[0], o[0], ll[0]])
Z = np.array([i[1], u[1], o[1], ll[1]])


# Divertor legs
# Points chosen to replicate divertor legs in AH's FIESTA demo
x_hfs = np.array(
    [1.42031, 1.057303, 0.814844, 0.669531, 0.621094, 0.621094, 0.645312, 0.596875]
)
z_hfs = np.array([4.79844, 5.0875, 5.37656, 5.72344, 6.0125, 6.6484, 6.82188, 7.34219])

x_lfs = np.array([1.85625, 2.24375, 2.53438, 2.89766, 3.43047, 4.27813, 5.80391, 6.7])
z_lfs = np.array(
    [4.79844, 5.37656, 5.83906, 6.24375, 6.59063, 6.76406, 6.70625, 6.70625]
)

xdiv = np.concatenate([x_lfs, x_lfs, x_hfs, x_hfs])
zdiv = np.concatenate([z_lfs, -z_lfs, z_hfs, -z_hfs])

constraint_set.constraints = [
    IsofluxConstraint(X, Z, ref_x=o[0], ref_z=o[1]),
    IsofluxConstraint(xdiv, zdiv, ref_x=o[0], ref_z=o[1]),
]


# Make a coilset
# No CS coils needed for the equilibrium (Last 2 coils are CS below)

# EFIT coils
coil_x = [1.1, 6.9, 6.9, 1.05, 3.2, 5.7, 5.3]
coil_z = [7.8, 4.7, 3.3, 6.05, 8.0, 7.8, 5.55]
coil_dx = [0.45, 0.5, 0.5, 0.3, 0.6, 0.5, 0.25]
coil_dz = [0.5, 0.8, 0.8, 0.8, 0.5, 0.5, 0.5]
currents = [0, 0, 0, 0, 0, 0, 0]

coils = []
for i in range(len(coil_x)):
    if coil_x[i] == 0.125:
        ctype = "CS"
    else:
        ctype = "PF"
    coil = SymmetricCircuit(
        coil_x[i],
        coil_z[i],
        dx=coil_dx[i] / 2,
        dz=coil_dz[i] / 2,
        current=currents[i],
        ctype=ctype,
    )
    coils.append(coil)


coilset = CoilSet(coils, R0)
coilset_temp = CoilSet(coils, R0)

# Temporarily add a simple plasma coil to get a good starting guess for psi
coilset_temp.add_coil(
    Coil(R0 + 0.5, Z0, dx=0, dz=0, current=Ip, name="plasma_dummy", control=False)
)

eq = Equilibrium(
    coilset_temp,
    grid,
    boundary="free",
    vcontrol=None,
    limiter=None,
    psi=None,
    Ip=0,
    li=None,
)
constraint_set(eq)
optimiser = Norm2Tikhonov(gamma=1e-7)  # This is still a bit of a magic number..
currents = optimiser(eq, constraint_set)

coilset_temp.set_control_currents(currents)
coilset.set_control_currents(currents)

psi = coilset_temp.psi(grid.x, grid.z).copy()


# Optional, slower, more accurate
# coilset.mesh_coils(0.2)


eq = Equilibrium(
    coilset,
    grid,
    boundary="free",
    vcontrol=None,
    psi=psi,
    Ip=Ip,
    li=None,
)
eq.plot()
plt.close("all")


# Simple unconstrained optimisation
optimiser = Norm2Tikhonov(gamma=1e-8)  # This is still a bit of a magic number..

program = PicardDeltaIterator(
    eq,
    profile,  # jetto
    constraint_set,
    optimiser,
    plot=True,
    gif=False,
    relaxation=0.3,
    # convergence=CunninghamConvergence(),
    maxiter=400,
)

if interactive:
    next(program)
else:
    program()
    plt.close("all")

    f, ax = plt.subplots()
    eq.plot(ax=ax)
    constraint_set.plot(ax=ax)
