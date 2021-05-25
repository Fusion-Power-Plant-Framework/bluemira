# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.
"""
ST equilibrium attempt
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from BLUEPRINT.base.lookandfeel import plot_defaults
from BLUEPRINT.base.file import get_BP_path
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.equilibria.profiles import CustomProfile
from BLUEPRINT.equilibria.gridops import Grid
from BLUEPRINT.equilibria.constraints import (
    SilverSurfer,
    IsofluxConstraint,
)
from BLUEPRINT.equilibria.coils import Coil, CoilSet, Circuit
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
constraints = SilverSurfer()

# Trying all sorts of isoflux constraints

X = [i[0], u[0], o[0], ll[0]]
Z = [i[1], u[1], o[1], ll[1]]


# Divertor legs

xdiv = []
zdiv = []
# Points chosen to replicate divertor legs in AH's FIESTA demo
x_lfs = np.array([4.27813])
z_lfs = np.array([6.76406])

for x, z in zip([x_lfs], [z_lfs]):
    # Mirror constraint
    xx = np.append(x, x)
    zz = np.append(z, -z)
    # Append to IsofluxConstraint
    xdiv = np.append(xdiv, xx)
    zdiv = np.append(zdiv, zz)


constraints.isoflux = {
    1: IsofluxConstraint(X, Z, refpoint=[o[0], o[1]]),
    2: IsofluxConstraint(xdiv, zdiv, refpoint=[o[0], o[1]]),
}


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
    coil = Circuit(
        coil_x[i],
        coil_z[i],
        dx=coil_dx[i] / 2,
        dz=coil_dz[i] / 2,
        current=currents[i],
        ctype=ctype,
    )
    coils.append(coil)


coilset = CoilSet(coils, R0)

# Temporarily add a simple plasma coil to get a good starting guess for psi
coilset.add_coil(
    Coil(R0 + 0.5, Z0, dx=0, dz=0, current=Ip, name="plasma_dummy", control=False)
)
coilset.n_coils += 1

eq = Equilibrium(
    coilset,
    grid,
    boundary="free",
    vcontrol=None,
    limiter=None,
    psi=None,
    Ip=0,
    li=None,
)
constraints(eq)
constraints.build_A()
constraints.build_b()
optimiser = Norm2Tikhonov(gamma=1e-7)  # This is still a bit of a magic number..
currents = optimiser(eq, constraints)

coilset.set_control_currents(currents)

psi = coilset.psi(grid.x, grid.z).copy()

coilset.remove_coil("plasma_dummy")

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
    constraints,
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
    constraints.plot(ax=ax)
