# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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

import numpy as np
import matplotlib.pyplot as plt
from bluemira.base.look_and_feel import plot_defaults
from bluemira.geometry._deprecated_loop import Loop
from bluemira.equilibria.profiles import CustomProfile, LaoPolynomialFunc
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.constraints import (
    MagneticConstraintSet,
    IsofluxConstraint,
)
from bluemira.equilibria.coils import Coil, CoilSet, SymmetricCircuit
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimiser import Norm2Tikhonov
from bluemira.equilibria.solve import PicardDeltaIterator

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

# Intialise some parameters

R0 = 2.6
Z0 = 0
A = 1.6
kappa = 2.75
delta = 0.5
Bt = 1.9
betap = 2.6
Ip = 16e6


# Make a grid

r0, r1 = 0.2, 8
z0, z1 = -8, 8
nx, nz = 129, 257

grid = Grid(r0, r1, z0, z1, nx, nz)


# Set up a custom profile object

pprime = LaoPolynomialFunc([3.65, -9.72, 13.2])
ffprime = LaoPolynomialFunc([0.96, -4.44, 5.05])

pprime = np.array(
    [
        -850951.204,
        -844143.13017241,
        -782311.79834483,
        -714610.93044828,
        -659676.93875862,
        -615987.80786207,
        -572963.3357931,
        -540556.27062069,
        -509991.1792069,
        -484261.4017931,
        -466462.60696552,
        -445186.63089655,
        -433472.80337931,
        -425413.968,
        -416325.90489655,
        -411020.49496552,
        -410672.2012069,
        -406795.05444828,
        -398001.66789655,
        -389309.11858621,
        -378528.69386207,
        -364607.59772414,
        -346119.5187931,
        -330297.40131034,
        -312817.87410345,
        -293764.66482759,
        -267515.93182759,
        -261466.16955172,
        -591725.6427931,
        -862663.326,
    ]
)
ffprime = np.array(
    [
        7.22630109,
        5.89253475,
        4.71921846,
        3.78440677,
        3.01944557,
        2.39290352,
        1.85861002,
        1.43177377,
        1.00779937,
        0.6150774,
        0.32635056,
        0.05879146,
        -0.27435619,
        -0.60864614,
        -0.86766641,
        -1.07292296,
        -1.23940376,
        -1.17988594,
        -0.83336877,
        -0.50809497,
        -0.20038229,
        0.08219129,
        0.23538989,
        0.17365059,
        0.13011496,
        0.09814434,
        0.06616453,
        0.04913329,
        0.14945902,
        0.28081309,
    ]
)

profile = CustomProfile(pprime, ffprime, R_0=R0, B_0=Bt, Ip=Ip)

# Make a family of constraints

x_lcfs = np.array([1.0, 1.67, 4.0, 1.73])
z_lcfs = np.array([0, 4.19, 0, -4.19])

lcfs_isoflux = IsofluxConstraint(x_lcfs, z_lcfs, ref_x=x_lcfs[2], ref_z=z_lcfs[2])

x_lfs = np.array([1.86, 2.24, 2.53, 2.90, 3.43, 4.28, 5.80, 6.70])
z_lfs = np.array([4.80, 5.38, 5.84, 6.24, 6.60, 6.76, 6.71, 6.71])
x_hfs = np.array([1.42, 1.06, 0.81, 0.67, 0.62, 0.62, 0.64, 0.60])
z_hfs = np.array([4.80, 5.09, 5.38, 5.72, 6.01, 6.65, 6.82, 7.34])

x_legs = np.concatenate([x_lfs, x_lfs, x_hfs, x_hfs])
z_legs = np.concatenate([z_lfs, -z_lfs, z_hfs, -z_hfs])

legs_isoflux = IsofluxConstraint(x_legs, z_legs, ref_x=x_lcfs[2], ref_z=z_lcfs[2])

constraint_set = MagneticConstraintSet([lcfs_isoflux, legs_isoflux])


# Make a coilset
# No CS coils needed for the equilibrium (Last 2 coils are CS below)

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


coilset = CoilSet(circuits)
coilset_temp = CoilSet(circuits)

# Temporarily add a simple plasma coil to get a good starting guess for psi
coilset_temp.add_coil(
    Coil(R0 + 0.5, Z0, dx=0, dz=0, current=-5 * Ip, name="plasma_dummy", control=False)
)

eq = Equilibrium(
    coilset_temp,
    grid,
    force_symmetry=True,
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


# Set up an equilibrium problem and solve it

eq = Equilibrium(
    coilset,
    grid,
    force_symmetry=True,
    vcontrol=None,
    psi=psi,
    Ip=Ip,
    li=None,
)
eq.plot()
plt.show()
# raise ValueError
# plt.close("all")


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
