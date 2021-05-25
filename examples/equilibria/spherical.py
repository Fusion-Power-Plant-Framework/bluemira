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
import pprint
import matplotlib.pyplot as plt
from BLUEPRINT.base.lookandfeel import plot_defaults
from BLUEPRINT.base.file import get_BP_path
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.equilibria.profiles import CustomProfile
from BLUEPRINT.equilibria.gridops import Grid
from BLUEPRINT.equilibria.constraints import (
    SilverSurfer,
    IsofluxConstraint,
    XpointConstraint,
)
from BLUEPRINT.equilibria.coils import Coil, CoilSet
from BLUEPRINT.equilibria.shapes import flux_surface_manickam
from BLUEPRINT.equilibria.equilibrium import Equilibrium
from BLUEPRINT.equilibria.limiter import Limiter
from BLUEPRINT.equilibria.optimiser import Norm2Tikhonov, FBIOptimiser
from BLUEPRINT.equilibria.solve import PicardDeltaIterator, CunninghamConvergence

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

# I use a lot of little objects that the normal user shouldn't see, because I
# still don't know how to do spherical equilibria from scratch properly. In
# future, we should turn this into a parameterised interface, so that it could
# be like in single_null.py, using an AbInitioEquilibriumProblem object as an
# interface. See bottom of file for indicate API use.

R0 = 2.5
Z0 = 0
A = 1.6
kappa = 2.84
delta = 0.44
Bt = 1.87
betap = 1.83
Ip = 21.2e6
indent = 0

# Make a grid
r0, r1 = 0.5, 6.5
z0, z1 = -7.4, 7.4
dr, dz = 0.1, 0.1

nx, nz = int((r1 - r0) // dr), int((z1 - z0) // dz)

grid = Grid(r0, r1, z0, z1, nx, nz)

# Make a "bumper" limiter
limiter = Limiter([(0.6, 0)])

# Make a plasma profile


# Load SCENE fixed boundary equilibrium profiles
# This works fine
folder = get_BP_path("eqdsk", subfolder="data")
name = "stepp.geqdsk"
filename = os.sep.join([folder, name])
profile = CustomProfile.from_eqdsk(filename)


# =============================================================================
# # We can try and constrain some integral values using that shape function
# # but it doesn't really work
# shape_function = profile.shape  # Copy shape function from data loaded above
# profile = BetaIpProfile(betap, Ip, R0, Bt, shape_function)
#
# # Otherwise we can just put a normal Lao poly and see what happens
# # it doesn't work either... plasma pressure pushes it off the grid?
# shape_function = LaoPolynomialFunc([1.1, -0.03, 0])
# profile = BetaIpProfile(betap, Ip, R0, Bt, shape_function)
#
# =============================================================================


# Make a family of constraints

shape = flux_surface_manickam(R0, Z0, R0 / A, kappa, delta, indent, n=100)

clip = np.where(shape.z > -1.5)
shape = Loop(shape.x[clip], z=shape.z[clip])

clip = np.where(shape.z < 1.5)
shape = Loop(shape.x[clip], z=shape.z[clip])

X, Z = shape.x, shape.z


# Divertor legs

x_hfs = np.array([0.6532, 0.6532, 0.6532, 0.7190])
z_hfs = np.array([5.8400, 6.3669, 6.8060, 7.1573])

x_lfs = np.array([2.4072, 2.8755, 3.3813, 3.8496, 4.3179, 4.8612, 5.3669])
z_lfs = np.array([5.2993, 5.7264, 6.0530, 6.2288, 6.3544, 6.4298, 6.5303])

for x, z in zip([x_hfs, x_lfs], [z_hfs, z_lfs]):
    # Mirror constraint
    xx = np.append(x, x)
    zz = np.append(z, -z)
    # Append to IsofluxConstraint
    X = np.append(X, xx)
    Z = np.append(Z, zz)

# Stiffening lines (I don't have numerical vertical stabilisation..)

# x_st = np.array([1.5995, 1.9836, 2.3868, 2.8285, 3.3470])
# z_st = np.array([6.1931, 6.3659, 6.5580, 6.7500, 6.9036])

# x_ldiv = np.array([4.2304, 4.8641, 5.4210, 5.9395])
# z_ldiv = np.array([6.1156, 6.1156, 6.1348, 6.1156])

# x_udiv = np.array([4.2304, 4.8641, 5.4210, 5.9395])
# z_udiv = np.array([7.1156, 7.1156, 7.1348, 7.1156])

# x_add = np.array([4.8641, 5.4210, 5.9395])
# z_add = np.array([6.4, 6.4, 6.4])


constraints = SilverSurfer()

constraints.isoflux = {1: IsofluxConstraint(X, Z, refpoint=[R0 - R0 / A, 0])}

constraints.xpoints = [XpointConstraint(1.725, 4.6), XpointConstraint(1.725, -4.6)]
constraints.X = shape.x
constraints.Z = shape.z

# Make a coilset (AH's case 2: best so far)

coil_x = [1.1, 6.9, 6.9, 1.05, 3.2, 5.7, 0.125, 0.125, 5.3]
coil_z = [7.8, 4.7, 3.3, 6.05, 8.0, 7.8, 3.5, 0.75, 5.5]
coil_dx = [0.45, 0.5, 0.5, 0.3, 0.6, 0.5, 0.15, 0.15, 0.25]
coil_dz = [0.5, 0.8, 0.8, 0.8, 0.5, 0.5, 2.0, 1.5, 0.5]
currents = [7.3, 2.2, -11.09, 4.4, 4.8, 6.5, -5, -5, -4.5]

# Now mirror it

coil_x = coil_x + coil_x
coil_z = np.array(coil_z)
coil_z = np.append(coil_z, -coil_z)
coil_dx = coil_dx + coil_dx
coil_dz = coil_dz + coil_dz

currents = 1e6 * np.array(2 * currents)

coils = []
for i in range(len(coil_x)):
    if coil_x[i] == 0.125:
        ctype = "CS"
    else:
        ctype = "PF"
    coil = Coil(
        coil_x[i],
        coil_z[i],
        dx=coil_dx[i] / 2,
        dz=coil_dz[i] / 2,
        current=currents[i],
        ctype=ctype,
    )
    coils.append(coil)


coilset = CoilSet(coils, R0)

coilset.fix_sizes()  # Normally I don't do this

# Temporarily add a simple plasma coil to get a good starting guess for psi
coilset.add_coil(Coil(R0, Z0, current=Ip, name="plasma_dummy"))

psi = coilset.psi(grid.x, grid.z).copy()

coilset.remove_coil("plasma_dummy")

# Optional, slower, more accurate
# coilset.mesh_coils(0.2)

eq = Equilibrium(
    coilset,
    grid,
    boundary="free",
    vcontrol="virtual",
    limiter=limiter,
    psi=psi,
    Ip=Ip,
    li=None,
)

# Simple unconstrained optimisation
optimiser = Norm2Tikhonov(gamma=1e-4)  # This is still a bit of a magic number..

program = PicardDeltaIterator(
    eq,
    profile,
    constraints,
    optimiser,
    plot=True,
    gif=False,
    relaxation=0.2,
    convergence=CunninghamConvergence(),
    maxiter=200,
)

if interactive:
    next(program)
else:
    program()

    plt.close("all")
    f, ax = plt.subplots()

    coilset.plot(ax)
    constraints.plot(ax)
    limiter.plot(ax)
    eq.plot(ax)

    # But this already quite far off the desired plasma..! Especially in terms of beta_p
    core_results = eq.analyse_plasma()

    print("\n")
    pprint.pprint(core_results)

    folder = get_BP_path("eqdsk", subfolder="data")
    name = "step_v7_format.geqdsk"

    filename = os.sep.join([folder, name])
    eq_agnieszka = Equilibrium.from_eqdsk(filename)

    # eq_agnieszka.plot(ax)

    f, ax = plt.subplots()
    eq.plot_core(ax)

    # Now do a constrained optimisation
    B_max = coilset.get_max_fields()  # Based on NbTi
    # Need to set CS fields very high, because of some complicated limitations in the
    # code
    B_max[-4:] = 50

    # Need to increase default current densities, because they are very high...
    coilset.assign_coil_materials("PF", "Nb3Sn", j_max=30)  # Default would be 16.5
    coilset.assign_coil_materials("CS", "Nb3Sn", j_max=30)
    I_max = coilset.get_max_currents(0)

    # No idea regarding force limits for STs, depends on coil cage structure design
    optimiser = FBIOptimiser(
        B_max,
        PF_Fz_max=200e6,  # MN
        CS_Fz_sum=200e6,  # MN
        CS_Fz_sep=200e6,  # MN
        gamma=1e-10,
    )
    # Have to do this later, because the optimiser is built to having changing coil
    # cross-sections
    optimiser.update_current_constraint(I_max)

# =============================================================================
# program = PicardAbsIterator(eq, profile, constraints, optimiser,
#                             plot=True)
# program()
# =============================================================================

# =============================================================================
# # What I would ideally do... but ST's are weird.
#
# from BLUEPRINT.geometry.parameterisations import PictureFrame
# from BLUEPRINT.equilibria.run import AbInitioEquilibriumProblem
# from BLUEPRINT.equilibria.profiles import LaoPolynomialFunc, DoublePowerFunc
# from BLUEPRINT.equilibria.solve import PicardDeltaIterator, PicardAbsIterator
# from BLUEPRINT.equilibria.profiles import BetaIpProfile
# # Choose a plasma profile parameterisation
# p = LaoPolynomialFunc([2, 2, 1])
#
# # Set up an equilibrium problem
# ST = AbInitioEquilibriumProblem(2.5,
#                                 B0=5.5,
#                                 A=1.9,
#                                 Ip=11e6,
#                                 betap=1.2,
#                                 li=0.8,
#                                 kappa=1.9,
#                                 delta=0.4,
#                                 Xcs=0.2,
#                                 dXcs=0.1,
#                                 tfbnd=TF,
#                                 nPF=6,
#                                 nCS=0,
#                                 eqtype='DN',
#                                 rtype='Normal',
#                                 profile=p, psi=None)
#
# # Do an initial solve
# eqref = ST.solve(plot=True)
#
#
# UP = Loop(x=[6, 12, 12, 6, 6], z=[3, 3, 14.5, 14.5, 3])
# LP = Loop(x=[10, 10, 12, 22, 22, 10], z=[-6, -6, -11, -11, -6, -6])
# EQ = Loop(x=[14, 22, 22, 14, 14], z=[-1.4, -1.4, 1.4, 1.4, -1.4])
#
# # Perform full constrained optimisation of positions and currents
# ST.optimise_positions(1.5*15e6,
#                        350e6,
#                        300e6,
#                        250e6,
#                        None,
#                        0.04,
#                        None,
#                        TF,
#                        [LP, EQ, UP],
#                        CS=False,
#                        plot=True,
#                        gif=False)
# =============================================================================
