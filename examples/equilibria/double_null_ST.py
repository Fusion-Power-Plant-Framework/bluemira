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

# %%[markdown]

# # Script to demonstrate optimisation of coilset of a double null equilibrium

# # Imports

# Import necessary Equilbrium module definitions.

# %%

import argparse
import copy

import matplotlib.pyplot as plt
import numpy as np

from bluemira.display.auto_config import plot_defaults
from bluemira.equilibria.coils import Coil, CoilSet, SymmetricCircuit
from bluemira.equilibria.eq_constraints import IsofluxConstraint, MagneticConstraintSet
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.opt_problems import (
    BoundedCurrentCOP,
    CoilsetPositionCOP,
    NestedCoilsetPositionCOP,
    UnconstrainedCurrentCOP,
)
from bluemira.equilibria.optimiser import Norm2Tikhonov
from bluemira.equilibria.profiles import CustomProfile
from bluemira.equilibria.solve import PicardCoilsetIterator, PicardDeltaIterator
from bluemira.geometry._deprecated_loop import Loop
from bluemira.utilities.optimiser import Optimiser

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
Bt = 1.9
Ip = 16e6

# %%[markdown]

# # Input Definitions

# ## Grid

# %%


def init_grid():
    """
    Create the grid for the FBE solver.
    """
    r0, r1 = 0.2, 8
    z0, z1 = -8, 8
    nx, nz = 129, 257
    grid = Grid(r0, r1, z0, z1, nx, nz)
    return grid


# %%[markdown]

# ## Plasma Profiles

# %%


def init_profile():
    """
    Create the plasma profiles for the FBE solver.
    """
    pprime = np.array(
        [
            -850951,
            -844143,
            -782311,
            -714610,
            -659676,
            -615987,
            -572963,
            -540556,
            -509991,
            -484261,
            -466462,
            -445186,
            -433472,
            -425413,
            -416325,
            -411020,
            -410672,
            -406795,
            -398001,
            -389309,
            -378528,
            -364607,
            -346119,
            -330297,
            -312817,
            -293764,
            -267515,
            -261466,
            -591725,
            -862663,
        ]
    )
    ffprime = np.array(
        [
            7.23,
            5.89,
            4.72,
            3.78,
            3.02,
            2.39,
            1.86,
            1.43,
            1.01,
            0.62,
            0.33,
            0.06,
            -0.27,
            -0.61,
            -0.87,
            -1.07,
            -1.24,
            -1.18,
            -0.83,
            -0.51,
            -0.2,
            0.08,
            0.24,
            0.17,
            0.13,
            0.1,
            0.07,
            0.05,
            0.15,
            0.28,
        ]
    )
    profile = CustomProfile(pprime, ffprime, R_0=R0, B_0=Bt, Ip=Ip)
    return profile


# %%[markdown]

# ## Magnetic Field Targets

# %%


def init_targets():
    """
    Create the set of constraints for the FBE solver.
    """
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
    core_constraints = MagneticConstraintSet([lcfs_isoflux])
    return constraint_set, core_constraints


# %%[markdown]

# ## Initial CoilSet

# %%


def init_coilset():
    """
    Create the initial coilset.
    """
    # Make a coilset
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
    return coilset


# %%[markdown]

# ## Initial allowed PF regions (if needed)

# %%


def init_pfregions(coilset):
    """
    Initialises regions in which coil position optimisation will be limited to.
    """
    max_coil_shifts = {
        "x_shifts_lower": -1.0,
        "x_shifts_upper": 1.0,
        "z_shifts_lower": -1.0,
        "z_shifts_upper": 1.0,
    }

    pfregions = {}
    for coil in coilset._ccoils:
        xu = coil.x + max_coil_shifts["x_shifts_upper"]
        xl = coil.x + max_coil_shifts["x_shifts_lower"]
        zu = coil.z + max_coil_shifts["z_shifts_upper"]
        zl = coil.z + max_coil_shifts["z_shifts_lower"]

        rect = Loop(x=[xl, xu, xu, xl, xl], z=[zl, zl, zu, zu, zl])

        pfregions[coil.name] = rect
    return pfregions


# %%[markdown]

# ## Initial Equilibrium

# %%


def init_equilibrium(grid, coilset, constraint_set):
    """
    Create an initial guess for the Equilibrium state.
    Temporarily add a simple plasma coil to get a good starting guess for psi.
    """
    coilset_temp = copy.deepcopy(coilset)

    coilset_temp.add_coil(
        Coil(
            R0 + 0.5, Z0, dx=0.5, dz=0.5, current=Ip, name="plasma_dummy", control=False
        )
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
    optimiser = UnconstrainedCurrentCOP(coilset_temp, eq, constraint_set, gamma=1e-7)
    coilset_temp = optimiser()

    coilset.set_control_currents(coilset_temp.get_control_currents())

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
    return eq


# %%[markdown]

# # Optimisation

# ## Handle to Iterator call

# %%


def optimise_fbe(program):
    """
    Run the iterator to optimise the FBE.
    """
    if interactive:
        next(program)
    else:
        program()
        plt.close("all")

        f, ax = plt.subplots()
        program.eq.plot(ax=ax)
        program.constraints.plot(ax=ax)
    return


# %%[markdown]

# ## Specification of initial coarse optimisation routine

# %%


def pre_optimise(eq, profile, constraint_set):
    """
    Run a simple unconstrained optimisation to improve the
    initial equilibrium for the main optimiser.
    """
    optimiser = UnconstrainedCurrentCOP(eq.coilset, eq, constraint_set, gamma=1e-8)

    program = PicardCoilsetIterator(
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

    eq = optimise_fbe(program)
    return eq


# %%[markdown]

# ## Specification of primary CoilSet OptimisationProblem to solve

# %%


def set_coilset_optimiser(
    coilset,
    eq,
    targets,
    optimiser_name,
    optimisation_options,
    suboptimiser_name="BoundedCurrentCOP",
    suboptimisation_options=None,
):
    """
    Create the optimiser to be used to optimise the coilset.
    """
    pfregions = init_pfregions(coilset)
    if optimiser_name in ["Norm2Tikhonov"]:
        optimiser = Norm2Tikhonov(**optimisation_options)
    if optimiser_name in ["UnconstrainedCurrentCOP"]:
        optimiser = UnconstrainedCurrentCOP(coilset, eq, targets, **optimisation_options)
    elif optimiser_name in ["BoundedCurrentCOP"]:
        optimiser = BoundedCurrentCOP(coilset, eq, targets, **optimisation_options)
    elif optimiser_name in ["CoilsetPositionCOP"]:
        optimiser = CoilsetPositionCOP(
            coilset, eq, targets, pfregions=pfregions, **optimisation_options
        )
    elif optimiser_name in ["NestedCoilsetPositionCOP"]:
        sub_optimiser = set_coilset_optimiser(
            coilset,
            eq,
            targets,
            optimiser_name=suboptimiser_name,
            optimisation_options=suboptimisation_options,
        )
        optimiser = NestedCoilsetPositionCOP(
            sub_optimiser, eq, targets, pfregions=pfregions, **optimisation_options
        )
    return optimiser


# %%[markdown]

# ## Specification of Equilibrium Iterator

# %%


def set_iterator(eq, profile, constraint_set, optimiser):
    """
    Create the iterator to be used to solve the FBE.
    """
    optimiser_name = type(optimiser).__name__
    iterator_args = (eq, profile, constraint_set, optimiser)
    iterator_kwargs = {"plot": True, "gif": False, "relaxation": 0.3, "maxiter": 400}

    if optimiser_name in [
        "BoundedCurrentCOP",
        "CoilsetPositionCOP",
        "NestedCoilsetPositionCOP",
        "UnconstrainedCurrentCOP",
    ]:
        program = PicardCoilsetIterator(*iterator_args, **iterator_kwargs)
    else:
        program = PicardDeltaIterator(*iterator_args, **iterator_kwargs)

    return program


# %%[markdown]

# ## Selection of OptimisationProblem specific options

# %%


def default_optimiser_options(optimiser_name):
    """
    Specifies default optimiser options.
    """
    options = {"optimiser_name": optimiser_name}
    if optimiser_name in ["Norm2Tikhonov", "UnconstrainedCurrentCOP"]:
        options["optimisation_options"] = {"gamma": 1e-8}
    elif optimiser_name in ["BoundedCurrentCOP"]:
        options["optimisation_options"] = {
            "max_currents": 3.0e7,
            "gamma": 1e-8,
        }
    elif optimiser_name in ["CoilsetPositionCOP"]:
        options["optimisation_options"] = {
            "max_currents": 3.0e7,
            "gamma": 1e-8,
            "optimiser": Optimiser(
                algorithm_name="SBPLX",
                opt_conditions={
                    "stop_val": 2.5e-2,
                    "max_eval": 100,
                },
                opt_parameters={},
            ),
        }
    elif optimiser_name in ["NestedCoilsetPositionCOP"]:
        options["optimisation_options"] = {
            "optimiser": Optimiser(
                algorithm_name="SBPLX",
                opt_conditions={
                    "stop_val": 2.5e-2,
                    "max_eval": 100,
                },
                opt_parameters={},
            )
        }
        options["suboptimiser_name"] = "BoundedCurrentCOP"
        options["suboptimisation_options"] = {"max_currents": 3.0e7, "gamma": 1e-8}
    else:
        print("Coilset optimiser name not supported for this example")
    return options


# %%[markdown]

# ## Main program

# %%


def run(args):
    """
    Main program to solve the specified FBE problem.
    """
    optimiser_name = args.optimiser_name
    grid = init_grid()
    profile = init_profile()
    constraint_set = init_targets()[0]
    coilset = init_coilset()
    eq = init_equilibrium(grid, coilset, constraint_set)

    # Perform a fast initial unconstrained optimisation to create a
    # self consistent initial state
    if args.pre_optimise is True:
        pre_optimise(eq, profile, constraint_set)
    if optimiser_name is not None:
        options = default_optimiser_options(optimiser_name)
        optimiser = set_coilset_optimiser(eq.coilset, eq, constraint_set, **options)
        program = set_iterator(eq, profile, constraint_set, optimiser)
        optimise_fbe(program)
    eq.plot()
    plt.show()


# %%[markdown]

# ## Argument parsing

# Parse command line to control OptimisationProblem to use in example.

# %%

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optimiser_name",
        help="Name of Coilset OptimisationProblem to use",
        choices=[
            "Norm2Tikhonov",
            "UnconstrainedCurrentCOP",
            "BoundedCurrentCOP",
            "CoilsetPositionCOP",
            "NestedCoilsetPositionCOP",
        ],
        type=str,
        default="UnconstrainedCurrentCOP",
    )
    parser.add_argument(
        "--no-pre_optimise",
        help="Flag controlling if state should not pre optimised using unconstrained optimisation before passing to the constrained optimiser",
        dest="pre_optimise",
        action="store_false",
    )
    args = parser.parse_args()
    run(args)
