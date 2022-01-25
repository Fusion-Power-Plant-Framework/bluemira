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
Finds a ST equilibrium in a double null configuration, using a constrained
optimisation method with bound constraints on the maximum coil currents
and on the position of the inboard midplane.
"""

import copy

import matplotlib.pyplot as plt
import numpy as np

import bluemira.equilibria.constraint_library as constraint_library
import examples.equilibria.double_null_ST as double_null_ST
from bluemira.display.auto_config import plot_defaults
from bluemira.equilibria.coils import Coil
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.opt_problems import BoundedCurrentCOP, UnconstrainedCurrentCOP
from bluemira.equilibria.solve import DudsonConvergence, PicardCoilsetIterator
from bluemira.utilities.opt_problems import OptimisationConstraint
from bluemira.utilities.optimiser import Optimiser

# %%[markdown]

# # Script to demonstrate constrained coilset optimisation of a double null equilibrium

# %%

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


def init_equilibrium(grid, coilset, targets, profile):
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
        profiles=profile,
        Ip=0,
        li=None,
    )
    targets(eq)
    optimiser = UnconstrainedCurrentCOP(coilset_temp, eq, targets, gamma=1e-7)
    coilset_temp = optimiser(eq, targets)

    coilset.set_control_currents(coilset_temp.get_control_currents())

    psi = coilset_temp.psi(grid.x, grid.z).copy()

    # Set up an equilibrium problem and solve it
    eq = Equilibrium(
        coilset,
        grid,
        force_symmetry=True,
        vcontrol=None,
        psi=psi,
        profiles=profile,
        Ip=Ip,
        li=None,
    )
    return eq


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


def pre_optimise(eq, profile, targets):
    """
    Run a simple unconstrained optimisation to improve the
    initial equilibrium for the main optimiser.
    """
    optimiser = UnconstrainedCurrentCOP(eq.coilset, eq, targets, gamma=1e-8)

    program = PicardCoilsetIterator(
        eq,
        profile,  # jetto
        targets,
        optimiser,
        plot=True,
        gif=False,
        relaxation=0.3,
        convergence=DudsonConvergence(1e-2),
        maxiter=400,
    )

    eq = optimise_fbe(program)
    return eq


def init_opt_constraints():
    """
    Create iterable of OptimisationConstraint objects to apply
    during the coilset optimisation.
    """
    opt_constraints = []
    constrain_core_isoflux_targets = OptimisationConstraint(
        constraint_library.current_midplane_constraint,
        {"radius": 1.0},
        np.array([1e-4]),
        "inequality",
    )
    opt_constraints.append(constrain_core_isoflux_targets)

    return []


def set_coilset_optimiser(coilset, eq, magnetic_targets):
    """
    Create the optimiser to be used to optimise the coilset.
    """
    opt_constraints = init_opt_constraints()
    optimisation_options = {
        "eq": eq,
        "targets": magnetic_targets,
        "gamma": 1e-8,
        "max_currents": 3.0e7,
        "optimiser": Optimiser(
            algorithm_name="SLSQP",
            opt_conditions={"max_eval": 200},
            opt_parameters={"initial_step": 0.03},
        ),
        "opt_constraints": opt_constraints,
    }
    optimiser = BoundedCurrentCOP(coilset, **optimisation_options)
    return optimiser


def set_iterator(eq, profile, targets, optimiser):
    """
    Create the iterator to be used to solve the FBE.
    """
    iterator_args = (eq, profile, targets, optimiser)
    iterator_kwargs = {
        "plot": False,
        "gif": False,
        "relaxation": 0.3,
        "maxiter": 400,
        "convergence": DudsonConvergence(1e-4),
    }

    program = PicardCoilsetIterator(*iterator_args, **iterator_kwargs)
    return program


def run():
    """
    Main program to solve the specified FBE problem.
    """
    grid = double_null_ST.init_grid()
    profile = double_null_ST.init_profile()
    targets, core_targets = double_null_ST.init_targets()
    coilset = double_null_ST.init_coilset()
    eq = init_equilibrium(grid, coilset, targets, profile)

    # Perform a fast initial unconstrained optimisation to create a
    # self consistent initial state
    pre_optimise(eq, profile, targets)

    optimiser = set_coilset_optimiser(eq.coilset, eq, targets)
    program = set_iterator(eq, profile, targets, optimiser)
    optimise_fbe(program)
    eq.plot()
    plt.show()


if __name__ == "__main__":
    run()
