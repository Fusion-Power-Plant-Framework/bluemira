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
import copy
from bluemira.display.auto_config import plot_defaults
from bluemira.equilibria.profiles import CustomProfile
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.constraints import (
    MagneticConstraintSet,
    IsofluxConstraint,
)
from bluemira.equilibria.coils import Coil, CoilSet, SymmetricCircuit
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimiser import (
    UnconstrainedCurrentOptimiser,
    BoundedCurrentOptimiser,
    ConnectionLengthOptimiser,
)
from bluemira.equilibria.solve import (
    PicardCoilsetIterator,
    DudsonConvergence,
)
from bluemira.utilities.optimiser import OptimiserConstraint
from bluemira.utilities.opt_tools import (
    ConstraintLibrary,
)
from bluemira.geometry._deprecated_loop import Loop

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


def init_grid():
    """
    Create the grid for the FBE solver.
    """
    r0, r1 = 0.2, 8
    z0, z1 = -8, 8
    nx, nz = 129, 257
    grid = Grid(r0, r1, z0, z1, nx, nz)
    return grid


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


def init_targets():
    """
    Create the set of isoflux targets for the FBE optimisation objective function.
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
    optimiser = UnconstrainedCurrentOptimiser(coilset_temp, gamma=1e-7)
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
    return


def pre_optimise(eq, profile, targets):
    """
    Run a simple unconstrained optimisation to improve the
    initial equilibrium for the main optimiser.
    """
    optimiser = UnconstrainedCurrentOptimiser(eq.coilset, gamma=1e-8)

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
    Create iterable of OptimiserConstraint objects to apply
    during the coilset optimisation.
    """
    constrain_core_isoflux_targets = OptimiserConstraint(
        ConstraintLibrary.objective_constraint,
        (BoundedCurrentOptimiser.f_min_objective, 0.2),
    )
    opt_constraints = [constrain_core_isoflux_targets]

    return opt_constraints


def set_coilset_optimiser(
    coilset,
):
    """
    Create the optimiser to be used to optimise the coilset.
    """
    opt_constraints = init_opt_constraints()
    optimisation_options = {
        "max_currents": 5.0e8,
        "gamma": 1e-8,
        "opt_args": {
            "algorithm_name": "COBYLA",
            "opt_conditions": {
                "stop_val": -10.0,
                "max_eval": 40,
            },
            "opt_parameters": {"initial_step": 0.01},
        },
        "opt_constraints": opt_constraints,
    }
    optimiser = ConnectionLengthOptimiser(coilset, **optimisation_options)
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
    grid = init_grid()
    profile = init_profile()
    targets, core_targets = init_targets()
    coilset = init_coilset()
    eq = init_equilibrium(grid, coilset, targets, profile)

    # Perform a fast initial unconstrained optimisation to create a
    # self consistent initial state
    pre_optimise(eq, profile, targets)

    optimiser = set_coilset_optimiser(eq.coilset)
    program = set_iterator(eq, profile, core_targets, optimiser)
    optimise_fbe(program)
    eq.plot()
    plt.show()


if __name__ == "__main__":
    run()
