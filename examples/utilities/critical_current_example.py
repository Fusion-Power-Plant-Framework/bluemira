import matplotlib.pyplot as plt
import numpy as np
from APECSfiles import Settings

import bluemira.equilibria.opt_constraints as opt_constraints
import examples.equilibria.double_null_ST as double_null_ST
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.opt_problems import BoundedCurrentCOP, UnconstrainedCurrentCOP
from bluemira.equilibria.solve import DudsonConvergence, PicardCoilsetIterator
from bluemira.utilities.opt_problems import OptimisationConstraint
from bluemira.utilities.optimiser import Optimiser

# Initiliase objects

grid = double_null_ST.init_grid()
profile = double_null_ST.init_profile()
coilset = double_null_ST.init_coilset()


eq = Equilibrium(
    coilset,
    grid,
    force_symmetry=True,
    vcontrol=None,
    psi=None,
    profiles=profile,
    Ip=16e6,
    li=None,
)


temperature_id = {
    "T_lts": 8.25,  # converged with 20 iterations at 7.75 ; 40 @ 8 ; 16 @ 8.25 ;
    "T_hts": 20,
}

conductor_id = (
    "ITER PF1,6 NbTi",
    "ITER PF2-4 NbTi",
    "ITER PF5 NbTi",
    "ACT CORC-CICC REBCO",
)


conductors = Settings.getDefaultConductors()
# Optimiser

optimiser = Optimiser(
    algorithm_name="COBYLA",
    opt_conditions={"max_eval": 300},  # default 200 increase to up stability
    opt_parameters={"initial_step": 0.01},  # default 0.03 reduce to increase stability
)

magnetic_targets, magnetic_core_targets = double_null_ST.init_targets()

opt_constraints = [
    OptimisationConstraint(
        f_constraint=opt_constraints.critical_current_constraint,
        # f_constraint_args={"eq": eq, "radius": 1.0},
        f_constraint_args={
            "eq": eq,
            "conductor_id": conductor_id,
            "temperature_id": temperature_id,
            "conductors": conductors,
            "scale": 1e6,
        },
        tolerance=np.array(
            [1e-4] * 7
        ),  # perhaps there is a better way to specify length of entries here
        constraint_type="inequality",
    )
]

opt_problem = BoundedCurrentCOP(
    coilset,
    eq,
    magnetic_targets,
    gamma=1e-8,
    max_currents=3.0e8,
    optimiser=optimiser,
    opt_constraints=opt_constraints,
)

constrained_iterator = PicardCoilsetIterator(
    eq,
    profile,
    magnetic_targets,  # magnetic_targets
    opt_problem,
    plot=True,
    relaxation=0.3,
    maxiter=400,
    convergence=DudsonConvergence(1e-4),
)

unconstrained_cop = UnconstrainedCurrentCOP(eq.coilset, eq, magnetic_targets, gamma=1e-8)

unconstrained_iterator = PicardCoilsetIterator(
    eq,
    profile,  # jetto
    magnetic_targets,
    unconstrained_cop,
    plot=False,
    relaxation=0.3,
    convergence=DudsonConvergence(1e-2),  # could make the same criterion f.e. 1e-3
    maxiter=400,
)

# f, ax = plt.subplots()
# eq.plot(ax=ax)
# unconstrained_iterator.constraints.plot(ax=ax)

unconstrained_iterator()
print(eq.coilset)
# f, ax = plt.subplots()
# eq.plot(ax=ax)
# unconstrained_iterator.constraints.plot(ax=ax)
# plt.show()

constrained_iterator()
print(eq.coilset)
f, ax = plt.subplots()
eq.plot(ax=ax)
constrained_iterator.constraints.plot(ax=ax)
plt.show()

t = 0
