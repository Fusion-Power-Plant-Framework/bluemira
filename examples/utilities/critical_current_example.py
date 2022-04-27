# TF object imports
import os

import matplotlib.pyplot as plt
import numpy as np
from APECSfiles import Settings

import bluemira.equilibria.opt_constraints as opt_constraints
import examples.equilibria.double_null_ST as double_null_ST
from bluemira.base.parameter import ParameterFrame
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.opt_problems import BoundedCurrentCOP, UnconstrainedCurrentCOP
from bluemira.equilibria.shapes import flux_surface_manickam
from bluemira.equilibria.solve import DudsonConvergence, PicardCoilsetIterator

# Current souce creation
from bluemira.magnetostatics.biot_savart import BiotSavartFilament
from bluemira.magnetostatics.circuits import HelmholtzCage
from bluemira.utilities.opt_problems import OptimisationConstraint
from bluemira.utilities.optimiser import Optimiser

# from BLUEPRINT.cad.model import CADModel
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.systems.optimisation_callbacks import TF_optimiser
from BLUEPRINT.systems.tfcoils import ToroidalFieldCoils

# %%

# Initiliase EQ objects
grid = double_null_ST.init_grid()
profile = double_null_ST.init_profile()
coilset = double_null_ST.init_coilset()

# Equilibrium object
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
# TF object parameters
params = [
    ["R_0", "Major radius", 3.639, "m", None, "Input"],
    ["B_0", "Toroidal field at R_0", 2.0, "T", None, "Input"],
    ["n_TF", "Number of TF coils", 12, "dimensionless", None, "Input"],
    ["tk_tf_nose", "TF coil inboard nose thickness", 0.0377, "m", None, "Input"],
    [
        "tk_tf_side",
        "TF coil inboard case minimum side wall thickness",
        0.02,
        "m",
        None,
        "Input",
    ],
    ["tk_tf_wp", "TF coil winding pack thickness", 0.569, "m", None, "PROCESS"],
    [
        "tk_tf_front_ib",
        "TF coil inboard steel front plasma-facing",
        0.02,
        "m",
        None,
        "Input",
    ],
    ["tk_tf_ins", "TF coil ground insulation thickness", 0.008, "m", None, "Input"],
    [
        "tk_tf_insgap",
        "TF coil WP insertion gap",
        1.0e-7,
        "m",
        "Backfilled with epoxy resin (impregnation)",
        "Input",
    ],
    [
        "r_tf_in",
        "Inboard radius of the TF coil inboard leg",
        0.148,
        "m",
        None,
        "PROCESS",
    ],
    [
        "tf_wp_depth",
        "TF coil winding pack depth (in y)",
        0.3644,
        "m",
        "Including insulation",
        "PROCESS",
    ],
    ["ripple_limit", "Ripple limit constraint", 0.6, "%", None, "Input"],
    [
        "r_tf_outboard_corner",
        "Corner Radius of TF coil outboard legs",
        0.8,
        "m",
        None,
        "Input",
    ],
    [
        "r_tf_inboard_corner",
        "Corner Radius of TF coil inboard legs",
        0.0,
        "m",
        None,
        "Input",
    ],
    ["tk_tf_inboard", "TF coil inboard thickness", 0.6267, "m", None, "PROCESS"],
]

parameters = ParameterFrame(params)

# Read the parameters
read_path = ""
write_path = "./"

# Define last closed line flux surface
lcfs = flux_surface_manickam(3.639, 0, 2.183, 2.8, 0.543, n=40)
lcfs.close()

# Define Keep Out Zone (KOZ)
name = os.sep.join([read_path, "KOZ_PF_test1.json"])
ko_zone = Loop.from_file("KOZ_PF_test1.json")

# Initiliase TF parameters
to_tf = {
    "name": "Example_PolySpline_TF",
    "plasma": lcfs,
    "koz_loop": ko_zone,
    "shape_type": "P",  # This is the shape parameterisation to use
    "wp_shape": "W",  # This is the winding pack shape choice for the inboard leg
    "conductivity": "SC",  # Resistive (R) or Superconducting (SC)
    "npoints": 200,
    "obj": "L",  # This is the optimisation objective: minimise length
    "ny": 3,  # This is the number of current filaments to use in y
    "nr": 2,  # This is the number of current filaments to use in x
    "nrip": 4,  # This is the number of points on the separatrix to calculate ripple for
    "read_folder": read_path,  # This is the path that the shape will be read from
    "write_folder": write_path,  # This is the path that the shape will be written to
}
# Build the TF object
tf = ToroidalFieldCoils(parameters, to_tf)
tf.build(TF_optimiser)

import copy

tf_centerline = copy.deepcopy(tf.geom["Centreline"])
tf_centerline.x *= 0.72
tf_centerline.z *= 0.72

arrays = tf_centerline
radius = tf.params.tf_wp_width
current = tf.params.I_tf

# center column TF object
tf_source = BiotSavartFilament(arrays, radius, current)
# HelmholtzCage
hmc = HelmholtzCage(tf_source, tf.params.n_TF)

temperature_id = {
    "T_lts": 5,  # converged with 20 iterations at 7.75 ; 40 @ 8 ; 16 @ 8.25 ;
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
            "tf_source": tf_source,
            "tf": tf,
            "tf_centerline": tf_centerline,
            "hmc": hmc,
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


unconstrained_iterator()
print(eq.coilset)

constrained_iterator()
print(eq.coilset)
f, ax = plt.subplots()
eq.plot(ax=ax)
constrained_iterator.constraints.plot(ax=ax)
plt.show()
