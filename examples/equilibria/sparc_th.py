# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Test MAST-U.
"""

# %%
from copy import deepcopy
import pickle

import matplotlib.pyplot as plt
import numpy as np

from bluemira.display.auto_config import plot_defaults
from bluemira.equilibria.coils import (
    Circuit,
    Coil,
    CoilSet,
    SymmetricCircuit,
)
from bluemira.equilibria.coils._grouping import CoilGroup
from bluemira.equilibria.diagnostics import PicardDiagnostic, PicardDiagnosticOptions
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.limiter import Limiter
from bluemira.equilibria.optimisation.constraints import (
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
)
from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import TauLimit, toroidal_harmonic_grid_and_coil_setup, toroidal_harmonic_approximation, plot_toroidal_harmonic_approximation
from bluemira.equilibria.optimisation.harmonics.harmonics_constraints import ToroidalHarmonicConstraint, ToroidalHarmonicConstraintFunction
from bluemira.equilibria.optimisation.problem._minimal_current import MinimalCurrentCOP
from bluemira.equilibria.optimisation.problem._tikhonov import TikhonovCurrentCOP, UnconstrainedTikhonovCurrentGradientCOP
from bluemira.equilibria.profiles import BetaIpProfile, DoublePowerFunc, CustomProfile
from bluemira.equilibria.solve import (
    DudsonConvergence,
    PicardIterator,
)
from bluemira.geometry.coordinates import Coordinates

plot_defaults()

# %%
# Get MAST-U coils used by FreeGSNKE and convert to
# BM coilset of circuits. 
# NOTE: Some coils appear to overlap in the FreeGSNKE example.
# We "fix" the solenoid here, but other overlaps are not addressed.
with open("SPARC_active_coils.pickle", "rb") as f:
    coil_dict = pickle.load(f)

with open("SPARC_limiter.pickle", "rb") as f:
    limiter_dict = pickle.load(f)

coordinates = Coordinates({"x": [d["R"] for d in limiter_dict], "y": 0, "z":[d["Z"] for d in limiter_dict]})
coordinates = coordinates.interpolate(dl=0.1)
limiter = Limiter(coordinates.x, coordinates.z)

circuits = []
currents = [-6.77520472e+04, -6.15831158e+04, -1.62891411e+05,  5.79804876e+04,
        1.00463966e+05,  8.82774883e+04, -9.75503810e+04, -1.40160914e+05,
        1.99916777e+04, -8.23151242e+03,  3.39000655e-03]
for j, (n, d) in enumerate(coil_dict.items()):
    upper = d["U"]
    lower = d["L"]
    ctype = "CS" if "CS" in n else "PF"

    coils = []
    for i in range(len(upper["R"])):
        coil = Coil(
            upper["R"][i],
            upper["Z"][i],
            current=0,
            dx=upper["dR"] / 2,
            dz=upper["dZ"] / 2,
            ctype=ctype,
            name=n + "_U_" + str(i),
        )
        coils.append(coil)
    for i in range(len(lower["R"])):
        coil = Coil(
            lower["R"][i],
            lower["Z"][i],
            current=0,
            dx=lower["dR"] / 2,
            dz=lower["dZ"] / 2,
            ctype=ctype,
            name=n + "_L_" + str(i),
        )
        coils.append(coil)
    circuit = Circuit(*coils, current=currents[j])
    circuits.append(circuit)

simple_coils = []
for circuit in circuits:
    x = np.average(circuit.x)
    dx = 0.5 * (np.max(circuit.x + circuit.dx) - np.min(circuit.x - circuit.dx) )
    dz = 0.5 * (np.max(np.abs(circuit.z) + circuit.dz) - np.min(np.abs(circuit.z) - circuit.dz) )
    
    z = np.average(np.abs((circuit.z)))
    current = np.sum(circuit.current)
    upper = Coil(x, z, current=current, dx=dx, dz=dz, name=circuit.name[0][:4], ctype=circuit.ctype[0])
   
    circuit = SymmetricCircuit(upper)
 
    circuit.discretisation = 0.1
    simple_coils.append(circuit)

simple_coilset = CoilSet(*simple_coils)
full_coilset = CoilSet(*circuits)
full_coilset.control = [n for n in full_coilset.name]
simple_coilset.control = [n for n in simple_coilset.name]

eq = Equilibrium.from_eqdsk("SPARC_equilibrium.eqdsk", from_cocos=7, force_symmetry=True)
eq.coilset = full_coilset
eq.limiter = None#limiter
eq.solve()
eq.solve()
eq.solve()
eq.solve()
eq.solve()
eq.solve()


orig_eq = deepcopy(eq)


setu = toroidal_harmonic_grid_and_coil_setup(eq, *eq.effective_centre(), TauLimit.MANUAL, min_tau_value=1.3)

eq.coilset.control = setu.th_coil_names
orig_eq.coilset.control = setu.th_coil_names

th_approx = toroidal_harmonic_approximation(eq, setu, n_degrees_of_freedom=5, plasma_mask=False)
_, x_points = eq.get_OX_points()

th_constraint = ToroidalHarmonicConstraint(th_approx, relative_tolerance_cos=1e-2, relative_tolerance_sin=1e-2)

rx_down = x_points[0].x
zx_down = x_points[0].z
rx_up = x_points[1].x
zx_up = x_points[1].z

nulls = [
    FieldNullConstraint(rx_down, zx_down, tolerance=1e-6),
    FieldNullConstraint(rx_down, zx_up, tolerance=1e-6),
]
th_eq = deepcopy(eq)

current_opt_problem = TikhonovCurrentCOP(
    th_eq,
    targets=MagneticConstraintSet(nulls),
    gamma=1e-8,
    opt_algorithm="COBYLA",
    opt_conditions={"max_eval": 1000, "ftol_rel": 1e-4},
    constraints=[th_constraint]+nulls,
    # opt_parameters={"initial_step": 0.1},
    # max_currents=3e10,
)
# start_currents = np.array(currents)
# result = current_opt_problem.optimise(x0=start_currents)

program = PicardIterator(
    th_eq,
    current_opt_problem,
    fixed_coils=True,
    diagnostic_plotting=PicardDiagnosticOptions(PicardDiagnostic.EQ),
    convergence=DudsonConvergence(3e-3),
    relaxation=0.0,
    maxiter=50,
)
program()


x_leg_in = np.array([1.3, rx_down, 1.53656704, 1.48805959, 1.43955214, 1.39104469, 1.34253724,
       1.29402979, 1.24552235, 1.1970149 , 1.14850745, 1.1       ])

z_leg_in = np.array([0.0, zx_down,-1.14243024, -1.19216021, -1.24189018, -1.29162016, -1.34135013,
       -1.3910801 , -1.44081008, -1.49054005, -1.54027003, -1.59      ])
upper_leg = IsofluxConstraint(x=x_leg_in, z=-z_leg_in, ref_x=x_leg_in[0], ref_z=-z_leg_in[0], tolerance=1e-6, weights=100)
lower_leg = IsofluxConstraint(x=x_leg_in, z=z_leg_in, ref_x=x_leg_in[0], ref_z=z_leg_in[0], tolerance=1e-6, weights=100)
th_eq_leg = deepcopy(th_eq)

constraint_set = MagneticConstraintSet([upper_leg, lower_leg])

current_opt_problem = TikhonovCurrentCOP(
    th_eq_leg,
    targets=constraint_set,
    gamma=1e-8,
    opt_algorithm="COBYLA",
    opt_conditions={"max_eval": 1000, "ftol_rel": 1e-4},
    constraints=nulls+[th_constraint],
    # opt_parameters={"initial_step": 0.1},
    # max_currents=3e10,
)

program = PicardIterator(
    th_eq_leg,
    current_opt_problem,
    fixed_coils=True,
    diagnostic_plotting=PicardDiagnosticOptions(PicardDiagnostic.EQ),
    convergence=DudsonConvergence(1e-3),
    relaxation=0.3,
    maxiter=20,
)
program()

ref_sep = eq.get_separatrix()

f, ax = plt.subplots(1, 3)
eq.plot(ax[0])
eq.coilset.plot(ax[0])
th_eq.plot(ax[1])
th_eq.coilset.plot(ax[1])

th_eq_leg.plot(ax[2])

th_eq_leg.coilset.plot(ax[2])


ax[0].contour(
    setu.R,
    setu.Z,
    th_approx.coilset_psi,
    levels=20,
    colors="red",
    linewidths=1,
)
constraint_set.plot(ax[2])
for s in ref_sep:
    ax[2].plot(s.x, s.z, color="b")
    
plt.show()
