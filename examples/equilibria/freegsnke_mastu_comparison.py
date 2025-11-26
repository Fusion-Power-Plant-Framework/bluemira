# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Test MAST-U.
"""

# %%
import pickle

import matplotlib.pyplot as plt
import numpy as np

from bluemira.display.auto_config import plot_defaults
from bluemira.equilibria.coils import (
    Circuit,
    Coil,
    CoilSet,
)
from bluemira.equilibria.diagnostics import PicardDiagnostic, PicardDiagnosticOptions
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.optimisation.constraints import (
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
)
from bluemira.equilibria.optimisation.problem._tikhonov import UnconstrainedTikhonovCurrentGradientCOP
from bluemira.equilibria.profiles import BetaIpProfile, DoublePowerFunc
from bluemira.equilibria.solve import (
    DudsonConvergence,
    PicardIterator,
)

plot_defaults()

# %%
# Get MAST-U coils used by FreeGSNKE and convert to
# BM coilset of circuits. 
# NOTE: Some coils appear to overlap in the FreeGSNKE example.
# We "fix" the solenoid here, but other overlaps are not addressed.
with open("MAST-U_like_active_coils.pickle", "rb") as f:
    coil_dict = pickle.load(f)

circuits = []
count = []
for n, d in coil_dict.items():
    coils = []
    if n == "Solenoid":
        i = 0
        count.append(len(d["R"]))
        for r, z in zip(d["R"], d["Z"], strict=False):
            coil = Coil(
                r,
                z,
                current=5000,
                dx=d["dR"] / 2,
                dz=0.00489474,  # NOTE: Not quite sure how FreeGSNKE handles dZ - appears to 
                # create overlaps
                ctype="CS",
                name=n + "_" + str(i),
            )
            coils.append(coil)
            i += 1
    else:
        i = 0
        count.append(len(d["1"]["R"]))
        for r1, z1, r2, z2 in zip(
            d["1"]["R"],
            d["1"]["Z"],
            d["2"]["R"],
            d["2"]["Z"],
            strict=False,
        ):
            coil_up = Coil(
                r1,
                z1,
                current=0,
                dx=d["1"]["dR"] / 2,
                dz=d["1"]["dZ"] / 2,
                ctype="PF",
                name=n + "U_" + str(i),
            )
            coil_low = Coil(
                r2,
                z2,
                current=0,
                dx=d["2"]["dR"] / 2,
                dz=d["2"]["dZ"] / 2,
                ctype="PF",
                name=n + "L_" + str(i),
            )
            coils.append(coil_up)
            coils.append(coil_low)
            i += 1
    circuit = Circuit(*coils)
    circuits.append(circuit)
full_coilset = CoilSet(*circuits)

full_coilset.control = [n for n in full_coilset.name if "Solenoid" not in n]

# %%
# Grid
grid = Grid(0.1, 2.0, -2.2, 2.2, 65, 129)

# Profile params
I_p = 6e5  # A
R_0 = 0.85  # m
A = 1.3
B_0 = 0.588  # T, in FreeGSNKE fvac = R * B_0 = 0.5
betap = 0.274207  # using Paxis profile in FreeGSNKE, we find the beta to be this

# Use DoublePowerFunc to match FreeGSNKE
profiles = BetaIpProfile(
    betap=betap,
    I_p=I_p,
    R_0=R_0,
    B_0=B_0,
    shape=DoublePowerFunc([1.8, 1.2]),  # FreeGSNKE: alpha_m=1.8, alpha_n=1.2
)


# Equilibrium
# Note sure which COCOS FreeGSNKE uses... probably 7 or 8
freegsnke_eq = Equilibrium.from_eqdsk("MASTU-FREEGSNKE.eqdsk", from_cocos=7)

# Forcing symmetry does not appear to be necessary here.
eq = Equilibrium(full_coilset, grid, profiles, force_symmetry=True)

# %%
# Constraints (straight from FreeGSNKE example)
Rx = 0.6      # X-point radius
Zx = 1.1      # X-point height
Ra = .85
Rout = 1.4    # outboard midplane radius
Rin = 0.34    # inboard midplane radius
x_point_u = FieldNullConstraint(Rx, Zx, tolerance=1e-6)  # FreeGSNKE target tolerence
x_point_l = FieldNullConstraint(Rx, -Zx, tolerance=1e-6)
isoflux = IsofluxConstraint([Rx, Rx, Rin, Rout, 1.3, 1.3, .8,.8], [Zx, -Zx, 0.,0., 2.1, -2.1,1.62,-1.62], Rx, Zx, tolerance=1e-6)

constraints_list = [isoflux, x_point_u, x_point_l]
constraints_set = MagneticConstraintSet(constraints_list)

# FreeGSNKE uses least squares problem with Tikhonov regularisation term
# I think that SLSQP is aapropriate algo, constrained least squares
current_opt_problem = UnconstrainedTikhonovCurrentGradientCOP(
    eq,
    targets=constraints_set,
    gamma=1e-8,
    opt_algorithm="SLSQP",
    opt_conditions={"max_eval": 2000, "ftol_rel": 1e-6},
    opt_parameters={},
)

program = PicardIterator(
    eq,
    current_opt_problem,
    fixed_coils=True,
    diagnostic_plotting=PicardDiagnosticOptions(PicardDiagnostic.EQ),
    convergence=DudsonConvergence(
        1.0e-3,
    ),
    relaxation=0.0,  # FreeGSNKE blend = 0.0
    maxiter=50,
)
_ = program()
f, ax = plt.subplots()
eq.plot(ax=ax)
constraints_set.plot(ax=ax)
eq.coilset.plot(ax=ax)
plt.show()


sep = eq.get_separatrix()
fsep = freegsnke_eq.get_separatrix()

f, ax = plt.subplots(figsize=[10, 9])
eq.coilset.plot(ax=ax)
ax.contour(eq.x, eq.z, eq.psi(), levels=20, cmap="viridis")
constraints_set.plot(ax=ax)
for fs in fsep:
    ax.plot(fs.x, fs.z, color="r")

for fs in sep:
    ax.plot(fs.x, fs.z, color="b", linestyle="--")

ax.set_aspect("equal")
f.savefig("mast_u_equilibrium.pdf", dpi=600, format="pdf", bbox_inches="tight")


pn = np.linspace(0, 1, 50)
f, ax = plt.subplots(1, 2)
ax[0].plot(pn, eq.ffprime(pn), color="b", ls="--")
ax[0].plot(pn, freegsnke_eq.ffprime(pn), color="r")
ax[1].plot(pn, eq.pprime(pn), color="b", ls="--")
ax[1].plot(pn, freegsnke_eq.pprime(pn), color="r")
plt.show()

