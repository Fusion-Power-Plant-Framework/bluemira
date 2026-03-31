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
Benchmark with FreeGNSKE on a MAST-U equilibrium.
"""

# %% [markdown]
#
# # FreeGSNKE MAST-U comparison

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.file import get_bluemira_path
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
from bluemira.equilibria.optimisation.problem import (
    UnconstrainedTikhonovCurrentGradientCOP,
)
from bluemira.equilibria.profiles import CustomProfile
from bluemira.equilibria.solve import (
    DudsonConvergence,
    PicardIterator,
)

plot_defaults()

# %% [markdown]
# Get MAST-U coils used by FreeGSNKE and convert to
# BM coilset of circuits.
# NOTE: Some coils appear to overlap in the FreeGSNKE example.
# We "fix" the solenoid here, but other overlaps are not addressed.

# %%
path = get_bluemira_path("equilibria", subfolder="examples")
name = "MAST-U_like_active_coils.json"
with open(Path(path, name)) as file:
    coil_dict = json.load(file)

circuits = []
count = []
for n, d in coil_dict.items():
    coils = []
    if n == "Solenoid":
        count.append(len(d["R"]))
        for i, (r, z) in enumerate(zip(d["R"], d["Z"], strict=False)):
            coil = Coil(
                r,
                z,
                current=5000,
                dx=d["dR"] / 2,
                dz=0.00489474,  # NOTE: Not quite sure how FreeGSNKE handles dZ
                # appears to create overlaps
                ctype="CS",
                name=n + "_" + str(i),
            )
            coils.append(coil)

    else:
        count.append(len(d["1"]["R"]))
        for i, (r1, z1, r2, z2) in enumerate(
            zip(
                d["1"]["R"],
                d["1"]["Z"],
                d["2"]["R"],
                d["2"]["Z"],
                strict=False,
            )
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
            coils.extend((coil_up, coil_low))

    circuit = Circuit(*coils)
    circuits.append(circuit)
full_coilset = CoilSet(*circuits)

full_coilset.control = [n for n in full_coilset.name if "Solenoid" not in n]

# %% [markdown]
# Load up the FreeGSNKE equilibrium for comparison purposes
# %%
path = get_bluemira_path("equilibria", subfolder="examples")
name = "MASTU-FREEGSNKE.eqdsk"

freegsnke_eq = Equilibrium.from_eqdsk(Path(path, name), from_cocos=7)


# %% [markdown]
# Match the profiles from the FreeGSNKE equilibrium with a CustomProfile

# %%
I_p = 6e5  # A
R_0 = 0.85  # m
B_0 = 0.588  # T, in FreeGSNKE fvac = R * B_0 = 0.5

pn = np.linspace(0, 1, 50)
profiles = CustomProfile(
    freegsnke_eq.profiles.pprime(pn), freegsnke_eq.ffprime(pn), R_0, B_0, I_p=I_p
)

# %% [markdown]
# Instantiate a new equilibrium

# %%
grid = Grid(0.1, 2.0, -2.2, 2.2, 65, 129)
eq = Equilibrium(full_coilset, grid, profiles, force_symmetry=True)

# %% [markdown]
# Match constraints (straight from FreeGSNKE example)

# %%
Rx = 0.6  # X-point radius
Zx = 1.1  # X-point height
Ra = 0.85
Rout = 1.4  # outboard midplane radius
Rin = 0.34  # inboard midplane radius
x_point_u = FieldNullConstraint(Rx, Zx, tolerance=1e-6)  # FreeGSNKE target tolerence
x_point_l = FieldNullConstraint(Rx, -Zx, tolerance=1e-6)
isoflux = IsofluxConstraint(
    [Rx, Rx, Rin, Rout, 1.3, 1.3, 0.8, 0.8],
    [Zx, -Zx, 0.0, 0.0, 2.1, -2.1, 1.62, -1.62],
    Rx,
    Zx,
    tolerance=1e-6,
)

constraints_list = [isoflux, x_point_u, x_point_l]
constraints_set = MagneticConstraintSet(constraints_list)

# %% [markdown]
# Set up an optimisation problem and converge an equilibrium.
# FreeGSNKE uses least squares problem with Tikhonov regularisation term
# (weights vary across coils - here we just use a floating point term)

# %%
current_opt_problem = UnconstrainedTikhonovCurrentGradientCOP(
    eq,
    targets=constraints_set,
    gamma=1e-8,
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

# %% [markdown]
# Compare the equilibria and profiles

# %%
f, ax = plt.subplots(1, 3)
freegsnke_eq.plot(ax=ax[0])
ax[0].set_title("FreeGSNKE")
eq.plot(ax=ax[1])
ax[1].set_title("BLUEMIRA")

sep = eq.get_separatrix()
fsep = freegsnke_eq.get_separatrix()

eq.coilset.plot(ax=ax[2])
ax[2].contour(eq.x, eq.z, eq.psi(), levels=20, cmap="viridis")
constraints_set.plot(ax=ax[2])
for fs in fsep:
    ax[2].plot(fs.x, fs.z, color="r")

for fs in sep:
    ax[2].plot(fs.x, fs.z, color="b", linestyle="--")

ax[2].set_title("Comparison")
ax[2].set_aspect("equal")
plt.show()


pn = np.linspace(0, 1, 50)
f, ax = plt.subplots(1, 2)
ax[0].plot(pn, eq.ffprime(pn), color="b", ls="--", label="BLUEMIRA")
ax[0].plot(pn, freegsnke_eq.ffprime(pn), color="r", label="FreeGSNKE")
ax[0].set_title("p'")
ax[0].set_xlabel(r"$\psi_n$")
ax[1].plot(pn, eq.pprime(pn), color="b", ls="--", label="BLUEMIRA")
ax[1].set_title("FF'")
ax[1].set_xlabel(r"$\psi_n$")
ax[1].plot(pn, freegsnke_eq.pprime(pn), color="r", label="FreeGSNKE")
ax[0].legend()
ax[1].legend()
plt.show()
