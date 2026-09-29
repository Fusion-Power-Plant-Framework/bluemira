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
# SPDX-FileCopyrightText: 2024-present Bluemira contributors
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Forward Grad-Shafranov equilibrium solve delegating to FreeGSNKE.
"""

# %% [markdown]
# # Forward Grad-Shafranov Solve with FreeGSNKE
#
# This tutorial demonstrates how to perform **static forward Grad-Shafranov (GS) equilibrium solves**
# in Bluemira by delegating to FreeGSNKE's high-performance Newton-Krylov solver (`NKGSsolver`).
#
# ## Forward vs. Inverse Equilibrium Solves
# - **Inverse (Backward) Solve**: Given a desired plasma shape, magnetic nulls (X-points), and isoflux
#   targets, Bluemira optimizes the currents in the poloidal field (PF) coils.
# - **Forward Solve**: Given fixed currents in active/passive coils and plasma source profiles
#   ($p'(\psi_n)$ and $FF'(\psi_n)$), solve the nonlinear 2D elliptic Grad-Shafranov equation:
#   $$\Delta^* \psi = -\mu_0 R^2 p'(\psi) - F F'(\psi)$$
#   to determine the self-consistent poloidal magnetic flux distribution $\psi(R, Z)$, the plasma current
#   density $j_{\text{tor}}(R, Z)$, and the resulting magnetic topology (magnetic axis, X-points, and separatrix).
#
# Bluemira provides two ergonomic interfaces to run forward solves:
# 1. Direct in-place method: `Equilibrium.forward_solve()`
# 2. Configurable solver runner: `ForwardGSSolver` (supporting 2nd- and 4th-order spatial operators)

# %%
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.file import get_bluemira_path
from bluemira.display.auto_config import plot_defaults
from bluemira.equilibria.coils import Circuit, Coil, CoilSet
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.freegsnke_bridge import ForwardGSSolver, ForwardSolveResult
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.limiter import Limiter
from bluemira.equilibria.profiles import CustomProfile

plot_defaults()

# %% [markdown]
# ## 1. Build the Tokamak Coilset & Limiter
#
# We model a realistic MAST-U spherical tokamak configuration using FreeGSNKE's machine definition.
# Active coils are loaded from `MAST-U_like_active_coils.json` and grouped into 12 circuits:
# - Central Solenoid (CS)
# - Poloidal field and divertor shaping coil pairs (PX, D1-D7, P4-P6)
#
# Calibrated coil currents from FreeGSNKE produce a realistic double-null diverted equilibrium.
# The 47-point vacuum vessel first wall limiter is loaded from `MAST-U_like_limiter.json`.

# %%
path = get_bluemira_path("equilibria", subfolder="examples")
coils_file = Path(path, "MAST-U_like_active_coils.json")
limiter_file = Path(path, "MAST-U_like_limiter.json")

with open(coils_file) as f:
    coil_dict = json.load(f)

with open(limiter_file) as f:
    limiter_data = json.load(f)

limiter = Limiter(
    [pt["R"] for pt in limiter_data],
    [pt["Z"] for pt in limiter_data],
)

# Calibrated active coil currents for MAST-U double-null diverted equilibrium
circuit_currents = {
    "Solenoid": 5000.0,
    "PX": 3696.81,
    "D1": 6941.24,
    "D2": 4390.86,
    "D3": 2924.36,
    "Dp": -2488.20,
    "D5": 249.21,
    "D6": -315.00,
    "D7": 511.08,
    "P4": -3564.38,
    "P5": -3954.24,
    "P6": 0.0,
}

circuits = []
for name, data in coil_dict.items():
    coils = []
    if name == "Solenoid":
        for i, (r, z) in enumerate(zip(data["R"], data["Z"], strict=False)):
            coil = Coil(
                r,
                z,
                current=0.0,
                dx=data["dR"] / 2.0,
                dz=0.00489474,
                ctype="CS",
                name=f"{name}_{i}",
            )
            coils.append(coil)
    else:
        for i, (r1, z1, r2, z2) in enumerate(
            zip(
                data["1"]["R"],
                data["1"]["Z"],
                data["2"]["R"],
                data["2"]["Z"],
                strict=False,
            )
        ):
            coils.append(
                Coil(
                    r1,
                    z1,
                    current=0.0,
                    dx=data["1"]["dR"] / 2.0,
                    dz=data["1"]["dZ"] / 2.0,
                    ctype="PF",
                    name=f"{name}U_{i}",
                )
            )
            coils.append(
                Coil(
                    r2,
                    z2,
                    current=0.0,
                    dx=data["2"]["dR"] / 2.0,
                    dz=data["2"]["dZ"] / 2.0,
                    ctype="PF",
                    name=f"{name}L_{i}",
                )
            )
    circ = Circuit(*coils)
    circ.current = circuit_currents.get(name, 0.0)
    circuits.append(circ)

coilset = CoilSet(*circuits)
print(f"Constructed CoilSet with {len(circuits)} circuit elements and attached MAST-U limiter.")

# %% [markdown]
# ## 2. Define Plasma Source Profiles & Computational Grid
#
# We load normalized reference profiles from an EQDSK file using Bluemira's strict COCOS interface
# (`from_cocos=7` for FreeGSNKE Wb/rad convention) and instantiate a `CustomProfile`:
# - Total plasma current: $I_p = 600\,	ext{kA}$
# - Major radius: $R_0 = 0.85\,	ext{m}$
# - Vacuum toroidal field: $B_0 = 0.588\,	ext{T}$ ($F_{	ext{vac}} = R_0 B_0 = 0.5\,	ext{T}\cdot	ext{m}$)

# %%
eqdsk_file = Path(path, "MASTU-FREEGSNKE.eqdsk")
ref_eq = Equilibrium.from_eqdsk(eqdsk_file, from_cocos=7)

psi_n = np.linspace(0.0, 1.0, 50)
profiles = CustomProfile(
    ref_eq.profiles.pprime(psi_n),
    ref_eq.ffprime(psi_n),
    R_0=0.85,
    B_0=0.588,
    I_p=6.0e5,
)

# Define 2D computational domain: R in [0.1, 2.0] m, Z in [-2.2, 2.2] m
grid = Grid(0.1, 2.0, -2.2, 2.2, nx=65, nz=129)

# Instantiate the Bluemira Equilibrium with coilset, limiter, and profiles
eq = Equilibrium(coilset, grid, profiles, limiter=limiter, force_symmetry=True)
print(f"Equilibrium initialized on grid ({grid.nx} x {grid.nz}).")

# %% [markdown]
# ## 3. Execute Forward Solve (In-Place Method)
#
# We now solve the forward problem by calling `eq.forward_solve()`.
# Behind the scenes:
# 1. Bluemira translates coils and circuits into a FreeGSNKE `Machine`.
# 2. Profiles are mapped into FreeGSNKE's `GeneralPprimeFFprime`.
# 3. FreeGSNKE's Newton-Krylov solver executes iterations to convergence.
# 4. Total poloidal flux, current density, plasma current, and magnetic topology are updated in-place on `eq`.

# %%
result: ForwardSolveResult = eq.forward_solve(
    target_relative_tolerance=1e-5,
    max_iterations=40,
    order=2,
    verbose=False,
    suppress=True,
)

print("=" * 50)
print("Forward Solve Result:")
print(f"  Converged:       {result.converged}")
print(f"  Iterations:      {result.iterations}")
print(f"  Relative Error:  {result.relative_error:.2e}")
print(f"  Elapsed Time:    {result.time_taken:.3f} s")
print(f"  Psi Axis:        {result.psi_axis:.4f} Wb/rad")
print(f"  Psi Boundary:    {result.psi_boundary:.4f} Wb/rad")
print(f"  Plasma Current:  {result.plasma_current / 1e3:.1f} kA")
print("=" * 50)

# %% [markdown]
# ## 4. Alternative: Object-Oriented `ForwardGSSolver` with Higher-Order Operator
#
# For advanced workflows, `ForwardGSSolver` allows fine-grained control over solver options,
# such as using 4th-order spatial finite-difference operators (`order=4`) for enhanced numerical precision:

# %%
solver_order4 = ForwardGSSolver(
    eq,
    target_relative_tolerance=1e-5,
    max_iterations=40,
    order=4,
)
result_order4 = solver_order4.solve(suppress=True)

print(
    f"4th-order operator solved in {result_order4.iterations} iterations "
    f"(rel error: {result_order4.relative_error:.2e})."
)

# %% [markdown]
# ## 5. Visualizing the Converged Equilibrium
#
# ### Plot 1: 2D Magnetic Flux Topology, Limiter, & Separatrix
# We visualize total poloidal flux contours $\psi(R, Z)$, coil blocks, magnetic axis (O-point),
# divertor nulls (X-points), the Last Closed Flux Surface (separatrix), and the vacuum vessel limiter:

# %%
fig, ax = plt.subplots(figsize=(7, 9))
eq.plot(ax=ax)
limiter.plot(ax=ax)
ax.set_title("MAST-U Forward Solve: Total Poloidal Flux & Topology")
ax.set_xlabel("R [m]")
ax.set_ylabel("Z [m]")
ax.set_aspect("equal")
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Plot 2: Toroidal Current Density, Source Profiles, and Safety Factor
# Here we examine internal plasma physics quantities:
# 1. 2D distribution of toroidal current density $j_{\text{tor}}(R, Z)$.
# 2. Pressure gradient $p'(\psi_n)$ and poloidal current function gradient $FF'(\psi_n)$.
# 3. Radial safety factor profile $q(\psi_n)$ across normalized flux surfaces.

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. 2D Toroidal Current Density
jtor_ma = eq._jtor * 1e-6
cf = axes[0].contourf(eq.x, eq.z, jtor_ma, levels=25, cmap="plasma")
fig.colorbar(cf, ax=axes[0], label=r"$j_{\mathrm{tor}}$ [$\mathrm{MA/m^2}$]")
axes[0].set_title(r"Toroidal Current Density $j_{\mathrm{tor}}(R, Z)$")
axes[0].set_xlabel("R [m]")
axes[0].set_ylabel("Z [m]")
axes[0].set_aspect("equal")

# 2. Plasma Source Profiles
axes[1].plot(psi_n, eq.profiles.pprime(psi_n) * 1e-3, "b-", lw=2, label=r"$p'(\psi_n)$ [$\mathrm{kPa/Wb}$]")
axes[1].set_xlabel(r"Normalized Flux $\psi_n$")
axes[1].set_ylabel(r"$p'(\psi_n)$ [$\mathrm{kPa/Wb}$]", color="b")
axes[1].tick_params(axis="y", labelcolor="b")

ax1_twin = axes[1].twinx()
ax1_twin.plot(psi_n, eq.profiles.ffprime(psi_n), "r--", lw=2, label=r"$FF'(\psi_n)$ [$(\mathrm{T\,m})^2/\mathrm{Wb}$]")
ax1_twin.set_ylabel(r"$FF'(\psi_n)$ [$(\mathrm{T\,m})^2/\mathrm{Wb}$]", color="r")
ax1_twin.tick_params(axis="y", labelcolor="r")
axes[1].set_title("Plasma Source Profiles")
axes[1].grid(True, alpha=0.3)

# 3. Safety Factor Profile q(psi_n)
psi_norm_q = np.linspace(0.05, 0.95, 15)
q_values = eq.q(psi_norm_q)
axes[2].plot(psi_norm_q, q_values, "k-o", markersize=4, lw=1.5)
axes[2].set_xlabel(r"Normalized Flux $\psi_n$")
axes[2].set_ylabel(r"Safety Factor $q$")
axes[2].set_title(r"Safety Factor Profile $q(\psi_n)$")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# %% [markdown]
# ### Plot 3: Midplane Poloidal Flux and Magnetic Field Components
# We analyze midplane ($Z = 0$) profiles to observe how plasma diamagnetism and coil currents
# shape the magnetic well:
# 1. Total poloidal flux decomposed into coil contribution $\psi_{\text{coil}}$ and plasma contribution $\psi_{\text{plasma}}$.
# 2. Vertical magnetic field $B_z(R, 0)$ decomposed into external coil field and plasma self-field.

# %%
r_mid = np.linspace(eq.grid.x_min, eq.grid.x_max, 100)
z_mid = np.zeros_like(r_mid)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Midplane Flux Decomposition
axes[0].plot(r_mid, eq.psi(r_mid, z_mid), "b-", lw=2, label=r"Total $\psi(R, 0)$")
axes[0].plot(r_mid, eq.coilset.psi(r_mid, z_mid), "g--", lw=1.5, label=r"Coil $\psi_{\mathrm{coil}}$")
axes[0].plot(r_mid, eq.plasma.psi(r_mid, z_mid), "m:", lw=1.5, label=r"Plasma $\psi_{\mathrm{plasma}}$")
axes[0].set_xlabel("R [m]")
axes[0].set_ylabel(r"Poloidal Flux $\psi$ [$\mathrm{Wb}$]")
axes[0].set_title("Midplane Poloidal Flux Decomposition")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Midplane Vertical Field Decomposition
axes[1].plot(r_mid, eq.Bz(r_mid, z_mid), "b-", lw=2, label=r"Total $B_z(R, 0)$")
axes[1].plot(r_mid, eq.coilset.Bz(r_mid, z_mid), "g--", lw=1.5, label=r"Coil $B_{z,\mathrm{coil}}$")
axes[1].plot(r_mid, eq.plasma.Bz(r_mid, z_mid), "m:", lw=1.5, label=r"Plasma $B_{z,\mathrm{plasma}}$")
axes[1].set_xlabel("R [m]")
axes[1].set_ylabel(r"Vertical Magnetic Field $B_z$ [$\mathrm{T}$]")
axes[1].set_title(r"Midplane Vertical Magnetic Field $B_z(R, 0)$")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
