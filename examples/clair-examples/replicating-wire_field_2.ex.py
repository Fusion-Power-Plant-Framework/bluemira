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
Replicating the wire_field_2.m file
"""

# %%
# copy coilset and grid from single_null.ex.py
# edit coilset so one of the PF coils has nonzero current
import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.constants import MU_0
from bluemira.equilibria.coils._coil import Coil
from bluemira.equilibria.coils._grouping import CoilSet
from bluemira.equilibria.grid import Grid
from bluemira.utilities.tools import cylindrical_to_toroidal, toroidal_to_cylindrical

# %%
# Creating legendre functions to test


# %%
# Creating single coil

x = [5.4, 14.0, 17.75, 17.75, 14.0, 7.0, 2.77, 2.77, 2.77, 2.77, 2.77]
z = [9.26, 7.9, 2.5, -2.5, -7.9, -10.5, 7.07, 4.08, -0.4, -4.88, -7.86]
dx = [0.6, 0.7, 0.5, 0.5, 0.7, 1.0, 0.4, 0.4, 0.4, 0.4, 0.4]
dz = [0.6, 0.7, 0.5, 0.5, 0.7, 1.0, 2.99 / 2, 2.99 / 2, 5.97 / 2, 2.99 / 2, 2.99 / 2]


# trial single coil in the middle:

coil1 = [Coil(10, 0, current=5, dx=dx[0], dz=dz[0], ctype="PF", name="PF_0")]
coilset1 = CoilSet(*coil1)
print(coilset1)
grid = Grid(3.0, 13.0, -10.0, 10.0, 65, 65)
f, ax = plt.subplots()
coilset1.plot(ax=ax)

psi = coilset1.psi(grid.x, grid.z, sum_coils=False)
print(np.shape(psi))
print(np.shape(grid.x))
print(np.shape(grid.z))
ax.contour(grid.x, grid.z, psi)


# %%
# Set one of the coils to have nonzero current
# TODO check what a realistic/appropriate value is
# coils = []
# j = 1
# for i, (xi, zi, dxi, dzi) in enumerate(zip(x, z, dx, dz, strict=False)):
#     if j > 6:
#         j = 1
#     ctype = "PF" if i < 6 else "CS"
#     current_val = 10 if i == 1 else 0
#     coil = Coil(
#         xi,
#         zi,
#         current=current_val,
#         dx=dxi,
#         dz=dzi,
#         ctype=ctype,
#         name=f"{ctype}_{j}",
#     )
#     coils.append(coil)
#     j += 1

# coilset = CoilSet(*coils)

# print(coilset)
# grid = Grid(3.0, 13.0, -10.0, 10.0, 65, 65)

# f, ax = plt.subplots()
# coilset.plot(ax=ax)


# psi = coilset.psi(grid.x, grid.z)
# print(psi)
# print(np.shape(grid.x))
# print(np.shape(grid.z))
# ax.contour(grid.x, grid.z, psi)

# %%

# wire location
R_c = 5.0
Z_c = 1.0
I_c = 5e6

# focus of toroidal coords
R_0 = 3.0
Z_0 = 0.2

# location of wire in toroidal coords - use my toroidal coord transform fns
tau_c, sigma_c = cylindrical_to_toroidal(R_0=R_0, z_0=Z_0, R=R_c, Z=Z_c)
print(tau_c)
print(sigma_c)

# range of tau values (copied from matlab - TODO find out why)
# using approximate value for d2_min to avoid infinities
# approximating the tau at the focus instead of using coord transform fns
# (as this avoids divide by 0 errors)
d2_min = 0.05
tau_max = np.log(2 * R_0 / d2_min)
n_tau = 200
tau = np.linspace(tau_c, tau_max, n_tau)
print("tau = ", tau)

n_sigma = 150
sigma = np.linspace(-np.pi, np.pi, n_sigma)
print("sigma = ", sigma)
# %%
tau, sigma = np.meshgrid(tau, sigma)
print("tau = ", tau)
print("sigma = ", sigma)

# Handy combination
Delta = np.cosh(tau) - np.cos(sigma)
Deltac = np.cosh(tau_c) - np.cos(sigma_c)

# Convert to cylindrical
R, Z = toroidal_to_cylindrical(R_0=R_0, z_0=Z_0, tau=tau, sigma=sigma)

print("R = ", R)
print("Z = ", Z)

# %%
# # Check matlab factorial_term is same as term in paper
# m_max = 5
# from math import factorial
# from scipy.special import factorial2
# print(factorial(0))
# print(factorial(10))
# matlab_values = []
# paper_values = []
# for m in range(m_max):
#     print("Current m = ", m)
#     print("Comparing factorial_term from MATLAB to paper")
#     if m == 0:  # noqa: SIM108
#         factorial_term = 1
#     else:
#         factorial_term = np.prod(1 + 0.5 / np.arange(1, m + 1))
#     print("factorial term from MATLAB = ", factorial_term)
#     missing_term_from_paper = 1 / (2**m) * (factorial2(2 * m + 1) / factorial(m))
#     print("term missing from paper = ", missing_term_from_paper)
#     matlab_values.append(factorial_term)
#     paper_values.append(missing_term_from_paper)
# print("matlab values = ", matlab_values)
# print("paper values = ", paper_values)

# TODO need to put numbers in and compare the factorial_term from matlab to that which
# comes from the paper to see if they're the same?
# ^ They are the same, will ask Oliver to explain how/why it simplifies to the
# factorial_term

# # %%
# # Calc coeffs - uses eq (19) from paper

# m_max = 5
# Am_cos = np.zeros(m_max)
# Am_sin = np.zeros(m_max)

# for m in range(m_max):
#     if m == 0:
#         factorial_term = 1
#     else:
#         factorial_term = np.prod(1 + 0.5 / np.arange(1, m + 1))
#     A_m = (
#         (MU_0 * I_c / 2 ** (5 / 2)) * factorial_term * (np.sinh(tau_c) / np.sqrt(Deltac))
#         # * legendreP
#     )  # TODO do legendre P
#     print("m = ", m, "A_m (without legendre yet) = ", A_m)
# print(MU_0)


# %%
# Legendre P and Q half integer - copied across from replicating_legendre_fns_from_matlab_work.ex.py

from math import factorial

from scipy.special import gamma, poch


# first try with built in functions
# F fn used in both legendreP and legendreQ
def F_hypergeometric(a, b, c, z, n_max):
    F = 0
    for s in range(n_max + 1):
        F += (poch(a, s) * poch(b, s)) / (gamma(c + s) * factorial(s)) * z**s
    return F


# set default n_max = 20 to match matlab
def myLegendreP(lam, mu, x, n_max=20):
    a = 1 / 2 * (mu - lam)
    b = 1 / 2 * (mu - lam + 1)
    c = mu + 1
    z = 1 - 1 / (x**2)
    F_sum = F_hypergeometric(a=a, b=b, c=c, z=z, n_max=n_max)  # noqa: N806
    legP = 2 ** (-mu) * x ** (lam - mu) * (x**2 - 1) ** (mu / 2) * F_sum  # noqa: N806
    return legP


def myLegendreQ(lam, mu, x, n_max=20):
    a = 1 / 2 * (lam + mu) + 1
    b = 1 / 2 * (lam + mu + 1)
    c = lam + 3 / 2
    z = 1 / (x**2)
    F_sum = F_hypergeometric(a=a, b=b, c=c, z=z, n_max=n_max)  # noqa: N806
    legQ = (
        (np.pi ** (1 / 2) * (x**2 - 1) ** (mu / 2))
        / (2 ** (lam + 1) * x ** (lam + mu + 1))
        * F_sum
    )
    if type(legQ) == np.float64:
        if x == 1:
            legQ == np.inf
    else:
        legQ[x == 1] = np.inf
    return legQ

# %%


m_max = 5

for m in range(m_max):

# %%
# Will compare to the values from Coilset1 initially (instead of fiesta output)
