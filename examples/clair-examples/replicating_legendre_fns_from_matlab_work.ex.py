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
Replicating the legendre fns from matlab
Aim: recreate plots produced from toroidal_harmonics.m
"""
# TODO check them from the paper (and against those in other papers)
# TODO put them in code here

# %%
# Need the focus point and to convert to toroidal coords

import matplotlib.pyplot as plt
import numpy as np

from bluemira.utilities.tools import cylindrical_to_toroidal

r = np.linspace(0, 6, 100)
z = np.linspace(-6, 6, 100)
print(f"r = {r}")
print(f"z = {z}")
R, Z = np.meshgrid(r, z)

R_0 = 3.0
Z_0 = 1.0

tau, sigma = cylindrical_to_toroidal(R_0=R_0, R=R, z_0=Z_0, Z=Z)
print(f"tau = {tau}")
print(f"sigma = {sigma}")

# wire location
R_c = 5.0
Z_c = 1.0
I_c = 5e6

tau_c, sigma_c = cylindrical_to_toroidal(R_0=R_0, z_0=Z_0, R=R_c, Z=Z_c)
print(f"tau_c = {tau_c}")
print(f"sigma_c = {sigma_c}")


# %%
# Legendre P
# needs 4 args - lambda, mu, x, n_max (to match matlab)
# lambda = n in paper
# mu = nu in paper

# recursively calculated in matlab (checked and it works out)
# also try with built in functions

from math import factorial

from scipy.special import gamma, poch


# first try with built in functions
# F fn used in both legendreP and legendreQ
def F_hypergeometric(a, b, c, z, n_max):
    F = 0
    # print("F initial = ", F)
    for s in range(n_max + 1):
        F += (poch(a, s) * poch(b, s)) / (gamma(c + s) * factorial(s)) * z**s
        # print(f"summed F at stage {s} = {F}")
    return F


# set default n_max = 20 to match matlab
def myLegendreP(lam, mu, x, n_max=20):
    a = 1 / 2 * (mu - lam)
    b = 1 / 2 * (mu - lam + 1)
    c = mu + 1
    z = 1 - 1 / (x**2)
    F_sum = F_hypergeometric(a=a, b=b, c=c, z=z, n_max=n_max)  # noqa: N806
    print("in myLegendreP:")
    print(f"f_sum = {F_sum}")
    legP = 2 ** (-mu) * x ** (lam - mu) * (x**2 - 1) ** (mu / 2) * F_sum  # noqa: N806
    print(f"legendreP = {legP}")
    return legP


print("legendreP cosh tauc:")
myLegendreP(1 - 1 / 2, 1, np.cosh(tau_c))

# TODO check if i can pass in array of x?
x = np.cosh(tau)
print("legendreP cosh tau (array)")
myLegendreP(1 - 1 / 2, 1, x)


def myLegendreQ(lam, mu, x, n_max=20):
    a = 1 / 2 * (lam + mu) + 1
    b = 1 / 2 * (lam + mu + 1)
    c = lam + 3 / 2
    z = 1 / (x**2)
    F_sum = F_hypergeometric(a=a, b=b, c=c, z=z, n_max=n_max)  # noqa: N806
    print("in LegendreQ:")
    print(f"f_sum = {F_sum}")
    legQ = (
        (np.pi ** (1 / 2) * (x**2 - 1) ** (mu / 2))
        / (2 ** (lam + 1) * x ** (lam + mu + 1))
        * F_sum
    )
    print(f"shape x = {x.shape}")
    print(f"shape legQ = {legQ.shape}")
    print(f"non-edited legendreQ = {legQ}")
    if type(legQ) == np.float64:
        if x == 1:
            legQ == np.inf
    else:
        legQ[x == 1] = np.inf
    legQcopy = legQ
    # np.nan_to_num(legQcopy, nan=10)
    print(f"legendreQ = {legQ}")
    return legQ


# NOTE: see a small difference in the values between here and the matlab eg matlab
# has -1.6961e-2 and python has 1.68837940e-2 so are close but not entirely the same


# TODO debug and replace nans with something else mid run then see how it changes the
# plot eg set all to 1
# TODO zoom in on centre to see what's going on


print("legendreQ cosh tauc:")
myLegendreQ(1 - 1 / 2, 1, np.cosh(tau_c))

x = np.cosh(tau)
print("legendreq cosh tau (array)")
ans = myLegendreQ(1 - 1 / 2, 1, x)

# print(np.cosh(tau)[1][0])
# print(ans[1][0])


# %%
# Try to replicate toroidal_harmonics.m now


nu = np.arange(0, 5)
fig_sin, axs_sin = plt.subplots(1, len(nu))
# for ax in axs_sin:
#     ax.set_xlim([2.5, 3.5])
#     ax.set_ylim([-1, 1])
fig_sin.suptitle("sin plots")
fig_cos, axs_cos = plt.subplots(1, len(nu))
# for ax in axs_cos:

#     ax.set_xlim([2.5, 3.5])
#     ax.set_ylim([-1, 1])
fig_cos.suptitle("cos plots")
psi_sin_python = []
psi_cos_python = []
for i_nu in range(len(nu)):
    foo = (
        R
        * np.sqrt(np.cosh(tau) - np.cos(sigma))
        * myLegendreQ(nu[i_nu] - 1 / 2, 1, np.cosh(tau))
    )
    psi_sin = foo * np.sin(nu[i_nu] * sigma)
    psi_cos = foo * np.cos(nu[i_nu] * sigma)
    psi_sin_python.append(psi_sin)
    psi_cos_python.append(psi_cos)
    axs_sin[i_nu].contour(R, Z, psi_sin, 50)
    axs_cos[i_nu].contour(R, Z, psi_cos, 50)
print(f"psi_sin = {psi_sin_python}")
print(f"psi_cos = {psi_cos_python}")

# %%
# # next try recursively to match matlab (and compare to above)
# def mylegendreQ_recursive(lam, mu, x, n_max=20):
#     n = np.arange(0, n_max)
#     an = np.zeros_like(n)
#     bn = np.zeros_like(n)
#     cn = np.zeros_like(n)
#     nfactorial = np.zeros_like(n)
#     # args for F from eq (29)
#     a = 1 / 2 * (lam + mu) + 1
#     b = 1 / 2 * (lam + mu + 1)
#     c = lam + 3 / 2

#     # set first entry of arrays (corresponds to s=0)
#     an[0] = 1
#     bn[0] = 1
#     cn[0] = gamma(c)
#     # instead of using this and recurrence relation, just use factorial function?
#     # nfactorial[0] = factorial(0)  # = 1
#     for i in n:
#         nfactorial[i] = factorial(i)
#         # recurrence relations (worked out on paper to get them to make sense)
#     for i in range(1, len(n)):
#         an[i] = (a + n[i] - 1) * an[i - 1]
#         bn[i] = (b + n[i] - 1) * bn[i - 1]
#         cn[i] = (c + n[i] - 1) * cn[i - 1]
#         # nfactorial[i] = n[i] * nfactorial[i-1] # commented b/c using factorial fn now
#         # instead of recurrence relation?

#         # create array of coeffs = (a)s (b)s / (gamma(c+s) * s!) (here n = s)
#     coeffs = an * bn / (cn * nfactorial)
#     print(coeffs)
#     print("coeffs shape = ", coeffs.shape)
#     print("x shape = ", x.shape)

#     # TODO reshaping stuff from the matlab
#     dims_x = len(x.shape)
#     coeffs = np.reshape(coeffs, (1,) * dims_x + (len(n),))
#     n = np.reshape(n, (1,) * dims_x + (len(n),))

#     y = np.sum(coeffs * x ** (-2 * n), axis=-1)
#     y = (
#         np.sqrt(np.pi)
#         * (x**2 - 1) ** (0.5 * mu)
#         / (2 ** (lam + 1) * x ** (lam + mu + 1))
#         * y
#     )

#     # Singular at x=1
#     y[x == 1] = np.inf
#     # TODO - 18th sept - just use my version and see if it matches oliver's recursive one

# copied from matlab to python converter
import numpy as np


def mylegendreQ_copied(lambda_, mu, x, n_max=20):
    # Evaluate coefficients in sum; gamma functions so simple recurrence relations
    n = np.arange(n_max + 1)
    an = np.zeros_like(n, dtype=float)
    bn = np.zeros_like(n, dtype=float)
    cn = np.zeros_like(n, dtype=float)
    nfactorial = np.zeros_like(n, dtype=float)

    a = 0.5 * mu + 0.5 * lambda_ + 1
    b = 0.5 * mu + 0.5 * lambda_ + 0.5
    c = lambda_ + 1.5

    an[0] = 1
    bn[0] = 1
    cn[0] = gamma(c)
    nfactorial[0] = 1

    for i in range(1, len(n)):
        an[i] = (a + n[i] - 1) * an[i - 1]
        bn[i] = (b + n[i] - 1) * bn[i - 1]
        cn[i] = (c + n[i] - 1) * cn[i - 1]
        nfactorial[i] = n[i] * nfactorial[i - 1]
    return an, bn, cn, nfactorial
    # coeffts = an * bn / (cn * nfactorial)

    # # Apply coeffts
    # dims_x = len(x.shape)
    # coeffts = coeffts.reshape(*(1 for _ in range(dims_x)), len(n))
    # print(f"coeffts shape = {coeffts.shape}")
    # n = n.reshape(*(1 for _ in range(dims_x)), len(n))

    # y = np.sum(coeffts * x ** (-2 * n), axis=-1)
    # y = (
    #     np.sqrt(np.pi)
    #     * (x**2 - 1) ** (0.5 * mu)
    #     / (2 ** (lambda_ + 1) * x ** (lambda_ + mu + 1))
    #     * y
    # )

    # # Singular at x=1
    # y[x == 1] = np.inf

    # return y


# print("python legendreQ cosh tauc:")
# myLegendreQ(1 - 1 / 2, 1, np.cosh(tau_c))

# x = np.cosh(tau)
# print("python legendreq cosh tau (array)")
# ans = myLegendreQ(1 - 1 / 2, 1, x)

print("legendreQ cosh tauc:")
mylegendreQ_copied(1 - 1 / 2, 1, np.cosh(tau_c))

x = np.cosh(tau)
print("legendreq cosh tau (array)")
mylegendreQ_copied(1 - 1 / 2, 1, x)

# %%
# nu = np.arange(0, 5)
# fig_sin, axs_sin = plt.subplots(1, len(nu))
# fig_sin.suptitle("sin plots")
# fig_cos, axs_cos = plt.subplots(1, len(nu))
# fig_cos.suptitle("cos plots")

# for i_nu in range(len(nu)):
#     foo = (
#         R
#         * np.sqrt(np.cosh(tau) - np.cos(sigma))
#         * mylegendreQ_copied(nu[i_nu] - 1 / 2, 1, np.cosh(tau))
#     )
#     psi_sin = foo * np.sin(nu[i_nu] * sigma)
#     psi_cos = foo * np.cos(nu[i_nu] * sigma)
#     axs_sin[i_nu].contour(R, Z, psi_sin, 50)
#     axs_cos[i_nu].contour(R, Z, psi_cos, 50)
# %%
# Legendre Q


# %%
# trying to get matlab variables saved in python
# from scipy.io import loadmat

# loadmat("/home/clair/development/matlab-stuff/toroidal_harmonics2")

# want to use the matlab values to compare

from oct2py import octave

octave.addpath("/home/clair/development/matlab-stuff")

# %%
x = np.cosh(tau)
x_c = np.cosh(tau_c)
# y = octave.legendreQ(0.5, 1, x_c)
y_octave = octave.legendreQ(0.5, 1, x)
plt.plot(x, y_octave)
print(repr(y_octave))

fig1, axs1 = plt.subplots(1, len(nu))
fig1.suptitle("sin plots MATLAB")

fig2, axs2 = plt.subplots(1, len(nu))
fig2.suptitle("cos plots MATLAB")

# nu = [0, 1, 2, 3, 4]
# for i_nu in range(len(nu)):
#     legenQ = octave.legendreQ(nu[i_nu] - 0.5, 1, np.cosh(tau))
#     foo = R * np.sqrt(np.cosh(tau) - np.cos(sigma)) * legenQ
#     psi_sin = foo * np.sin(nu[i_nu] * sigma)
#     psi_cos = foo * np.cos(nu[i_nu] * sigma)
#     axs1[i_nu].contour(R, Z, psi_sin, 50)
#     axs2[i_nu].contour(R, Z, psi_cos, 50)

# %%
# octave.run("toroidal_harmonics.m")

[
    psi_sin_octave_1,
    psi_sin_octave_2,
    psi_sin_octave_3,
    psi_sin_octave_4,
    psi_sin_octave_5,
    psi_cos_octave_1,
    psi_cos_octave_2,
    psi_cos_octave_3,
    psi_cos_octave_4,
    psi_cos_octave_5,
] = octave.toroidal_harmonics(R_0, Z_0, nout=10)


# %%
psi_sin_octave_arry = np.array([
    psi_sin_octave_1,
    psi_sin_octave_2,
    psi_sin_octave_3,
    psi_sin_octave_4,
    psi_sin_octave_5,
])

psi_cos_octave_arry = np.array([
    psi_cos_octave_1,
    psi_cos_octave_2,
    psi_cos_octave_3,
    psi_cos_octave_4,
    psi_cos_octave_5,
])

fig1, axs1 = plt.subplots(1, len(nu))
fig1.suptitle("sin plots MATLAB")

fig2, axs2 = plt.subplots(1, len(nu))
fig2.suptitle("cos plots MATLAB")
for i in range(5):
    axs1[i].contour(R, Z, psi_sin_octave_arry[i], 50)
    axs2[i].contour(R, Z, psi_cos_octave_arry[i], 50)


# why are there always lines going through the focus?

# %%
for i in range(100):
    plt.plot(r, psi_sin_octave_arry[0][0][i])
