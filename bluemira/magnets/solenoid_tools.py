# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
Tools for simple solenoid calculations.
"""

import numpy as np

from bluemira.base.constants import MU_0
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.magnetostatics.semianalytic_2d import semianalytic_Bx, semianalytic_Bz


def calculate_B_max(
    rho_j: float, r_inner: float, r_outer: float, height: float, z_0: float = 0.0
) -> float:
    """
    Calculate the maximum self-field in a solenoid. This is always located
    at (r_inner, z_0)

    Parameters
    ----------
    rho_j:
        Current density across the solenoid winding pack [A/m^2]
    r_inner:
        Solenoid inner radius [m]
    r_outer:
        Solenoid outer radius [m]
    height:
        Solenoid vertical extent [m]

    Returns
    -------
    Maximum field in a solenoid [T]

    Notes
    -----
    Cross-checked graphically with data from Boom and Livingstone, "Superconducting solenoids",
    1962
    """
    dxc = 0.5 * (r_outer - r_inner)
    xc = r_inner + dxc
    dzc = 0.5 * height
    x_bmax = r_inner
    I = rho_j * (height * (r_outer - r_inner))
    Bx_max = I * semianalytic_Bx(xc, z_0, x_bmax, z_0, dxc, dzc)
    Bz_max = I * semianalytic_Bz(xc, z_0, x_bmax, z_0, dxc, dzc)
    return np.hypot(Bx_max, Bz_max)


def calculate_hoop_stress(
    B_in: float,
    B_out: float,
    rho_j: float,
    r_inner: float,
    r_outer: float,
    r: float,
    poisson_ratio: float = 0.3,
) -> float:
    """
    Calculate the hoop stress at a radial location in a solenoid

    Parameters
    ----------
    B_in:
        Field at the inside edge of the solenoid [T]
    B_out:
        Field at the outside edge of the solenoid [T]
    rho_j:
        Current density across the solenoid winding pack [A/m^2]
    r_inner:
        Solenoid inner radius [m]
    r_outer:
        Solenoid outer radius [m]
    r:
        Radius at which to calculate [m]
    poisson_ratio:
        Poisson ratio of the material

    Returns
    -------
    Hoop stress at the radial location [Pa]

    Notes
    -----
    Must still factor in the fraction of load-bearing material
    """
    alpha = r_outer / r_inner
    eps = r / r_inner
    nu = poisson_ratio

    K = (alpha * B_in - B_out) * rho_j * r_inner / (alpha - 1)  # noqa: N806
    M = (B_in - B_out) * rho_j * r_inner / (alpha - 1)  # noqa: N806
    a = K * (2 + nu) / (3 * (alpha + 1))
    b = (
        1.0
        + alpha
        + alpha**2 * (1 + 1 / eps**2)
        - eps * (1 + 2 * nu) * (alpha + 1) / (2 + nu)
    )
    c = M * (3 + nu) / 8
    d = 1.0 + alpha**2 * (1 + 1 / eps**2) - eps**2 * (1 + 3 * nu) / (3 + nu)
    hoop_stress = a * b - c * d

    return hoop_stress


def calculate_axial_stress(
    r_inner: float, r_outer: float, height: float, current: float
) -> float:
    """
    Calculate the axial stress in a solenoid

    Parameters
    ----------
    r_inner:
        Solenoid inner radius [m]
    r_outer:
        Solenoid outer radius [m]
    height:
        Solenoid vertical extent [m]
    current:
        Current in the solenoid [A]

    Returns
    -------
    Axial stress [Pa]

    Notes
    -----
    Must still factor in the fraction of load-bearing material
    """
    hh = 0.5 * height
    a = -0.5 * MU_0 * current**2
    # TODO: I don't trust things without pi
    b = 0
    c = 0

    force = a * (b - c)
    area = np.pi * (r_outer**2 - r_inner**2)
    return force / area


def calculate_flux_max(B_max: float, r_inner: float, r_outer: float) -> float:
    """
    Calculate the maximum flux achievable from a solenoid

    Parameters
    ----------
    B_max:
        Maximum field in the solenoid [T]
    r_inner:
        Solenoid inner radius [m]
    r_outer:
        Solenoid outer radius [m]

    Returns
    -------
    Maximum flux achievable from a solenoid [V.s]
    """
    return np.pi / 3 * B_max * (r_outer**2 + r_inner**2 + r_outer * r_inner)


if __name__ == "__main__":
    import csv

    import matplotlib.pyplot as plt
    import pandas as pd

    from bluemira.magnetostatics.semianalytic_2d import semianalytic_Bx, semianalytic_Bz

    data_101 = pd.read_csv("1_01.csv")
    data_1025 = pd.read_csv("1_025.csv")
    data_11 = pd.read_csv("1_1.csv")
    data_20 = pd.read_csv("2_0.csv")

    def K_semianalytic(
        rho_j: float, r_inner: float, r_outer: float, height: float
    ) -> float:
        dxc = 0.5 * (r_outer - r_inner)
        xc = r_inner + dxc
        zc = 1
        dzc = 0.5 * height
        x = r_inner
        z = 1
        Bx_max = semianalytic_Bx(xc, zc, x, z, dxc, dzc)
        Bz_max = semianalytic_Bz(xc, zc, x, z, dxc, dzc)
        I = rho_j * (height * (r_outer - r_inner))
        Bx_max *= I
        Bz_max *= I
        Bx_0 = semianalytic_Bx(xc, zc, 1e-6, 1, dxc, dzc)
        Bz_0 = semianalytic_Bz(xc, zc, 1e-6, 1, dxc, dzc)
        Bx_0 *= I
        Bz_0 *= I
        return np.hypot(Bx_max, Bz_max) / np.hypot(Bx_0, Bz_0)

    def K_semianalytic_ab(alpha, beta, rho_j, r_inner):
        r_outer = alpha * r_inner
        half_height = beta * r_inner
        return K_semianalytic(rho_j, r_inner, r_outer, 2 * half_height)

    rho_j = 5e6
    r_inner = 10.0

    n1, n2 = 50, 60
    alpha = np.linspace(1.0001, 3.2, n1)
    beta = np.linspace(0.0001, 3.2, n2)
    x, y = np.meshgrid(alpha, beta, indexing="ij")

    Kprocess = np.zeros((n1, n2))
    Ksemi = np.zeros((n1, n2))

    for i, a in enumerate(alpha):
        for j, b in enumerate(beta):
            if a <= 2.0 and (0.5 < b < 3.0):
                Kprocess[i, j] = calculate_K_process(a, b, rho_j, r_inner)
            else:
                Kprocess[i, j] = np.nan
            Ksemi[i, j] = K_semianalytic_ab(a, b, rho_j, r_inner)

    levels = [
        1.0075,
        1.00875,
        1.01,
        1.0125,
        1.015,
        1.02,
        1.025,
        1.0325,
        1.04,
        1.055,
        1.07,
        1.1,
        1.15,
        1.2,
        1.3,
        1.5,
        1.75,
        2.0,
        2.5,
        2.8,
    ]
    f, ax = plt.subplots()
    # cm = ax.contour(x, y, Kprocess, cmap="viridis", levels=levels)
    # cb = f.colorbar(cm, ax=ax, pad=0)
    # cb.set_label("PROCESS $k = B_{max}/B_{0}$")
    cm = ax.contour(x, y, Ksemi, cmap="plasma", levels=levels)
    cb = f.colorbar(cm, ax=ax, pad=0.04)
    cb.set_label("BLUEMIRA $k = B_{max}/B_{0}$")
    ax.scatter(data_101["alpha"], data_101["beta"], label="k=1.01")
    ax.scatter(data_1025["alpha"], data_1025["beta"], label="k=1.025")
    ax.scatter(data_11["alpha"], data_11["beta"], label="k=1.1")
    ax.scatter(data_20["alpha"], data_20["beta"], label="k=2.0")
    leg = ax.legend()
    leg.set_title("Boom and Livingstone 1962 data")
    ax.set_xlabel("$\\alpha$")
    ax.set_xlim([1.0, 3.2])
    ax.set_ylabel("$\\beta$")
    ax.set_ylim([0, 3.2])
    plt.show()
