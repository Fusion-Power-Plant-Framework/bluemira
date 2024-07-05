# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Toroidal coordinate transform functions.
"""

import numpy as np


def toroidal_to_cylindrical(R_0: float, z_0: float, tau: np.ndarray, sigma: np.ndarray):
    """
    Convert from toroidal coordinates to cylindrical coordinates in the poloidal plane
    Toroidal coordinates are denoted by (\\tau, \\sigma, \\phi)
    Cylindrical coordinates are denoted by (r, z, \\phi)
    We are in the poloidal plane so take the angle \\phi = 0

    .. math::
        R = R_{0} \\frac{\\sinh{\\tau}}{\\cosh{\\tau} - \\cos{\\sigma}}
        z = R_{0} \\frac{\\sin{\\tau}}{\\cosh{\\tau} - \\cos{\\sigma}} + z_{0}

    Parameters
    ----------
    R_0:
        r coordinate of focus in poloidal plane
    z_0:
        z coordinate of focus in poloidal plane
    tau:
        the tau coordinates to transform
    sigma:
        the sigma coordinates to transform

    Returns
    -------
    [R, Z]:
        Array containing transformed coordinates in cylindrical form
    """
    R = R_0 * np.sinh(tau) / (np.cosh(tau) - np.cos(sigma))  # noqa: N806
    Z = R_0 * np.sin(sigma) / (np.cosh(tau) - np.cos(sigma)) + z_0  # noqa: N806
    return [R, Z]


def cylindrical_to_toroidal(R_0: float, z_0: float, R: np.ndarray, Z: np.ndarray):  # noqa: N803
    """
    Convert from cylindrical coordinates to toroidal coordinates in the poloidal plane
    Toroidal coordinates are denoted by (\\tau, \\sigma, \\phi)
    Cylindrical coordinates are denoted by (r, z, \\phi)
    We are in the poloidal plane so take the angle \\phi = 0

    .. math::
        \\tau = \\ln\\frac{d_{1}}{d_{2}}
        \\sigma = sign(z - z_{0}) \\arccos\\frac{d_{1}^2 + d_{2}^2
                                        - 4 R_{0}^2}{2 d_{1} d_{2}}

        d_{1}^2 = (R + R_{0})^2 + (z - z_{0})^2
        d_{2}^2 = (R - R_{0})^2 + (z - z_{0})^2

    Parameters
    ----------
    R_0:
        r coordinate of focus in poloidal plane
    z_0:
        z coordinate of focus in poloidal plane
    R:
        the r coordinates to transform
    Z:
        the z coordinates to transform

    Returns
    -------
    [tau, sigma]:
        Array containing transformed coordinates in toroidal form
    """
    d_1 = np.sqrt((R + R_0) ** 2 + (Z - z_0) ** 2)
    d_2 = np.sqrt((R - R_0) ** 2 + (Z - z_0) ** 2)
    tau = np.log(d_1 / d_2)
    sigma = np.sign(Z - z_0) * np.arccos(
        (d_1**2 + d_2**2 - 4 * R_0**2) / (2 * d_1 * d_2)
    )
    return [tau, sigma]
