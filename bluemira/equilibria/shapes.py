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
Useful parameterisations for plasma flux surface shapes.
"""

import numpy as np

from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.geometry.tools import interpolate_bspline
from bluemira.geometry.wire import BluemiraWire
from bluemira.utilities.opt_variables import BoundedVariable, OptVariables

__all__ = [
    "flux_surface_cunningham",
    "flux_surface_johner",
    "flux_surface_manickam",
    "JohnerLCFS",
]


def flux_surface_cunningham(r_0, z_0, a, kappa, delta, delta2=None, n=20):
    """
    As featured in Geof Cunningham's FIESTA (shape_fun)

    Parameters
    ----------
    r_0: float
        Plasma geometric major radius [m]
    z_0: float
        Plasma geometric vertical height [m]
    a: float
        Plasma geometric minor radius [m]
    kappa: float
        Plasma elongation
    delta: float
        Plasma triangularity
    delta2: float
        Plasma "delta2" curliness?
    n: int
        Number of points

    Returns
    -------
    flux_surface: Coordinates
        Plasma flux surface shape
    """
    t = np.linspace(0, 2 * np.pi, n)[:-1]  # Theta

    if delta2 is None:
        x = r_0 + a * np.cos(t + delta * np.sin(t))
    else:
        x = r_0 + a * np.cos(t + delta * np.sin(t) + delta2 * np.sin(2 * t))
    z = z_0 + a * kappa * np.sin(t)
    return Coordinates({"x": x, "z": z})


def flux_surface_manickam(r_0, z_0, a, kappa=1, delta=0, indent=0, n=20):
    """
    J. Manickam, Nucl. Fusion 24 595 (1984)

    Parameters
    ----------
    r_0: float
        Plasma geometric major radius [m]
    z_0: float
        Plasma geometric vertical height [m]
    a: float
        Plasma geometric minor radius [m]
    kappa: float
        Plasma elongation
    delta: float
        Plasma triangularity
    indent: float
        Plasma indentation (beaniness)
    n: int
        Number of points

    Returns
    -------
    flux_surface: Coordinates
        Plasma flux surface shape
    """
    t = np.linspace(0, 2 * np.pi, n)[:-1]  # Theta
    x = r_0 - indent + (a + indent * np.cos(t)) * np.cos(t + delta * np.sin(t))
    z = z_0 + kappa * a * np.sin(t)
    return Coordinates({"x": x, "z": z})


def calc_t_neg(delta, kappa, phi_neg):
    return np.tan(phi_neg) * (1 - delta) / kappa


def calc_t_pos(delta, kappa, phi_pos):
    return np.tan(phi_pos) * (1 + delta) / kappa


def calc_angles_neg_below(delta, kappa, t_neg):
    alpha_0_neg = -(delta - (1 + delta) * t_neg) / (1 - 2 * t_neg)
    alpha_neg = (1 - delta) * (1 - t_neg) / (1 - 2 * t_neg)
    beta_neg = kappa * (1 - t_neg) / np.sqrt(1 - 2 * t_neg)
    return alpha_0_neg, alpha_neg, beta_neg


def calc_angles_neg_above(delta, kappa, t_neg):
    alpha_0_neg = -((1 + delta) * t_neg - delta) / (2 * t_neg - 1)
    alpha_neg = (1 - delta) * (1 - t_neg) / (2 * t_neg - 1)
    beta_neg = kappa * (1 - t_neg) / np.sqrt(2 * t_neg - 1)
    return alpha_0_neg, alpha_neg, beta_neg


def calc_angles_pos_below(delta, kappa, t_pos):
    alpha_0_pos = -(delta + (1 - delta) * t_pos) / (1 - 2 * t_pos)
    alpha_pos = (1 + delta) * (1 - t_pos) / (1 - 2 * t_pos)
    beta_pos = kappa * (1 - t_pos) / np.sqrt(1 - 2 * t_pos)
    return alpha_0_pos, alpha_pos, beta_pos


def calc_angles_pos_above(delta, kappa, t_pos):
    alpha_0_pos = ((1 - delta) * t_pos + delta) / (2 * t_pos - 1)
    alpha_pos = -(1 + delta) * (1 - t_pos) / (2 * t_pos - 1)
    beta_pos = kappa * (1 - t_pos) / np.sqrt(2 * t_pos - 1)
    return alpha_0_pos, alpha_pos, beta_pos


def flux_surface_johner_quadrants(
    r_0,
    z_0,
    a,
    kappa_u,
    kappa_l,
    delta_u,
    delta_l,
    psi_u_neg,
    psi_u_pos,
    psi_l_neg,
    psi_l_pos,
    n=100,
):
    """
    Initial plasma shape parametrerisation from HELIOS author
    J. Johner (CEA). Sets initial separatrix shape for the plasma core
    (does not handle divertor target points or legs).
    Can handle:
    - DN (positive, negative delta) [TESTED]
    - SN (positive, negative delta) (upper, lower) [TESTED]

    Parameters
    ----------
    r_0: float
        Major radius [m]
    z_0: float
        Vertical position of major radius [m]
    a: float
        Minor radius [m]
    kappa_u: float
        Upper elongation at the plasma edge (psi_n=1)
    kappa_l: float
        Lower elongation at the plasma edge (psi_n=1)
    delta_u: float
        Upper triangularity at the plasma edge (psi_n=1)
    delta_l: float
        Lower triangularity at the plasma edge (psi_n=1)
    psi_u_neg: float
        Upper inner angle [°]
    psi_u_pos: float
        Upper outer angle [°]
    psi_l_neg: float
        Lower inner angle [°]
    psi_l_pos: float
        Lower outer angle [°]
    n: int (defeault = 100)
        Number of point to generate on the flux surface

    Returns
    -------
    flux_surface: Coordinates
        Plasma flux surface shape
    """
    # May appear tempting to refactor, but equations subtly different
    # Careful of bad angles or invalid plasma shape parameters
    if delta_u < 0 and delta_l < 0:
        delta_u *= -1
        delta_l *= -1
        negative = True
    else:
        negative = False
    psi_u_neg, psi_u_pos, psi_l_neg, psi_l_pos = [
        np.deg2rad(i) for i in [psi_u_neg, psi_u_pos, psi_l_neg, psi_l_pos]
    ]

    n_pts = int(n / 4)
    # inner upper
    t_neg = calc_t_neg(delta_u, kappa_u, psi_u_neg)
    if t_neg < 0.5:
        theta_u_neg = np.arcsin(np.sqrt(1 - 2 * t_neg) / (1 - t_neg))
        alpha_0_neg, alpha_neg, beta_neg = calc_angles_neg_below(delta_u, kappa_u, t_neg)
        theta = np.linspace(0, theta_u_neg, n_pts)
        x_ui = alpha_0_neg - alpha_neg * np.cos(theta)
        z_ui = beta_neg * np.sin(theta)
    elif t_neg == 0.5:
        z_ui = np.linspace(0, kappa_u, n_pts)
        x_ui = -1 + z_ui**2 * (1 - delta_u) / kappa_u**2
    elif t_neg == 1:
        z_ui = np.linspace(0, kappa_u, n_pts)
        x_ui = -1 + z_ui * (1 - delta_u) / kappa_u
    elif t_neg > 0.5:
        phi_u_neg = np.arcsinh(np.sqrt(2 * t_neg - 1) / (1 - t_neg))
        alpha_0_neg, alpha_neg, beta_neg = calc_angles_neg_above(delta_u, kappa_u, t_neg)
        phi = np.linspace(0, phi_u_neg, n_pts)
        x_ui = alpha_0_neg + alpha_neg * np.cosh(phi)
        z_ui = beta_neg * np.sinh(phi)
    else:
        raise ValueError("Something is wrong with the Johner parameterisation.")
    # inner lower
    t_neg = calc_t_neg(delta_l, kappa_l, psi_l_neg)
    if t_neg < 0.5:
        theta_u_neg = np.arcsin(np.sqrt(1 - 2 * t_neg) / (1 - t_neg))
        alpha_0_neg, alpha_neg, beta_neg = calc_angles_neg_below(delta_l, kappa_l, t_neg)
        theta = np.linspace(-theta_u_neg, 0, n_pts)
        x_li = alpha_0_neg - alpha_neg * np.cos(theta)
        z_li = beta_neg * np.sin(theta)
    elif t_neg == 0.5:
        z_li = np.linspace(-kappa_u, 0, n_pts)
        x_li = -1 + z_li**2 * (1 - delta_l) / kappa_l**2
    elif t_neg == 1:
        z_li = np.linspace(-kappa_l, 0, n_pts)
        x_li = -1 + z_li * (1 - delta_l) / kappa_l
    elif t_neg > 0.5:
        phi_u_neg = np.arcsinh(np.sqrt(2 * t_neg - 1) / (1 - t_neg))
        alpha_0_neg, alpha_neg, beta_neg = calc_angles_neg_above(delta_l, kappa_l, t_neg)
        phi = np.linspace(-phi_u_neg, 0, n_pts)
        x_li = alpha_0_neg + alpha_neg * np.cosh(phi)
        z_li = beta_neg * np.sinh(phi)
    else:
        raise ValueError("Something is wrong with the Johner parameterisation.")
    # outer upper
    t_pos = calc_t_pos(delta_u, kappa_u, psi_u_pos)
    if t_pos < 0.5:
        theta_u_pos = np.arcsin(np.sqrt(1 - 2 * t_pos) / (1 - t_pos))
        alpha_0_pos, alpha_pos, beta_pos = calc_angles_pos_below(delta_u, kappa_u, t_pos)
        theta = np.linspace(0, theta_u_pos, n_pts)
        x_uo = alpha_0_pos + alpha_pos * np.cos(theta)
        z_uo = beta_pos * np.sin(theta)
    elif t_pos == 0.5:
        z_uo = np.linspace(0, kappa_u, n_pts)
        x_uo = -1 - z_uo**2 * (1 + delta_u) / kappa_u**2
    elif t_pos == 1:
        z_uo = np.linspace(0, kappa_u, n_pts)
        x_uo = 1 - z_uo * (1 + delta_u) / kappa_u
    elif t_pos > 0.5:
        phi_u_pos = np.arcsinh(np.sqrt(2 * t_pos - 1) / (1 - t_pos))
        alpha_0_pos, alpha_pos, beta_pos = calc_angles_pos_above(delta_u, kappa_u, t_pos)
        phi = np.linspace(0, phi_u_pos, n_pts)
        x_uo = alpha_0_pos + alpha_pos * np.cosh(phi)
        z_uo = beta_pos * np.sinh(phi)
    else:
        raise ValueError("Something is wrong with the Johner parameterisation.")
    # outer lower
    t_pos = calc_t_pos(delta_l, kappa_l, psi_l_pos)
    if t_pos < 0.5:
        theta_l_pos = np.arcsin(np.sqrt(1 - 2 * t_pos) / (1 - t_pos))
        alpha_0_pos, alpha_pos, beta_pos = calc_angles_pos_below(delta_l, kappa_l, t_pos)
        theta = np.linspace(-theta_l_pos, 0, n_pts)
        x_lo = alpha_0_pos + alpha_pos * np.cos(theta)
        z_lo = beta_pos * np.sin(theta)
    elif t_pos == 0.5:
        z_lo = np.linspace(-kappa_l, 0, n_pts)
        x_lo = -1 - z_lo**2 * (1 + delta_l) / kappa_l**2
    elif t_pos == 1:
        z_lo = np.linspace(-kappa_l, 0, n_pts)
        x_lo = 1 - z_lo * (1 + delta_l) / kappa_l
    elif t_pos > 0.5:
        phi_l_pos = np.arcsinh(np.sqrt(2 * t_pos - 1) / (1 - t_pos))
        alpha_0_pos, alpha_pos, beta_pos = calc_angles_pos_above(delta_l, kappa_l, t_pos)
        phi = np.linspace(-phi_l_pos, 0, n_pts)
        x_lo = alpha_0_pos + alpha_pos * np.cosh(phi)
        z_lo = beta_pos * np.sinh(phi)
    else:
        raise ValueError("Something is wrong with the Johner parameterisation.")

    x_quadrants = [x_ui, x_uo[::-1], x_lo[::-1], x_li]
    z_quadrants = [z_ui, z_uo[::-1], z_lo[::-1], z_li]
    x_quadrants = [a * xq + r_0 for xq in x_quadrants]
    z_quadrants = [a * zq for zq in z_quadrants]
    if negative:
        x_quadrants = [-(xq - 2 * r_0) for xq in x_quadrants]

    z_quadrants = [zq + z_0 for zq in z_quadrants]
    return x_quadrants, z_quadrants


def flux_surface_johner(
    r_0,
    z_0,
    a,
    kappa_u,
    kappa_l,
    delta_u,
    delta_l,
    psi_u_neg,
    psi_u_pos,
    psi_l_neg,
    psi_l_pos,
    n=100,
):
    """
    Initial plasma shape parametrerisation from HELIOS author
    J. Johner (CEA). Sets initial separatrix shape for the plasma core
    (does not handle divertor target points or legs).
    Can handle:
    - DN (positive, negative delta) [TESTED]
    - SN (positive, negative delta) (upper, lower) [TESTED]

    Parameters
    ----------
    r_0: float
        Major radius [m]
    z_0: float
        Vertical position of major radius [m]
    a: float
        Minor radius [m]
    kappa_u: float
        Upper elongation at the plasma edge (psi_n=1)
    kappa_l: float
        Lower elongation at the plasma edge (psi_n=1)
    delta_u: float
        Upper triangularity at the plasma edge (psi_n=1)
    delta_l: float
        Lower triangularity at the plasma edge (psi_n=1)
    psi_u_neg: float
        Upper inner angle [°]
    psi_u_pos: float
        Upper outer angle [°]
    psi_l_neg: float
        Lower inner angle [°]
    psi_l_pos: float
        Lower outer angle [°]
    n: int (defeault = 100)
        Number of point to generate on the flux surface

    Returns
    -------
    flux_surface: Coordinates
        Plasma flux surface shape
    """
    x_quadrants, z_quadrants = flux_surface_johner_quadrants(
        r_0,
        z_0,
        a,
        kappa_u,
        kappa_l,
        delta_u,
        delta_l,
        psi_u_neg,
        psi_u_pos,
        psi_l_neg,
        psi_l_pos,
        n=n,
    )

    return Coordinates(
        {"x": np.concatenate(x_quadrants), "z": np.concatenate(z_quadrants)}
    )


class JohnerLCFS(GeometryParameterisation):
    """
    Johner last closed flux surface geometry parameterisation.
    """

    __slots__ = ()

    def __init__(self, var_dict=None):
        variables = OptVariables(
            [
                BoundedVariable(
                    "r_0", 9, lower_bound=6, upper_bound=12, descr="Major radius"
                ),
                BoundedVariable(
                    "z_0",
                    0,
                    lower_bound=-1,
                    upper_bound=1,
                    descr="Vertical coordinate at geometry centroid",
                ),
                BoundedVariable(
                    "a", 6, lower_bound=1, upper_bound=6, descr="Minor radius"
                ),
                BoundedVariable(
                    "kappa_u",
                    1.6,
                    lower_bound=1.0,
                    upper_bound=3.0,
                    descr="Upper elongation",
                ),
                BoundedVariable(
                    "kappa_l",
                    1.8,
                    lower_bound=1.0,
                    upper_bound=3.0,
                    descr="Lower elongation",
                ),
                BoundedVariable(
                    "delta_u",
                    0.4,
                    lower_bound=0.0,
                    upper_bound=1.0,
                    descr="Upper triangularity",
                ),
                BoundedVariable(
                    "delta_l",
                    0.4,
                    lower_bound=0.0,
                    upper_bound=1.0,
                    descr="Lower triangularity",
                ),
                BoundedVariable(
                    "phi_u_neg",
                    180,
                    lower_bound=0,
                    upper_bound=190,
                    descr="Upper inner angle [°]",
                ),
                BoundedVariable(
                    "phi_u_pos",
                    10,
                    lower_bound=0,
                    upper_bound=20,
                    descr="Upper outer angle [°]",
                ),
                BoundedVariable(
                    "phi_l_neg",
                    -120,
                    lower_bound=-130,
                    upper_bound=45,
                    descr="Lower inner angle [°]",
                ),
                BoundedVariable(
                    "phi_l_pos",
                    30,
                    lower_bound=0,
                    upper_bound=45,
                    descr="Lower outer angle [°]",
                ),
            ],
            frozen=True,
        )
        variables.adjust_variables(var_dict, strict_bounds=False)
        super().__init__(variables)

    def create_shape(self, label="LCFS", n_points=1000):
        """
        Make a CAD representation of the Johner LCFS.

        Parameters
        ----------
        label: str, default = "LCFS"
            Label to give the wire
        n_points: int
            Number of points to use when creating the Bspline representation

        Returns
        -------
        shape: BluemiraWire
            CAD Wire of the geometry
        """
        x_quadrants, z_quadrants = flux_surface_johner_quadrants(
            *self.variables.values, n=n_points
        )

        wires = []
        labels = ["upper_inner", "upper_outer", "lower_outer", "lower_inner"]
        for x_q, z_q, lab in zip(x_quadrants, z_quadrants, labels):
            wires.append(
                interpolate_bspline(
                    np.array([x_q, np.zeros(len(x_q)), z_q]).T, label=lab
                )
            )

        return BluemiraWire(wires, label=label)
