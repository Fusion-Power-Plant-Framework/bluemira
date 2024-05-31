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
Paris Law fatigue model with FE-inspired analytical crack propagation
"""

import abc
from dataclasses import dataclass

import numpy as np

__all__ = [
    "ConductorInfo",
    "EllipticalEmbeddedCrack",
    "ParisFatigueMaterial",
    "ParisFatigueSafetyFactors",
    "QuarterEllipticalCornerCrack",
    "SemiEllipticalSurfaceCrack",
    "calculate_n_pulses",
]


@dataclass
class ConductorInfo:
    """
    Cable in conduit conductor information for Paris fatigue model
    """

    tk_radial: float  # [m] in the loaded direction
    width: float  # [m] in the loaded direction
    max_hoop_stress: float  # [Pa]
    residual_stress: float  # [Pa]
    walker_coeff: float


@dataclass
class ParisFatigueMaterial:
    """
    Material properties for the Paris fatigue model
    """

    C: float  # Paris law material constant
    m: float  # Paris law material exponent
    K_ic: float  # Fracture toughness  [Pa/m^(1/2)]


@dataclass
class ParisFatigueSafetyFactors:
    """
    Safety factors for the Paris fatigue model
    """

    sf_n_cycle: float
    sf_depth_crack: float
    sf_width_crack: float
    sf_fracture: float


def _stress_intensity_factor(
        hoop_stress: float,
        bend_stress: float,
        a: float,
        H: float,  # noqa: N803
        Q: float,  # noqa: N803
        F: float,
) -> float:
    """
    Equation 1a of Newman and Raju, 1984
    """
    return (hoop_stress + H * bend_stress) * np.sqrt(np.pi * a / Q) * F


def _boundary_correction_factor(
        a_d_t, m1: float, m2: float, m3: float, g: float, f_phi: float, f_w: float
) -> float:
    """
    Equation 1b of Newman and Raju, 1984
    """
    return (m1 + m2 * a_d_t ** 2 + m3 * a_d_t ** 4) * g * f_phi * f_w


def _bending_correction_factor(h1: float, h2: float, p: float, phi: float) -> float:
    """
    Equation 1c of Newman and Raju, 1984
    """
    return h1 + (h2 - h1) * np.sin(phi) ** p


def _ellipse_shape_factor(ratio: float) -> float:
    """
    Equation 2 of Newman and Raju, 1984
    """
    return 1.0 + 1.464 * ratio ** 1.65


def _angular_location_correction(a: float, c: float, phi: float) -> float:
    """
    Equation 10 and 13 of Newman and Raju, 1984
    """
    if a <= c:
        return ((a / c) ** 2 * np.cos(phi) ** 2 + np.sin(phi) ** 2) ** 0.25  # (10)
    return ((c / a) ** 2 * np.sin(phi) ** 2 + np.cos(phi) ** 2) ** 0.25  # (13)


def _finite_width_correction(a_d_t: float, c: float, w: float) -> float:
    """
    Equation 11 of Newman and Raju, 1984
    """
    return 1.0 / np.sqrt(np.cos(np.sqrt(a_d_t) * np.pi * c / (2 * w)))  # (11)


class Crack(abc.ABC):
    """
    Crack description ABC for the Paris fatigue model

    Parameters
    ----------
    depth:
        Crack depth in the plate thickness direction
    width:
        Crack width along the plate length direction
    """

    alpha = None

    def __init__(self, depth: float, width: float):
        self.depth = depth  # a
        self.width = width  # c

    @classmethod
    def from_area(cls, area: float, aspect_ratio: float):
        """
        Instatiate a crack from an area and aspect ratio
        """
        depth = np.sqrt(area / (cls.alpha * np.pi * aspect_ratio))
        width = aspect_ratio * depth
        return cls(depth, width)

    @property
    def area(self) -> float:
        """
        Cross-sectional area of the crack
        """
        return self.alpha * np.pi * self.depth * self.width

    @abc.abstractmethod
    def stress_intensity_factor(
            self,
            hoop_stress: float,
            bend_stress: float,
            t: float,
            w: float,
            a: float,
            c: float,
            phi: float,
    ) -> float:
        """
        Calculate the crack stress intensity factor
        """
        pass


class QuarterEllipticalCornerCrack(Crack):
    """
    Quarter-elliptical corner crack

    Parameters
    ----------
    depth:
        Crack depth in the plate thickness direction
    width:
        Crack width along the plate length direction
    """

    alpha = 0.25

    def stress_intensity_factor(
            self,
            hoop_stress: float,
            bend_stress: float,
            t: float,
            w: float,
            a: float,
            c: float,
            phi: float,
    ) -> float:
        """
        Calculate quarter-elliptical corner crack stress intensity factor.

        Parameters
        ----------
        hoop_stress:
            Hoop stress in the plate [Pa]
        bend_stress:
            Bending stress in the plate [Pa]
        t:
            Plate thickness [m]
        w:
            Plate width [m]
        a:
            Crack depth [m]
        c:
            Crack width [m]
        phi:
            Crack angle [rad]

        Returns
        -------
        Stress intensity factor

        Notes
        -----
        Newman and Raju, 1984, Stress-intensity factor equations for cracks in
        three-dimensional finite bodies subjected to tension and bending loads
        https://ntrs.nasa.gov/api/citations/19840015857/downloads/19840015857.pdf
        """
        a_d_t = a / t

        if a <= c:  # a/c <= 1
            ratio = a / c
            m1 = 1.08 - 0.03 * ratio  # (39)
            m2 = -0.44 + 1.06 / (0.3 + ratio)  # (40)
            m3 = -0.5 + 0.25 * ratio + 14.8 * (1 - ratio) ** 15  # (41)
            g1 = 1 + (0.08 + 0.4 * a_d_t ** 2) * (1 - np.sin(phi)) ** 3  # (42)
            g2 = 1 + (0.08 + 0.15 * a_d_t ** 2) * (1 - np.cos(phi)) ** 3  # (43)

            g21 = -1.22 - 0.12 * ratio  # (24)
            g22 = 0.64 - 1.05 * ratio ** 0.75 + 0.47 * ratio ** 1.5  # (46)
            h1 = 1.0 - 0.34 * a_d_t - 0.11 * ratio * a_d_t  # (22)
            h2 = 1.0 + g21 * a_d_t + g22 * a_d_t ** 2  # (23)
        else:  # a/c > 1
            ratio = c / a
            m1 = np.sqrt(ratio) * (1.08 - 0.03 * ratio)  # (47)
            m2 = 0.375 * ratio ** 2  # (48)
            m3 = -0.25 * ratio ** 2  # (49)
            g1 = 1 + (0.08 + 0.4 * (c / t) ** 2) * (1 - np.sin(phi)) ** 3  # (50)
            g2 = 1 + (0.08 + 0.15 * (c / t) ** 2) * (1 - np.cos(phi)) ** 3  # (51)

            g11 = -0.04 - 0.41 * ratio  # (33)
            g12 = 0.55 - 1.93 * ratio ** 0.75 + 1.38 * ratio ** 1.5  # (34)
            g21 = -2.11 + 0.77 * ratio  # (35)
            g22 = 0.64 - 0.72 * ratio ** 0.75 + 0.14 * ratio ** 1.5  # (52)
            h1 = 1.0 + g11 * a_d_t + g12 * a_d_t ** 2  # (31)
            h2 = 1.0 + g21 * a_d_t + g22 * a_d_t ** 2  # (32)

        llambda = c / w * np.sqrt(a_d_t)
        f_w = (  # (44)
                1.0
                - 0.2 * llambda
                + 9.4 * llambda ** 2
                - 19.4 * llambda ** 3
                + 27.1 * llambda ** 4
        )
        f_phi = _angular_location_correction(a, c, phi)

        p = 0.2 + ratio + 0.6 * a_d_t  # (21 & 30)
        H = _bending_correction_factor(h1, h2, p, phi)  # noqa: N806
        Q = _ellipse_shape_factor(ratio)  # noqa: N806
        F = _boundary_correction_factor(a_d_t, m1, m2, m3, g1 * g2, f_phi, f_w)
        return _stress_intensity_factor(hoop_stress, bend_stress, a, H, Q, F)


class SemiEllipticalSurfaceCrack(Crack):
    """
    Semi-elliptical surface crack

    Parameters
    ----------
    depth:
        Crack depth in the plate thickness direction
    width:
        Crack width along the plate length direction
    """

    alpha = 0.5

    def stress_intensity_factor(
            self,
            hoop_stress: float,
            bend_stress: float,
            t: float,
            w: float,
            a: float,
            c: float,
            phi: float,
    ) -> float:
        """
        Calculate semi-elliptical surface crack stress intensity factor.

        Parameters
        ----------
        hoop_stress:
            Hoop stress in the plate [Pa]
        bend_stress:
            Bending stress in the plate [Pa]
        t:
            Plate thickness [m]
        w:
            Plate width [m]
        a:
            Crack depth [m]
        c:
            Crack width [m]
        phi:
            Crack angle [rad]

        Returns
        -------
        Stress intensity factor

        Notes
        -----
        Newman and Raju, 1984, Stress-intensity factor equations for cracks in
        three-dimensional finite bodies subjected to tension and bending loads
        https://ntrs.nasa.gov/api/citations/19840015857/downloads/19840015857.pdf
        """
        a_d_t = a / t

        if a <= c:  # a/c <= 1
            ratio = a / c
            m1 = 1.13 - 0.09 * ratio  # (16)
            m2 = -0.54 + 0.89 / (0.2 + ratio)  # (17)
            m3 = 0.5 - 1.0 / (0.65 + ratio) + 14.0 * (1 - ratio) ** 24  # (18)
            g = 1.0 + (0.1 + 0.35 * a_d_t ** 2) * (1 - np.sin(phi)) ** 2  # (19)

            g21 = -1.22 - 0.12 * ratio  # (24)
            g22 = 0.55 - 1.05 * ratio ** 0.75 + 0.47 * ratio ** 1.5  # (25)
            h1 = 1.0 - 0.34 * a_d_t - 0.11 * ratio * a_d_t  # (22)
            h2 = 1.0 + g21 * a_d_t + g22 * a_d_t ** 2  # (23)

        else:  # a/c > 1
            ratio = c / a
            m1 = np.sqrt(ratio) * (1.0 + 0.04 * ratio)  # (26)
            m2 = 0.2 * ratio ** 4  # (27)
            m3 = -0.11 * ratio ** 4  # (28)
            g = 1.0 + (0.1 + 0.35 * ratio * a_d_t ** 2) * (1 - np.sin(phi)) ** 2  # (29)

            g11 = -0.04 - 0.41 * ratio  # (33)
            g12 = 0.55 - 1.93 * ratio ** 0.75 + 1.38 * ratio ** 1.5  # (34)
            g21 = -2.11 + 0.77 * ratio  # (35)
            g22 = 0.55 - 0.72 * ratio ** 0.75 + 0.14 * ratio ** 1.5  # (36)
            h1 = 1.0 + g11 * a_d_t + g12 * a_d_t ** 2  # (31)
            h2 = 1.0 + g21 * a_d_t + g22 * a_d_t ** 2  # (32)

        f_phi = _angular_location_correction(a, c, phi)
        f_w = _finite_width_correction(a_d_t, c, w)
        p = 0.2 + ratio + 0.6 * a_d_t  # (21 & 30)
        H = _bending_correction_factor(h1, h2, p, phi)  # noqa: N806
        Q = _ellipse_shape_factor(ratio)  # noqa: N806
        F = _boundary_correction_factor(a_d_t, m1, m2, m3, g, f_phi, f_w)
        return _stress_intensity_factor(hoop_stress, bend_stress, a, H, Q, F)


class EllipticalEmbeddedCrack(Crack):
    """
    Full elliptical embedded crack

    Parameters
    ----------
    depth:
        Crack depth in the plate thickness direction
    width:
        Crack width along the plate length direction
    """

    alpha = 1.0

    def stress_intensity_factor(
            self,
            hoop_stress: float,
            bend_stress: float,
            t: float,
            w: float,
            a: float,
            c: float,
            phi: float,
    ) -> float:
        """
        Calculate elliptical embedded crack stress intensity factor.

        Parameters
        ----------
        hoop_stress:
            Hoop stress in the plate [Pa]
        bend_stress:
            Bending stress in the plate [Pa]
        t:
            Plate thickness [m]
        w:
            Plate width [m]
        a:
            Crack depth [m]
        c:
            Crack width [m]
        phi:
            Crack angle [rad]

        Returns
        -------
        Stress intensity factor

        Notes
        -----
        Newman and Raju, 1984, Stress-intensity factor equations for cracks in
        three-dimensional finite bodies subjected to tension and bending loads
        https://ntrs.nasa.gov/api/citations/19840015857/downloads/19840015857.pdf
        """
        # NOTE: for embedded cracks, t is defined as one-half the plate thickness
        t = 0.5 * t
        a_d_t = a / t

        if a <= c:  # a/c <= 1
            ratio = a / c
            m1 = 1.0  # (6)

        else:  # a/c > 1
            ratio = c / a
            m1 = np.sqrt(ratio)  # (12)

        m2 = 0.05 / (0.11 + (a / c) ** 1.5)  # (7)
        m3 = 0.29 / (0.23 + (a / c) ** 1.5)  # (8)
        g = 1.0 - (a_d_t ** 4 * np.sqrt(2.6 - 2 * a_d_t)) / (1 + 4 * a / c) * np.abs(
            np.cos(phi)
        )  # (9)
        f_phi = _angular_location_correction(a, c, phi)
        f_w = _finite_width_correction(a_d_t, c, w)

        Q = _ellipse_shape_factor(ratio)  # noqa: N806
        F = _boundary_correction_factor(a_d_t, m1, m2, m3, g, f_phi, f_w)
        return _stress_intensity_factor(hoop_stress, bend_stress, a, 0.0, Q, F)


def calculate_n_pulses(
        conductor: ConductorInfo,
        crack: Crack,
        material: ParisFatigueMaterial,
        safety: ParisFatigueSafetyFactors,
) -> int:
    """
    Calculate the number of plasma pulses possible prior to fatigue.

    Parameters
    ----------
    conductor:
        Conductor information
    crack:
        Postulated initiating crack size
    material:
        Material values
    safety:
        Safety factors

    Returns
    -------
    Number of plasma pulses

    Notes
    -----
    Assumes two stress cycles per pulse.
    Calculates using the lifecycle method.
    """
    mean_stress_ratio = conductor.residual_stress / (
            conductor.max_hoop_stress + conductor.residual_stress
    )

    C_r = material.C * (1 - mean_stress_ratio) ** (  # noqa: N806
            material.m * (conductor.walker_coeff - 1)
    )

    max_crack_depth = conductor.tk_radial / safety.sf_depth_crack
    max_crack_width = conductor.width / safety.sf_width_crack
    max_stress_intensity = material.K_ic / safety.sf_fracture

    a = crack.depth
    c = crack.width
    K_max = 0.0  # noqa: N806
    n_cycles = 0

    delta = 1e-4  # Crack size increment

    while a < max_crack_depth and c < max_crack_width and K_max < max_stress_intensity:
        Ka = crack.stress_intensity_factor(  # noqa: N806
            conductor.max_hoop_stress,
            0.0,
            conductor.tk_radial,
            conductor.width,
            a,
            c,
            0.5 * np.pi,
        )
        Kc = crack.stress_intensity_factor(  # noqa: N806
            conductor.max_hoop_stress,
            0.0,
            conductor.tk_radial,
            conductor.width,
            a,
            c,
            0.0,
        )
        K_max = max(Ka, Kc)  # noqa: N806

        a += delta / (Ka / K_max) ** material.m
        c += delta / (Kc / K_max) ** material.m
        n_cycles += delta / (C_r * K_max ** material.m)

    n_cycles /= safety.sf_n_cycle

    return n_cycles // 2