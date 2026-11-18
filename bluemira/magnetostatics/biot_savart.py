# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Biot-Savart filament object
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numba as nb
import numpy as np

from bluemira.base.constants import MU_0, MU_0_4PI
from bluemira.geometry.bound_box import BoundingBox
from bluemira.geometry.coordinates import rotation_matrix
from bluemira.magnetostatics.baseclass import CurrentSource
from bluemira.magnetostatics.error import MagnetostaticsError
from bluemira.magnetostatics.tools import process_coords_array
from bluemira.utilities.plot_tools import Plot3D

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy.typing as npt
    from matplotlib.pyplot import Axes

    from bluemira.geometry.coordinates import Coordinates

__all__ = ["BiotSavartFilament"]


@nb.njit(cache=True, inline="always")
def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


@nb.njit(cache=True, inline="always")
def _norm2(a) -> float:
    return _dot(a, a)


@nb.njit(cache=True, inline="always")
def _cross(a, b) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


@nb.njit(cache=True, inline="always")
def _sub(a, b) -> tuple[float, float, float]:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


@nb.njit(cache=True, inline="always")
def _along(d, s: float, ea, t: float, eb) -> tuple[float, float, float]:
    """D + s ea - t eb"""  # noqa: DOC201
    return (
        d[0] + s * ea[0] - t * eb[0],
        d[1] + s * ea[1] - t * eb[1],
        d[2] + s * ea[2] - t * eb[2],
    )


@nb.njit(cache=True, inline="always")
def _log_r_plus(x: float, rest: float) -> float:
    """
    ln(R + x) with R^2 = x^2 + rest, without cancellation when x is close to -R.

    R is recomputed here rather than shared, so that it is consistent with x when
    the two segments are nearly parallel.

    Returns
    -------
    :
        ln(R + x)
    """
    r = np.sqrt(x**2 + rest)
    return np.log(r + x) if x >= 0 else np.log(rest / (r - x))


@nb.njit(cache=True, inline="always")
def _lengths_and_directions(
    d_l: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    lengths = np.sqrt((d_l**2).sum(axis=1))
    return lengths, d_l / lengths.reshape(-1, 1)


@nb.njit(cache=True, inline="always")
def _segment_geometry(point, start, length: float, e_hat, a2: float):
    """
    Quantities shared by the field and potential of one finite-radius segment.

    Returns
    -------
    cross:
        e_hat x (point - start), of magnitude rho
    rho2:
        Squared distance from the point to the segment's axis
    rho2_c:
        rho2, clipped to the conductor radius squared
    r_sum:
        R1 + R2, the distances to the segment ends (with rho clipped)
    r1r2:
        R1 * R2
    denom:
        rho_c^2 + R1 R2 + t1 t2, equal to ((R1 + R2)^2 - L^2) / 2

    Notes
    -----
    In the notation of Hanson & Hirshman (Phys. Plasmas 9, 4410, 2002),
    R1 = R_i, R2 = R_f, t1 = R_i|| and t2 = R_f||.
    """
    r1 = _sub(point, start)
    t1 = _dot(r1, e_hat)
    t2 = t1 - length
    cross = _cross(e_hat, r1)
    rho2 = _dot(cross, cross)

    # Not in the paper: inside the conductor rho is clipped to the radius, which
    # removes the singularity on the segment (see the module docstring)
    rho2_c = max(rho2, a2)
    big_r1 = np.sqrt(rho2_c + t1**2)
    big_r2 = np.sqrt(rho2_c + t2**2)
    r1r2 = big_r1 * big_r2
    # The paper's denominator (R1 + R2)^2 - L^2 cancels catastrophically next to
    # the segment, where R1 + R2 ~ L. Expanding it with L = t1 - t2 gives
    # 2 (rho^2 + q), where q = R1 R2 + t1 t2 is computed without cancellation when
    # t1 t2 < 0 (i.e. beside the segment)
    t1t2 = t1 * t2
    if t1t2 >= 0:
        q = r1r2 + t1t2
    else:
        q = rho2_c * (t1**2 + t2**2 + rho2_c) / (r1r2 - t1t2)
    return cross, rho2, rho2_c, big_r1 + big_r2, r1r2, rho2_c + q


@nb.njit(cache=True)
def _field_kernel(
    points: npt.NDArray[np.float64],
    starts: npt.NDArray[np.float64],
    d_l: npt.NDArray[np.float64],
    radius: float,
) -> npt.NDArray[np.float64]:
    """
    Field per unit current of finite-radius straight segments.

    Parameters
    ----------
    points:
        Field points, shape (n, 3)
    starts:
        Segment start points, shape (m, 3)
    d_l:
        Segment vectors, shape (m, 3)
    radius:
        Conductor radius [m]

    Returns
    -------
    :
        Field / (mu_0 I / 4 pi), shape (n, 3)

    Notes
    -----
    Eq. (8) of Hanson & Hirshman (Phys. Plasmas 9, 4410, 2002):

    :math:`\\dfrac{4\\pi}{\\mu_0 I}\\mathbf{B} = \\hat{e} \\times \\mathbf{R}_i \\dfrac{2L(R_i + R_f)}{R_i R_f} \\dfrac{1}{(R_i + R_f)^2 - L^2}`

    rearranged to avoid cancellation near the conductor, and with the distance
    to the axis clipped to the conductor radius.
    """  # noqa: W505, E501
    a2 = radius**2
    lengths, e_hats = _lengths_and_directions(d_l)
    field = np.zeros((len(points), 3))
    for i in range(len(points)):
        for j in range(len(starts)):
            cross, _, _, r_sum, r1r2, denom = _segment_geometry(
                points[i], starts[j], lengths[j], e_hats[j], a2
            )
            # Eq. (8), with (R1 + R2)^2 - L^2 = 2 denom
            s_field = lengths[j] * r_sum / (r1r2 * denom)
            for k in range(3):
                field[i, k] += s_field * cross[k]
    return field


@nb.njit(cache=True)
def _potential_kernel(
    points: npt.NDArray[np.float64],
    starts: npt.NDArray[np.float64],
    d_l: npt.NDArray[np.float64],
    radius: float,
) -> npt.NDArray[np.float64]:
    """
    Vector potential per unit current of finite-radius straight segments.

    Parameters
    ----------
    points:
        Field points, shape (n, 3)
    starts:
        Segment start points, shape (m, 3)
    d_l:
        Segment vectors, shape (m, 3)
    radius:
        Conductor radius [m]

    Returns
    -------
    :
        Potential / (mu_0 I / 4 pi), shape (n, 3)

    Notes
    -----
    Eq. (7) of Hanson & Hirshman (Phys. Plasmas 9, 4410, 2002):

    :math:`\\dfrac{4\\pi}{\\mu_0 I}\\mathbf{A} = \\hat{e} \\ln\\dfrac{R_i + R_f + L}{R_i + R_f - L}`

    rearranged to avoid cancellation both near the conductor and in the far
    field, and with a uniform current density core term inside the conductor.
    """  # noqa: W505, E501
    a2 = radius**2
    lengths, e_hats = _lengths_and_directions(d_l)
    potential = np.zeros((len(points), 3))
    for i in range(len(points)):
        for j in range(len(starts)):
            length = lengths[j]
            _, rho2, rho2_c, r_sum, r1r2, denom = _segment_geometry(
                points[i], starts[j], length, e_hats[j], a2
            )
            # Eq. (7), written as ln(1 + 2L / (R1 + R2 - L)) and multiplied through
            # by (R1 + R2 + L) to use denom. log1p keeps precision in the far field,
            # where the paper's ratio tends to 1 (cf. its power series, Eq. (10))
            s_pot = np.log1p(length * (r_sum + length) / denom)
            if rho2 < a2:
                # Not in the paper: uniform current density core term, giving
                # A(rho) = A(a) + mu_0 I / 4 pi (1 - rho^2 / a^2) for a long segment
                s_field = length * r_sum / (r1r2 * denom)
                s_pot += 0.5 * (a2 - rho2) / a2 * rho2_c * s_field
            for k in range(3):
                potential[i, k] += s_pot * e_hats[j, k]
    return potential


@nb.njit(cache=True)
def _neumann(
    starts: npt.NDArray[np.float64], d_l: npt.NDArray[np.float64], n_ref: int, gmd: float
) -> float:
    """
    Sum of the Neumann double integrals between the first n_ref (reference)
    segments and all the segments.

    Parameters
    ----------
    starts:
        Segment start points, shape (m, 3)
    d_l:
        Segment vectors, shape (m, 3)
    n_ref:
        Number of reference segments, at the start of the arrays
    gmd:
        Geometric mean distance of the conductor cross-section [m]

    Returns
    -------
    :
        :math:`\\sum_{i < n_{ref}}\\sum_{j}\\int_i\\int_j d\\mathbf{l}_i \\cdot d\\mathbf{l}_j / \\sqrt{|\\mathbf{r}|^2 + gmd^2}`

    Notes
    -----
    The integral is symmetric in i and j, so each pair of reference segments is
    evaluated once and counted twice.
    """  # noqa: W505, E501
    lengths, e_hats = _lengths_and_directions(d_l)

    def pair(i, j):
        return _neumann_pair(
            starts[i], e_hats[i], lengths[i], starts[j], e_hats[j], lengths[j], gmd
        )

    total = 0.0
    for i in range(n_ref):
        total += pair(i, i)
        for j in range(i + 1, n_ref):
            total += 2 * pair(i, j)
        for j in range(n_ref, len(starts)):
            total += pair(i, j)
    return total


@nb.njit(cache=True)
def _neumann_pair(a0, ea, la: float, b0, eb, lb: float, gmd: float) -> float:
    """
    Neumann double integral between two straight segments.

    Returns
    -------
    :
        :math:`\\int_a\\int_b d\\mathbf{l}_a \\cdot d\\mathbf{l}_b / \\sqrt{|\\mathbf{r}|^2 + gmd^2}`

    Notes
    -----
    Segment a is :math:`a_0 + s\\hat{a}`, :math:`s \\in [0, l_a]`, and segment b is
    :math:`b_0 + t\\hat{b}`, :math:`t \\in [0, l_b]`, at an angle
    :math:`\\varepsilon` to each other. With p and q measured along each line from
    the feet of their common perpendicular (length h), the integral is
    :math:`\\cos\\varepsilon` times the double antiderivative of 1/R

    :math:`p \\ln(R + q - p\\cos\\varepsilon) + q \\ln(R + p - q\\cos\\varepsilon) - \\dfrac{h}{\\sin\\varepsilon}\\tan^{-1}\\dfrac{h^2 \\cos\\varepsilon + pq \\sin^2\\varepsilon}{h R \\sin\\varepsilon}`

    evaluated between the segment ends. This is the indefinite form in the proof
    of formula (1) of Campbell (1915), with p, q, h his S, s, Pp; the arctangent
    is the indefinite form of his solid angle term, formula (10).

    For parallel segments the feet are undefined and the solid angle term is
    0/0. As Campbell notes, it then reduces to R, and the integral is evaluated
    as :math:`\\cos\\varepsilon` times
    :math:`-\\cos\\varepsilon\\,(z \\sinh^{-1}(z/h) - R)`, with z the axial
    separation, which is equivalent to his formula (3).

    References
    ----------
    G. A. Campbell, "Mutual inductances of circuits composed of straight wires",
    Phys. Rev. 5, 452-458 (1915), https://doi.org/10.1103/PhysRev.5.452

    F. W. Grover, "Inductance Calculations: Working Formulas and Tables",
    Van Nostrand (1946); Dover reprint (2004), which reproduces Campbell's result
    """  # noqa: W505, E501
    d = _sub(a0, b0)  # r(s, t) = d + s ea - t eb
    d_a, d_b = _dot(d, ea), _dot(d, eb)
    cos = _dot(ea, eb)
    sin2 = _norm2(_cross(ea, eb))  # Not 1 - cos^2: precise when small
    parallel = sin2 < 1e-12  # noqa: PLR2004

    # Feet of the common perpendicular, along each segment (undefined if parallel)
    s0 = 0.0 if parallel else (cos * d_b - d_a) / sin2
    t0 = 0.0 if parallel else (d_b - cos * d_a) / sin2
    # Squared distance between the lines. Computed from the feet (not d . n / sin)
    # so that its rounding is consistent with p and q
    h_vec = _cross(d, ea) if parallel else _along(d, s0, ea, t0, eb)
    h2 = _norm2(h_vec) + gmd**2

    total = 0.0
    for s, t, sign in ((la, lb, 1.0), (la, 0.0, -1.0), (0.0, lb, -1.0), (0.0, 0.0, 1.0)):
        if parallel:
            total += sign * _parallel_antiderivative(d_a + s - cos * t, cos, h2)
        else:
            total += sign * _skew_antiderivative(s - s0, t - t0, cos, sin2, h2)
    return cos * total


@nb.njit(cache=True, inline="always")
def _skew_antiderivative(p: float, q: float, cos: float, sin2: float, h2: float):
    """
    Double antiderivative of 1/R, with p and q measured from the feet of the
    common perpendicular.

    Returns
    -------
    :
        The antiderivative, see :func:`_neumann_pair`
    """
    h, sin = np.sqrt(h2), np.sqrt(sin2)
    r = np.sqrt(h2 + p**2 + q**2 - 2 * cos * p * q)
    return (
        p * _log_r_plus(q - cos * p, h2 + sin2 * p**2)
        + q * _log_r_plus(p - cos * q, h2 + sin2 * q**2)
        - h / sin * np.arctan((cos * h2 + sin2 * p * q) / (h * sin * r))
    )


@nb.njit(cache=True, inline="always")
def _parallel_antiderivative(z: float, cos: float, h2: float) -> float:
    """
    Double antiderivative of 1/R for parallel segments, where z is the axial
    separation.

    Returns
    -------
    :
        The antiderivative, see :func:`_neumann_pair`
    """
    h = np.sqrt(h2)
    return -cos * (z * np.arcsinh(z / h) - np.hypot(h, z))


class BiotSavartFilament(CurrentSource):
    """
    Class to calculate field and vector potential from an arbitrary filament.

    Parameters
    ----------
    arrays:
        The arbitrarily shaped closed current Coordinates. Alternatively provide the
        list of Coordinates objects.
    radius:
        The nominal radius of the coil [m].
    current:
        The current flowing through the filament [A]. Defaults to 1 A to enable
        current to be optimised separately from the field response.
    """

    def __init__(
        self,
        arrays: Coordinates
        | npt.NDArray[np.float64]
        | list[Coordinates]
        | list[npt.NDArray[np.float64]],
        radius: float,
        current: float = 1.0,
    ):
        if not radius > 0:
            raise MagnetostaticsError(f"Radius must be positive, not {radius}.")
        array_list = arrays if isinstance(arrays, list) else [arrays]
        self._arrays = [
            np.asarray(process_coords_array(a), dtype=float) for a in array_list
        ]

        starts, d_ls = [], []
        for xyz in self._arrays:
            d_l = np.diff(xyz, axis=0)
            keep = np.any(d_l != 0, axis=1)  # Drop repeated points
            if not np.any(keep):
                raise MagnetostaticsError("Each filament needs two distinct points.")
            starts.append(xyz[:-1][keep])
            d_ls.append(d_l[keep])

        self._n_ref = len(d_ls[0])  # The first Coordinates is the reference
        self._points = np.vstack(starts)
        self._d_l = np.vstack(d_ls)
        self._radius = radius
        self.current = current
        ref_lengths = np.linalg.norm(self.ref_d_l, axis=1)
        self.length = float(np.sum(ref_lengths))
        self.length_scale = float(np.min(ref_lengths))

    @property
    def ref_d_l(self) -> npt.NDArray[np.float64]:
        """Segment vectors of the first (reference) Coordinates."""
        return self._d_l[: self._n_ref]

    @property
    def ref_mid_points(self) -> npt.NDArray[np.float64]:
        """Segment mid-points of the first (reference) Coordinates."""
        return self._points[: self._n_ref] + 0.5 * self.ref_d_l

    @staticmethod
    def _process_points(x, y, z) -> tuple[npt.NDArray[np.float64], tuple[int, ...]]:
        """
        Returns
        -------
        points:
            The coordinates as (n, 3) points
        shape:
            The shape of the input coordinates, () for a single point

        Raises
        ------
        MagnetostaticsError
            If the coordinates are not of equal shape, or have more than 2 dimensions
        """
        x, y, z = (np.asarray(c, dtype=float) for c in (x, y, z))
        if not x.shape == y.shape == z.shape or x.ndim > 2:  # noqa: PLR2004
            raise MagnetostaticsError(
                "Coordinates must be floats or 1-D or 2-D arrays of equal shape."
            )
        shape = () if x.shape == (1,) else x.shape
        return np.stack([x, y, z], axis=-1).reshape(-1, 3), shape

    def _evaluate(
        self,
        kernel: Callable[
            [
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                npt.NDArray[np.float64],
                float,
            ],
            npt.NDArray[np.float64],
        ],
        x: float | npt.NDArray[np.float64],
        y: float | npt.NDArray[np.float64],
        z: float | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        points, shape = self._process_points(x, y, z)
        values = kernel(points, self._points, self._d_l, self._radius)
        return MU_0_4PI * self.current * values.T.reshape(3, *shape)

    def potential(
        self,
        x: float | npt.NDArray[np.float64],
        y: float | npt.NDArray[np.float64],
        z: float | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Calculate the vector potential of an arbitrarily shaped Coordinates.

        Parameters
        ----------
        x:
            The x coordinate(s) of the points at which to calculate the potential
        y:
            The y coordinate(s) of the points at which to calculate the potential
        z:
            The z coordinate(s) of the points at which to calculate the potential

        Returns
        -------
        :
            The vector potential {Ax, Ay, Az} at the point(s) [V.s/m]

        Notes
        -----
        Exact for each straight segment (outside the conductor):

        \t:math:`\\dfrac{\\mu_{0} I}{4\\pi}\\hat{e}\\left[\\sinh^{-1}\\dfrac{t_1}{\\rho}-\\sinh^{-1}\\dfrac{t_2}{\\rho}\\right]`
        """  # noqa: W505, E501
        return self._evaluate(_potential_kernel, x, y, z)

    def field(
        self,
        x: float | npt.NDArray[np.float64],
        y: float | npt.NDArray[np.float64],
        z: float | npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """
        Calculate the field due to the arbitrarily shaped Coordinates.

        Parameters
        ----------
        x:
            The x coordinate(s) of the points at which to calculate the field
        y:
            The y coordinate(s) of the points at which to calculate the field
        z:
            The z coordinate(s) of the points at which to calculate the field

        Returns
        -------
        :
            The field {Bx, By, Bz} at the point(s) [T]

        Notes
        -----
        Exact for each straight segment (outside the conductor):

        \t:math:`\\dfrac{\\mu_{0} I}{4\\pi \\rho^2}\\hat{e}\\times\\mathbf{r}\\left[\\dfrac{t_1}{\\sqrt{\\rho^2+t_1^2}}-\\dfrac{t_2}{\\sqrt{\\rho^2+t_2^2}}\\right]`

        Inside the conductor, :math:`\\rho` is replaced by the conductor radius in
        the scalar factors, recovering a uniform current density wire.
        """  # noqa: W505, E501
        return self._evaluate(_field_kernel, x, y, z)

    def inductance(self) -> float:
        """
        Calculate the total inductance of the BiotSavartFilament.

        Returns
        -------
        :
            The total inductance (including self-inductance of the reference
            Coordinates) [H]

        Notes
        -----
        \t:math:`L = \\dfrac{\\mu_0}{4\\pi}\\sum_{i \\in ref}\\sum_{j}\\int_i\\int_j \\dfrac{d\\mathbf{l}_i \\cdot d\\mathbf{l}_j}{|\\mathbf{r}_{ij}|}`

        evaluated in closed form for every pair of straight segments, between the
        first (reference) Coordinates and all the Coordinates. The finite radius
        is accounted for with the geometric mean distance of a round conductor,
        :math:`g = a e^{-1/4}`, by replacing :math:`|\\mathbf{r}_{ij}|^2` with
        :math:`|\\mathbf{r}_{ij}|^2 + g^2`. This gives the round wire
        self-inductance :math:`\\frac{\\mu_0 L}{2\\pi}(\\ln(2L/a) - 3/4)` for a
        long straight segment, and differs from the thin filament result by
        :math:`O(a^2/d^2)` for conductors a distance d apart. As each pair is
        exact, the result does not depend on how straight runs of conductor are
        discretised.

        The closed form for each pair of segments follows Campbell (1915) and
        Grover (1946); the geometric mean distance treatment follows Rosa (1908).

        References
        ----------
        G. A. Campbell, "Mutual inductances of circuits composed of straight
        wires", Phys. Rev. 5, 452-458 (1915), https://doi.org/10.1103/PhysRev.5.452

        F. W. Grover, "Inductance Calculations: Working Formulas and Tables",
        Van Nostrand (1946); Dover reprint (2004)

        E. B. Rosa, "The self and mutual inductances of linear conductors",
        Bull. Bur. Stand. 4, 301-344 (1908)
        """  # noqa: W505, E501
        gmd = self._radius * np.exp(-0.25)
        return MU_0_4PI * _neumann(self._points, self._d_l, self._n_ref, gmd)

    def rotate(self, angle: float, axis: str | np.ndarray):
        """
        Rotate the CurrentSource about an axis.

        Parameters
        ----------
        angle:
            The rotation degree [degree]
        axis:
            The axis of rotation
        """
        r = rotation_matrix(np.deg2rad(angle), axis).T
        self._points @= r
        self._d_l @= r
        self._arrays = [array @ r for array in self._arrays]

    def plot(
        self,
        ax: Axes | None = None,
        *,
        show_coord_sys: bool = False,
        **kwargs,  # noqa: ARG002
    ):
        """
        Plot the CurrentSource.

        Parameters
        ----------
        ax:
            The matplotlib axes to plot on
        show_coord_sys:
            Whether or not to plot the coordinate systems
        """
        if ax is None:
            ax = Plot3D()
            # If no ax provided, we assume that we want to plot only this source,
            # and thus set aspect ratio equality on this term only
            # Invisible bounding box to set equal aspect ratio plot
            xbox, ybox, zbox = BoundingBox.from_xyz(*self._points.T).get_box_arrays()
            ax.plot(1.1 * xbox, 1.1 * ybox, 1.1 * zbox, "s", alpha=0)

        for array in self._arrays:
            ax.plot(*array.T, color="b", linewidth=1)

        # Plot local coordinate system
        if show_coord_sys:
            origin = [0, 0, 0]
            dcm = np.eye(3)
            ax.scatter([0], [0], [0], color="k")
            ax.quiver(*origin, *dcm[0], length=self.length_scale, color="r")
            ax.quiver(*origin, *dcm[1], length=self.length_scale, color="r")
            ax.quiver(*origin, *dcm[2], length=self.length_scale, color="r")


def Bz_coil_axis(
    r: float, z: float = 0, pz: npt.ArrayLike = 0, current: float = 1
) -> float | npt.NDArray[np.float64]:
    """
    Calculate the theoretical vertical magnetic field of a filament coil
    (of radius r and centred in (0, z)) on a point on the coil axis at
    a distance pz from the axis origin.

    Parameters
    ----------
    r:
        Coil radius [m]
    z:
        Vertical position of the coil centroid [m]
    pz:
        Vertical position of the point on the axis on which the magnetic field
        shall be calculated [m]
    current:
        Current of the coil [A]

    Returns
    -------
    :
        Vertical magnetic field on the axis [T]

    Notes
    -----
    \t:math:`\\dfrac{1}{2}\\dfrac{\\mu_{0}Ir^2}{(r^{2}+(pz-z)^{2})^{3/2}}`
    """
    return 0.5 * MU_0 * current * r**2 / (r**2 + (np.asarray(pz) - z) ** 2) ** 1.5
