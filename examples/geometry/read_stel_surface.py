# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
For some reason I can't get bluemira to import at the same time as simsopt in IPython.
I'll just use this script to scrape the coefficienct data from simsopt/tess/testfiles
instead.
"""

from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy import typing as npt
from simsopt.geo import SurfaceRZFourier


def stellarator_curve_function_factory(surface: SurfaceRZFourier, phi: float):
    """
    Parameters
    ----------
    phi:
        toroidal angle in radian

    Returns
    -------
    function to calculate the poloidal slice
    """
    ntor = surface.ntor
    nfp = surface.nfp
    m = np.arange(0, surface.mpol + 1)
    n = np.arange(-ntor, ntor + 1)

    def slice_at_poloidal_angle(theta):
        """Gives the r and z coordinates of the curve at angle (phi, theta)."""
        # treat non-iterables fairly
        # if isinstance(theta, Iterable):
        #     return np.array(slice_at_poloidal_angle(theta_i) for theta_i in theta)
        original_shape = np.shape(theta)
        theta = np.atleast_1d(theta)
        len_theta = len(theta)

        nfp_n_phi = np.broadcast_to(nfp * n * phi, [len_theta, len(m), len(n)])
        m_theta = np.broadcast_to(np.outer(theta, m).T, [len(n), len(m), len_theta]).T
        in_paranth = m_theta - nfp_n_phi

        r = surface.rc * np.cos(in_paranth) + surface.rs * np.sin(in_paranth)
        z = surface.zc * np.cos(in_paranth) + surface.zs * np.sin(in_paranth)
        r = r.sum(axis=-1).sum(axis=-1)  # TODO: use fsum here?
        z = z.sum(axis=-1).sum(axis=-1)  # TODO: use fsum here?

        if original_shape == ():  # is not an Iterable
            return r[0], z[0]
        return r, z

    def slope_and_curve_slice_at_poloidal_angle(theta):
        """
        Gives the tangent vector of the curve at angle (phi, theta).
        Separate to slice_at_poloidal_angle because this one isn't called every time.

        Parameters
        ----------
        theta
        """
        original_shape = np.shape(theta)
        theta = np.atleast_1d(theta)
        len_theta = len(theta)

        nfp_n_phi = np.broadcast_to(nfp * n * phi, [len_theta, len(m), len(n)])
        m_theta = np.broadcast_to(np.outer(theta, m).T, [len(n), len(m), len_theta]).T
        in_paranth = m_theta - nfp_n_phi
        m_long = np.broadcast_to(
            np.broadcast_to(m, [len_theta, len(m)]).T, [len(n), len(m), len_theta]
        ).T

        vr_t = m_long * (
            -surface.rc * np.sin(in_paranth) + surface.rs * np.cos(in_paranth)
        )
        vz_t = m_long * (
            -surface.zc * np.sin(in_paranth) + surface.zs * np.cos(in_paranth)
        )
        vr_t = vr_t.sum(axis=-1).sum(axis=-1)  # TODO: use fsum here?
        vz_t = vz_t.sum(axis=-1).sum(axis=-1)  # TODO: use fsum here?

        vr_t_ = -(m_long**2) * (
            surface.rc * np.cos(in_paranth) + surface.rs * np.sin(in_paranth)
        )
        vz_t_ = -(m_long**2) * (
            surface.zc * np.cos(in_paranth) + surface.zs * np.sin(in_paranth)
        )
        vr_t_ = vr_t_.sum(axis=-1).sum(axis=-1)  # TODO: use fsum here?
        vz_t_ = vz_t_.sum(axis=-1).sum(axis=-1)  # TODO: use fsum here?

        v_length = np.linalg.norm([vr_t, vz_t], axis=0)
        tangents = np.array([vr_t, vz_t]) / v_length

        curvature = (
            vr_t_ * vz_t - vz_t_ * vr_t
        ) / v_length**3  # * abs(vz_t*np.cos(theta) - vr_t*np.sin(theta)) # WRONG

        if original_shape == ():  # is not an Iterable
            return tangents[:, 0], curvature[0]
        return tangents, curvature  # shape == (2, len_theta)

    return slice_at_poloidal_angle, slope_and_curve_slice_at_poloidal_angle


def break_up_angle_curve(curve: npt.NDArray):
    """
    When plotting angle in cartesian coordinates, when theta(t) wraps around from
    2*pi back to 0, we'll get an ugly sharp snap at (t).
    To fix this, instead of plotting it as a single curve, we plot it as several
    curves, breaking it up where the snaps occur.

    Parameters
    ----------
    curve
        the theta(t) curve. (t) does not need to be provided.

    Yields
    ------
    mask
        each mask is a segment of the curve, such that curve[mask] does not include
        any snaps.
    """
    break_point = np.where(abs(np.diff(curve)) > np.pi)[0]
    break_point = [-1, break_point.tolist(), -1]
    for i, j in pairwise(break_point):
        mask = np.zeros_like(curve, dtype=bool)
        mask[i + 1 : j] = True
        yield mask


tau = 2 * np.pi

if __name__ == "__main__":
    local_simsopt_installation = Path("../../../simsopt/")
    data1 = (
        local_simsopt_installation / "tests/test_files/input.LandremanPaul2021_QA_lowres"
    )
    s = SurfaceRZFourier.from_vmec_input(data1, range="half period")
    poloidal_angle = 1
    # assert 0<=poloidal_angle<=tau/s.nfp, ("There's no point going past the point of"
    # "symmetry to the next segment because the"
    # "stellarator's cross-section is periodic past that point.")
    curve_s, derivatives_s = stellarator_curve_function_factory(s, poloidal_angle)
    np.savez("LandremanPaul2021_QA_lowres.npz", rc=s.rc, rs=s.rs, zc=s.zc, zs=s.zs)
    poloidal_sweep = np.linspace(0, tau, 500)

    fig, (ax0, ax1) = plt.subplots(2)
    ax0.plot(*curve_s(poloidal_sweep))
    ax0.set_aspect(1.0)
    ax0.set_title("Shape of curve")
    ax0.set_xlabel("R")
    ax0.set_ylabel("Z")

    tangent_unit_vector, curvature = derivatives_s(poloidal_sweep)
    tangent_angle = np.arctan2(*tangent_unit_vector)
    for i, segment in enumerate(break_up_angle_curve(tangent_angle)):
        ax1.plot(
            poloidal_sweep[segment],
            tangent_angle[segment],
            color="C1",
            label="angle" if i == 0 else None,
        )
    ax1.plot(poloidal_sweep, curvature, label="curvature", color="C2")
    ax1.set_title("1st and 2nd Derivatives")
    ax1.set_xlabel("poloidal angle parameter theta (radian)")
    ax1.legend()
    plt.show()
    plt.close()
