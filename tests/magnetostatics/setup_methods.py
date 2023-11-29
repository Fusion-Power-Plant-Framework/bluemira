# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import matplotlib.pyplot as plt
import numpy as np


def _plot_verification_test(
    xc, zc, dx, dz, xx, zz, Bx_coil, Bz_coil, Bp_coil, Bx, Bz, Bp
):
    def relative_diff(x1, x2):
        diff = np.zeros(x1.shape)
        mask = np.nonzero(x1)
        diff[mask] = np.abs(x2[mask] - x1[mask]) / x1[mask]
        return diff

    x_corner = np.array([xc - dx, xc + dx, xc + dx, xc - dx, xc - dx])
    z_corner = np.array([zc - dz, zc - dz, zc + dz, zc + dz, zc - dz])

    f, ax = plt.subplots(3, 3)
    cm = ax[0, 0].contourf(xx, zz, Bx_coil)
    f.colorbar(cm, ax=ax[0, 0])
    ax[0, 0].set_title(
        "Axisymmetric analytical PF coil (rectangular cross-section)\n$B_x$"
    )
    cm = ax[0, 1].contourf(xx, zz, Bx)
    f.colorbar(cm, ax=ax[0, 1])
    ax[0, 1].set_title("Analytical PF coil of prisms (rectangular cross-section)\n$B_x$")
    cm = ax[0, 2].contourf(xx, zz, relative_diff(Bx_coil, Bx))
    f.colorbar(cm, ax=ax[0, 2])

    cm = ax[1, 0].contourf(xx, zz, Bz_coil)
    f.colorbar(cm, ax=ax[1, 0])
    ax[1, 0].set_title("$B_z$")
    cm = ax[1, 1].contourf(xx, zz, Bz)
    f.colorbar(cm, ax=ax[1, 1])
    ax[1, 1].set_title("$B_z$")
    cm = ax[1, 2].contourf(xx, zz, relative_diff(Bz_coil, Bz))
    ax[1, 1].set_title("$B_z$")
    f.colorbar(cm, ax=ax[1, 2])

    cm = ax[2, 0].contourf(xx, zz, Bp_coil)
    f.colorbar(cm, ax=ax[2, 0])
    ax[2, 0].set_title("$B_p$")
    cm = ax[2, 1].contourf(xx, zz, Bp)
    f.colorbar(cm, ax=ax[2, 1])
    ax[2, 1].set_title("$B_p$")
    cm = ax[2, 2].contourf(xx, zz, relative_diff(Bp_coil, Bp))
    f.colorbar(cm, ax=ax[2, 2])
    for i, axis in enumerate(ax.flat):
        if (i + 1) % 3 == 0:
            axis.set_title("relative difference [%]")
        axis.plot(x_corner, z_corner, color="r")
        axis.set_xlabel("$x$ [m]")
        axis.set_ylabel("$z$ [m]")
        axis.set_aspect("equal")
