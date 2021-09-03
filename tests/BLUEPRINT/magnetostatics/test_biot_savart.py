# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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

import pytest
import numpy as np
import matplotlib.pyplot as plt
from bluemira.base.constants import MU_0
from bluemira.base.look_and_feel import plot_defaults
from BLUEPRINT.geometry.geomtools import circle_seg
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.magnetostatics.greens import (
    greens_all,
    circular_coil_inductance_elliptic,
    circular_coil_inductance_kirchhoff,
)
from BLUEPRINT.equilibria.gridops import Grid
from BLUEPRINT.magnetostatics.biot_savart import BiotSavartFilament
import tests


def test_biot_savart_loop():
    """
    Test that a circular GreenFieldLoop (Biot Savart differential form)
    matches with the analytical Greens function for a circular coil.

    This is a verification test.
    """
    nx, nz = 100, 100
    x_coil, z_coil = 4, 0
    current = 10e7
    radius = 0.0000000000001
    grid = Grid(0.1, 10, -5, 5, nx, nz)

    # Analytical field values
    _, Bx, Bz = greens_all(x_coil, z_coil, grid.x, grid.z)

    # Differential Biot-Savart approach
    # We use 2000 instead of 1000, because Simon's method did some extra
    # discretisation, which approximated curves
    xc, yc = circle_seg(x_coil, h=[0, z_coil], npoints=2000)
    loop = Loop(xc, yc, 0)
    bsf = BiotSavartFilament(loop, radius)

    Bx2, Bz2 = np.zeros((nx, nz)), np.zeros((nx, nz))
    for i in range(nx):
        for j in range(nz):
            bx, _, bz = bsf.field([grid.x_1d[i], 0, grid.z_1d[j]])
            Bx2[i, j] = bx
            Bz2[i, j] = bz

    Bx *= current
    Bz *= current
    Bx2 *= current
    Bz2 *= current
    Bp = np.sqrt(Bx ** 2 + Bz ** 2)
    Bp2 = np.sqrt(Bx2 ** 2 + Bz ** 2)

    # Relative errors
    err_Bx = np.abs(100 * (Bx - Bx2) / Bx)
    err_Bz = np.abs(100 * (Bz - Bz2) / Bz)
    err_Bp = np.abs(100 * (Bp - Bp2) / Bp)

    # RMS errors
    rms_Bx = np.average((Bx - Bx2) ** 2)
    rms_Bz = np.average((Bz - Bz2) ** 2)
    rms_Bp = np.average((Bp - Bp2) ** 2)

    if tests.PLOTTING:
        plot_errors(Bx, Bz, Bp, Bx2, Bz2, Bp2, grid)

    # Note that we're using a lot of points on our circle...
    # But the errors are predominantly around the centre of the coil
    # So a for 2000 point circle we can get less than 0.5 % discrepancy with
    # an analytical solution (Green's functions for Bx and Bz)
    assert np.amax(err_Bx) <= 0.0081
    assert np.amax(err_Bz) <= 0.49
    assert np.amax(err_Bp) <= 0.003
    assert rms_Bx <= 5e-7
    assert rms_Bz <= 5e-7
    assert rms_Bp <= 5e-7
    assert np.allclose(Bp, Bp2, rtol=5e-5)

    # Current on axis analytical relation
    centreline = np.c_[np.zeros(100), np.zeros(100), np.linspace(-10, 10, 100)]
    bz_differential = np.zeros(100)
    for i in range(100):
        bz_differential[i] = current * bsf.field(centreline[i])[2]

    bz_analytical = (
        (MU_0 / (4 * np.pi))
        * (2 * np.pi * x_coil ** 2 * current)
        / (centreline.T[2] ** 2 + x_coil ** 2) ** (3 / 2)
    )

    assert np.allclose(bz_analytical, bz_differential)


def plot_errors(Bx, Bz, Bp, Bx2, Bz2, Bp2, grid):
    """
    Plotting utility for supporting comparisons. Not a test.
    """
    plot_defaults()
    f, ax = plt.subplots(3, 4)
    ax[0, 0].contourf(grid.x, grid.z, Bx)
    ax[0, 0].set_title("$B_{x}$ Green's function")
    ax[0, 1].contourf(grid.x, grid.z, Bx2)
    ax[0, 1].set_title("$B_{x}$ Biot-Savart differential function")
    cm = ax[0, 2].contourf(grid.x, grid.z, 100 * (Bx - Bx2) / Bx)
    f.colorbar(cm, ax[0, 3])
    ax[0, 2].set_title("Relative error [%]")
    ax[0, 0].set_aspect("equal")
    ax[0, 1].set_aspect("equal")
    ax[0, 2].set_aspect("equal")
    ax[0, 3].set_aspect(20)

    ax[1, 0].contourf(grid.x, grid.z, Bx)
    ax[1, 0].set_title("$B_{z}$ Green's function")
    ax[1, 1].contourf(grid.x, grid.z, Bx2)
    ax[1, 1].set_title("$B_{z}$ Biot-Savart differential function")
    cm = ax[1, 2].contourf(grid.x, grid.z, 100 * (Bz - Bz2) / Bz)
    f.colorbar(cm, ax[1, 3])
    ax[1, 2].set_title("Relative error [%]")

    ax[1, 0].set_aspect("equal")
    ax[1, 1].set_aspect("equal")
    ax[1, 2].set_aspect("equal")
    ax[1, 3].set_aspect(20)

    ax[2, 0].contourf(grid.x, grid.z, Bp)
    ax[2, 0].set_title("$B_{p}$ Green's function")
    ax[2, 1].contourf(grid.x, grid.z, Bp2)
    ax[2, 1].set_title("$B_{p}$ Biot-Savart differential function")
    cm = ax[2, 2].contourf(grid.x, grid.z, 100 * (Bp - Bp2) / Bp)
    ax[2, 2].set_title("Relative error [%]")
    f.colorbar(cm, ax[2, 3])

    ax[2, 0].set_aspect("equal")
    ax[2, 1].set_aspect("equal")
    ax[2, 2].set_aspect("equal")
    ax[2, 3].set_aspect(20)
    plt.show()


@pytest.mark.longrun
def test_inductance():
    radii = np.linspace(1, 100, 100)
    rci = np.linspace(0.0001, 1, 100)
    ind, ind2 = np.zeros((100, 100)), np.zeros((100, 100))

    ind3 = np.zeros((100, 100))
    for i, r in enumerate(radii):
        x, y = circle_seg(r, npoints=200)
        loop = Loop(x=x, z=y)
        for j, rc in enumerate(rci):
            bsf = BiotSavartFilament(loop, rc)
            ind[i, j] = circular_coil_inductance_elliptic(r, rc)
            ind2[i, j] = circular_coil_inductance_kirchhoff(r, rc)
            ind3[i, j] = bsf.inductance()

    if tests.PLOTTING:
        levels = np.linspace(np.amin(ind), np.amax(ind), 20)
        f, ax = plt.subplots(1, 4)
        xx, yy = np.meshgrid(radii, rci)
        ax[0].contourf(xx, yy, ind, levels=levels)
        ax[1].contourf(xx, yy, ind2, levels=levels)
        ax[2].contourf(xx, yy, ind3, levels=levels)
        diff = 100 * (ind - ind3) / ind
        cm = ax[3].contourf(xx, yy, diff, levels=np.linspace(-10, 10, 100))
        f.colorbar(cm)
        plt.show()


if __name__ == "__main__":
    tests.PLOTTING = True
    pytest.main([__file__])
