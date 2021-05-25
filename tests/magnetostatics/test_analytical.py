# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt
import pytest
import tests
from BLUEPRINT.geometry.geomtools import circle_seg
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.equilibria.coils import Coil
from BLUEPRINT.magnetostatics.analytical import (
    TrapezoidalPrismCurrentSource,
    CircularArcCurrentSource,
    ArbitraryPlanarCurrentLoop,
    AnalyticalMagnetostaticSolver,
)


def _plot_verification_test(coil, xx, zz, Bx_coil, Bz_coil, Bp_coil, Bx, Bz, Bp):
    def relative_diff(x1, x2):
        diff = np.zeros(x1.shape)
        mask = np.where(x1 != 0)
        diff[mask] = np.abs(x2[mask] - x1[mask]) / x1[mask]
        return diff

    x_corner = np.append(coil.x_corner, coil.x_corner[0])
    z_corner = np.append(coil.z_corner, coil.z_corner[0])

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

    plt.show()


def test_paper_example():
    """
    Verification test.

    Babic and Aykel example

    https://onlinelibrary.wiley.com/doi/epdf/10.1002/jnm.594?saml_referrer=
    """
    # Babic and Aykel example (single trapezoidal prism)
    source = TrapezoidalPrismCurrentSource(
        np.array([0, 0, 0]),
        np.array([2 * 2.154700538379251, 0, 0]),  # This gives b=1
        np.array([0, 1, 0]),
        np.array([0, 0, 1]),
        1,
        1,
        np.pi / 3,
        np.pi / 6,
        4e5,
    )
    field = source.field([2, 2, 2])
    abs_field = 1e3 * np.sqrt(sum(field ** 2))  # Field in mT
    # As per Babic and Aykel paper
    # Assume truncated last digit and not rounded...
    field_7decimals = np.trunc(abs_field * 10 ** 7) / 10 ** 7
    field_7true = 15.5533805
    assert field_7decimals == field_7true

    # Test singularity treatments:
    field = source.field([1, 1, 1])
    abs_field = 1e3 * np.sqrt(sum(field ** 2))  # Field in mT
    # Assume truncated last digit and not rounded...
    field_9decimals = np.trunc(abs_field * 10 ** 9) / 10 ** 9
    field_9true = 53.581000397
    assert field_9decimals == field_9true


def test_2D_vs_3D():
    """
    Verification test. Compare 3-D analytical magnetostatics for a prismed circle
    vs exact 2-D analytical magnetostatics for a rectangular cross-section PF coil.
    """
    xc, zc = 4, 4
    current = 1e6
    dx_coil, dz_coil = 0.5, 0.75

    # Build a PF coil
    coil = Coil(xc, zc, current, dx=dx_coil, dz=dz_coil)
    coil.mesh_coil(0.1)

    # Build a corresponding arbitrary current loop
    xl, yl = circle_seg(xc, (0, 0), npoints=500)
    loop = Loop(x=xl, y=yl)
    loop.translate([0, 0, zc], update=True)
    a = ArbitraryPlanarCurrentLoop(loop, dx_coil, dz_coil, current)

    nx, nz = 25, 25
    Bx, Bz = np.zeros((nx, nz)), np.zeros((nx, nz))
    x = np.linspace(0.1, 5, nx)
    z = np.linspace(2.75, 5.25, nz)
    xx, zz = np.meshgrid(x, z, indexing="ij")
    for i, xi in enumerate(x):
        for j, zi in enumerate(z):
            B = a.field([xi, 0, zi])
            Bx[i, j] = B[0]
            Bz[i, j] = B[2]

    # Use analytical relation and not Green's function
    cBx = coil._control_Bx_analytical(xx, zz)
    cBz = coil._control_Bz_analytical(xx, zz)
    Bx_coil = current * cBx
    Bz_coil = current * cBz
    Bp_coil = np.hypot(Bx_coil, Bz_coil)
    Bp = np.hypot(Bx, Bz)

    # These are never really going to be 0... trapezoids and all that
    assert np.allclose(Bx_coil, Bx, rtol=1e-3)
    assert np.allclose(Bz_coil, Bz, rtol=4e-2)

    if tests.PLOTTING:
        _plot_verification_test(coil, xx, zz, Bx_coil, Bz_coil, Bp_coil, Bx, Bz, Bp)


class TestCircularArcCurrentSource:
    @classmethod
    def setup_class(cls):
        cls.xc, cls.zc = 4, 4
        dx = 0.5
        dz = 1.0
        cls.current = 1e6

        cls.coil = Coil(cls.xc, cls.zc, dx=dx, dz=dz, current=cls.current)

        cls.arc = CircularArcCurrentSource(
            [0, 0, cls.zc],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            dx,
            dz,
            cls.xc,
            2 * np.pi,
            cls.current,
        )

    def test_2D_vs_3D_circular(self):

        nx, nz = 50, 60
        x = np.linspace(self.xc - 2, self.xc + 2, nx)
        z = np.linspace(self.zc - 2, self.zc + 2, nz)
        xx, zz = np.meshgrid(x, z, indexing="ij")

        B = np.zeros((nx, nz, 3))
        for i, xi in enumerate(x):
            for j, zi in enumerate(z):
                B[i, j] = self.arc.field([xi, 0, zi])

        cBx = self.coil._control_Bx_analytical(xx, zz)
        cBz = self.coil._control_Bz_analytical(xx, zz)
        Bx_coil = self.current * cBx
        Bz_coil = self.current * cBz
        Bp_coil = np.hypot(Bx_coil, Bz_coil)
        Bx = B[:, :, 0]
        Bz = B[:, :, 2]
        Bp = np.hypot(Bx, Bz)

        # Because this is a circular calculation, we expect them to be almost identical
        assert np.allclose(Bx_coil, Bx)
        assert np.allclose(Bz_coil, Bz)

        if tests.PLOTTING:
            self.arc.plot()
            _plot_verification_test(
                self.coil, xx, zz, Bx_coil, Bz_coil, Bp_coil, Bx, Bz, Bp
            )

    def test_singularities(self):
        """
        Trigger singularities (ZeroDivisionErrors and such should not come up)
        """
        self.arc.field([self.arc.radius - self.arc.breadth, 0, 0])
        self.arc.field([self.arc.radius - self.arc.breadth, 0, self.arc.depth])
        self.arc.field([self.arc.radius - self.arc.breadth, 0, -self.arc.depth])
        self.arc.field([self.arc.radius + self.arc.breadth, 0, self.arc.depth])
        self.arc.field([self.arc.radius + self.arc.breadth, 0, -self.arc.depth])
        self.arc.field([self.arc.radius, 0, 1])
        self.arc.field([self.arc.radius, 0, 0])
        self.arc.field([0, 0, -1])


def test_analyticalsolvergrouper():
    xc, zc = 4, 4
    current = 1e6
    dx_coil, dz_coil = 0.5, 0.75

    # Build a corresponding arbitrary current loop
    xl, yl = circle_seg(xc, (0, 0), npoints=10)
    loop = Loop(x=xl, y=yl)
    loop.translate([0, 0, zc], update=True)
    a = ArbitraryPlanarCurrentLoop(loop, dx_coil, dz_coil, current)
    loop2 = loop.translate([0, 0, -2 * zc], update=False)
    a2 = ArbitraryPlanarCurrentLoop(loop2, dx_coil, dz_coil, current)
    solver = AnalyticalMagnetostaticSolver([a, a2])

    points = np.random.uniform(low=-10, high=10, size=(10, 3))
    for point in points:
        field = solver.field(point)  # random point :)
        field2 = a.field(point) + a2.field(point)
        assert np.all(field == field2)


def test_mixedsourcesolver():
    current = 1e6
    dx = 0.125
    dz = 0.25
    bar_1 = TrapezoidalPrismCurrentSource(
        [0, 0, 2], [-2, 0, 0], [0, 0, -1], [0, 1, 0], dx, dz, 0, 0, current
    )
    bar_2 = TrapezoidalPrismCurrentSource(
        [-2, 0, 0], [0, 0, -2], [1, 0, 0], [0, 1, 0], dx, dz, 0, 0, current
    )
    bar_3 = TrapezoidalPrismCurrentSource(
        [0, 0, -2], [2, 0, 0], [0, 0, 1], [0, 1, 0], dx, dz, 0, 0, current
    )
    bar_4 = TrapezoidalPrismCurrentSource(
        [2, 0, 0], [0, 0, 2], [-1, 0, 0], [0, 1, 0], dx, dz, 0, 0, current
    )

    arc_1 = CircularArcCurrentSource(
        [-1, 0, 1], [0, 0, 1], [-1, 0, 0], [0, 1, 0], dz, dx, 1, np.pi / 2, current
    )
    arc_2 = CircularArcCurrentSource(
        [-1, 0, -1], [-1, 0, 0], [0, 0, -1], [0, 1, 0], dz, dx, 1, np.pi / 2, current
    )
    arc_3 = CircularArcCurrentSource(
        [1, 0, -1], [0, 0, -1], [1, 0, 0], [0, 1, 0], dz, dx, 1, np.pi / 2, current
    )
    arc_4 = CircularArcCurrentSource(
        [1, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], dz, dx, 1, np.pi / 2, current
    )

    solver = AnalyticalMagnetostaticSolver(
        [bar_1, bar_2, bar_3, bar_4, arc_1, arc_2, arc_3, arc_4]
    )

    nx, nz = 100, 100
    x = np.linspace(-4, 4, nx)
    z = np.linspace(-4, 4, nz)
    xx, zz = np.meshgrid(x, z, indexing="ij")
    B = np.zeros((nx, nz, 3))
    for i, xi in enumerate(x):
        for j, zi in enumerate(z):
            B[i, j, :] = solver.field([xi, 0, zi])

    Bt = B[:, :, 1]

    # Test symmetry of the field in four quadranrs (rotation by matrix manipulations :))
    # Bottom-left (reference)
    bt_bl = Bt[:50, :50]
    # Bottom-right
    bt_br = Bt[50:, :50][::-1].T
    # Top-right
    bt_tr = Bt[50:, 50:][::-1].T[::-1]
    # Top-left
    bt_tl = Bt[:50, 50:].T[::-1]

    assert np.allclose(bt_bl, bt_br)
    assert np.allclose(bt_bl, bt_tr)
    assert np.allclose(bt_bl, bt_tl)

    if tests.PLOTTING:
        solver.plot()
        f, ax = plt.subplots()
        ax.contourf(xx, zz, Bt)
        ax.set_aspect("equal")


if __name__ == "__main__":
    pytest.main([__file__])
