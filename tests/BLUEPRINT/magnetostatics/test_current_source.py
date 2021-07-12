# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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

import numpy as np
import matplotlib.pyplot as plt
import pytest
import tests
import json
from BLUEPRINT.base.file import get_BP_path
from BLUEPRINT.utilities.tools import innocent_smoothie
from BLUEPRINT.geometry.geomtools import circle_seg
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.equilibria.coils import Coil
from BLUEPRINT.magnetostatics.baseclass import SourceGroup
from BLUEPRINT.magnetostatics.current_source import (
    ArbitraryPlanarRectangularXSCircuit,
    HelmholtzCage,
)
from BLUEPRINT.magnetostatics.trapezoidal_prism import TrapezoidalPrismCurrentSource
from BLUEPRINT.magnetostatics.circular_arc import CircularArcCurrentSource
from tests.BLUEPRINT.magnetostatics.setup_methods import _plot_verification_test


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
    a = ArbitraryPlanarRectangularXSCircuit(loop, dx_coil, dz_coil, current)

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


def test_analyticalsolvergrouper():
    xc, zc = 4, 4
    current = 1e6
    dx_coil, dz_coil = 0.5, 0.75

    # Build a corresponding arbitrary current loop
    xl, yl = circle_seg(xc, (0, 0), npoints=10)
    loop = Loop(x=xl, y=yl)
    loop.translate([0, 0, zc], update=True)
    a = ArbitraryPlanarRectangularXSCircuit(loop, dx_coil, dz_coil, current)
    loop2 = loop.translate([0, 0, -2 * zc], update=False)
    a2 = ArbitraryPlanarRectangularXSCircuit(loop2, dx_coil, dz_coil, current)
    solver = SourceGroup([a, a2])

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

    solver = SourceGroup([bar_1, bar_2, bar_3, bar_4, arc_1, arc_2, arc_3, arc_4])

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


class TestCariddiBenchmark:
    """
    This is a code comparison benchmark to some work from F. Villone (CREATE) in
    their report DEMO_D_2M97UY
    """

    @classmethod
    def setup_class(cls):
        root = get_BP_path("BLUEPRINT/magnetostatics/test_data", subfolder="tests")
        width = 0.64
        depth = 1.15
        B_0 = 5.77
        R_0 = 8.87
        n_TF = 18

        with open(root + "/DEMO_2015_cariddi_ripple_xz.json", "r") as f:
            data = json.load(f)
            cls.cariddi_ripple = data["z"]

        with open(root + "/DEMO_2015_ripple_xz.json", "r") as f:
            data = json.load(f)
            cls.x_rip = data["x"]
            cls.z_rip = data["z"]

        with open(root + "/DEMO_2015_TF_xz.json", "r") as f:
            data = json.load(f)
            x = data["x"]
            z = data["z"]
            coil_loop = Loop(x=x, z=z)
            coil_loop.close()
            coil_loop.interpolate(300)
            coil_loop = coil_loop.offset(width / 2)

        # Smooth out graphically determined TF centreline...
        x, z = innocent_smoothie(coil_loop.x, coil_loop.z, n=150, s=0.02)
        coil_loop = Loop(x=x[:-10], z=z[:-10])
        coil_loop.close()
        cls.coil_loop = coil_loop

        circuit = ArbitraryPlanarRectangularXSCircuit(
            coil_loop, width / 2, depth / 2, current=1.0
        )

        # Set the current in the HelmholtzCage to generate the desired B_T,0 field
        cage = HelmholtzCage(circuit, n_TF)
        field_response = cage.field([R_0, 0, 0])[1]
        current = B_0 / field_response
        cage.set_current(current)
        cls.cage = cage

    def test_cariddi(self):
        ripple = []
        for xr, zr in zip(self.x_rip[1:19], self.z_rip[1:19]):
            ripple.append(self.cage.ripple([xr, 0, zr]))

        ripple = np.array(ripple)

        if tests.PLOTTING:
            f, (ax2, ax) = plt.subplots(1, 2)
            ax.scatter(
                list(range(1, 19)), self.cariddi_ripple, marker="o", label="CARIDDI"
            )
            ax.scatter(
                list(range(1, 19)), ripple, marker="x", label="BLUEPRINT", zorder=20
            )
            ax.legend(loc="upper left")

            ax.set_ylabel("$\\delta_{\\phi}$ [%]")
            ax.yaxis.set_label_position("right")
            ax.yaxis.tick_right()
            ax.set_xlabel("Point index")
            ax.set_xticks(np.arange(1, 19, 2))

            self.coil_loop.plot(ax2, fill=False)
            ax2.plot(self.x_rip[1:19], self.z_rip[1:19], "s", marker=".", color="r")
            plt.show()

        assert np.max(np.abs(ripple - self.cariddi_ripple)) < 0.04


if __name__ == "__main__":
    pytest.main([__file__])
