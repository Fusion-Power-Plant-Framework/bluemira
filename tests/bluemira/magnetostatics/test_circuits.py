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

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

import tests
from bluemira.base.file import get_bluemira_path
from bluemira.geometry._deprecated_loop import Loop
from bluemira.geometry._deprecated_tools import innocent_smoothie
from bluemira.geometry.tools import make_circle
from bluemira.magnetostatics.baseclass import SourceGroup
from bluemira.magnetostatics.circuits import (
    ArbitraryPlanarRectangularXSCircuit,
    HelmholtzCage,
)
from bluemira.magnetostatics.circular_arc import CircularArcCurrentSource
from bluemira.magnetostatics.trapezoidal_prism import TrapezoidalPrismCurrentSource


def test_analyticalsolvergrouper():
    xc, zc = 4, 4
    current = 1e6
    dx_coil, dz_coil = 0.5, 0.75

    # Build a corresponding arbitrary current loop
    circle = make_circle(center=[0, 0, zc], radius=xc).discretize(ndiscr=10)
    a = ArbitraryPlanarRectangularXSCircuit(circle, dx_coil, dz_coil, current)
    circle2 = make_circle(center=[0, 0, -zc], radius=xc).discretize(ndiscr=10)
    a2 = ArbitraryPlanarRectangularXSCircuit(circle2, dx_coil, dz_coil, current)
    solver = SourceGroup([a, a2])

    points = np.random.uniform(low=-10, high=10, size=(10, 3))
    for point in points:
        field = solver.field(*point)  # random point :)
        field2 = a.field(*point) + a2.field(*point)
        assert np.all(field == field2)

    field = solver.field(*points.T)
    new_current = 2e6
    solver.set_current(new_current)
    field_new = solver.field(*points.T)

    assert np.allclose(field_new, new_current / current * field)


def test_sourcegroup_set_current():
    circle = make_circle(radius=10).discretize(ndiscr=50)
    dx_coil, dz_coil = 0.5, 0.75
    a = ArbitraryPlanarRectangularXSCircuit(circle, dx_coil, dz_coil, current=1)
    x, y, z = 4, 4, 4
    response = a.field(x, y, z)
    new_current = 1e6
    a.set_current(new_current)
    new_response = a.field(x, y, z)
    assert np.allclose(new_response, new_current * response)


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

    nx, nz = 20, 20
    nx2, nz2 = nx // 2, nz // 2
    x = np.linspace(-4, 4, nx)
    z = np.linspace(-4, 4, nz)
    xx, zz = np.meshgrid(x, z, indexing="ij")

    _, Bt, _ = solver.field(xx, np.zeros_like(xx), zz)

    # Test symmetry of the field in four quadranrs (rotation by matrix manipulations :))
    # Bottom-left (reference)
    bt_bl = Bt[:nx2, :nz2]
    # Bottom-right
    bt_br = Bt[nx2:, :nz2][::-1].T
    # Top-right
    bt_tr = Bt[nx2:, nz2:][::-1].T[::-1]
    # Top-left
    bt_tl = Bt[:nx2, nz2:].T[::-1]

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
        root = get_bluemira_path("bluemira/magnetostatics/test_data", subfolder="tests")
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
        field_response = cage.field(R_0, 0, 0)[1]
        current = B_0 / field_response
        cage.set_current(current)
        cls.cage = cage

    def test_cariddi(self):

        ripple = self.cage.ripple(self.x_rip[1:19], np.zeros(18), self.z_rip[1:19])

        if tests.PLOTTING:
            f, (ax2, ax) = plt.subplots(1, 2)
            ax.scatter(
                list(range(1, 19)), self.cariddi_ripple, marker="o", label="CARIDDI"
            )
            ax.scatter(
                list(range(1, 19)), ripple, marker="x", label="bluemira", zorder=20
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
