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
import os
import numpy as np
import matplotlib.pyplot as plt
import json
from BLUEPRINT.base.file import get_BP_path
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.nova.coilcage import HelmholtzCage
from bluemira.geometry._deprecated_tools import innocent_smoothie
import tests


def test_pattern():
    path = get_BP_path("BLUEPRINT/nova/test_data", subfolder="tests")
    filename = os.sep.join([path, "tf_centreline_reference.json"])
    tf_centreline = Loop.from_file(filename)
    filename = os.sep.join([path, "lcfs_reference.json"])
    lcfs = Loop.from_file(filename)
    R_0 = 9.06
    Z_0 = 0.0
    B_0 = 4.933
    n_TF = 16
    wp = {"width": 0.55344, "depth": 1.3040917789306743}
    rc = 0.5
    ny = 1
    nr = 3
    npts = 100

    # New HelmholtzCage
    h_cage1 = HelmholtzCage(n_TF, R_0, Z_0, B_0, lcfs, wp, rc, ny, nr, npts)
    h_cage1.set_coil(tf_centreline)
    loops = h_cage1.pattern()
    assert len(loops) == n_TF * nr * ny

    ny = 3
    nr = 1

    h_cage2 = HelmholtzCage(n_TF, R_0, Z_0, B_0, lcfs, wp, rc, ny, nr, npts)
    h_cage2.set_coil(tf_centreline)
    loops = h_cage2.pattern()
    assert len(loops) == n_TF * nr * ny

    ny = 3
    nr = 3

    h_cage3 = HelmholtzCage(n_TF, R_0, Z_0, B_0, lcfs, wp, rc, ny, nr, npts)
    h_cage3.set_coil(tf_centreline)
    loops = h_cage3.pattern()
    assert len(loops) == n_TF * nr * ny

    # Check that the toroidal field is more or less equal for all discretisations
    point = [9, 0, 0]
    bt1 = h_cage1.get_field(point)[1]
    bt2 = h_cage2.get_field(point)[1]
    bt3 = h_cage3.get_field(point)[1]
    assert np.isclose(bt1, bt2)
    assert np.isclose(bt2, bt3)
    assert np.isclose(bt1, bt3)


class TestCariddiBenchmark:
    """
    This is a code comparison benchmark to some work from F. Villone (CREATE) in
    their report DEMO_D_2M97UY
    """

    @classmethod
    def setup_class(cls):
        root = get_BP_path("BLUEPRINT/nova/test_data", subfolder="tests")
        width = 0.64
        depth = 1.15
        B_0 = 5.77
        R_0 = 8.87
        n_TF = 18

        with open(root + "/DEMO_2015_cariddi_ripple_xz.json", "r") as f:
            data = json.load(f)
            x_crip = data["x"]
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

        separatrix = Loop(x=[0, 1, 2], z=[0, -1, 1])  # dummy
        cage = HelmholtzCage(
            n_TF, R_0, 0, B_0, separatrix, {"depth": depth, "width": width}, nr=3, ny=3
        )
        cage.set_coil(coil_loop)
        cls.cage = cage

    def test_cariddi(self):
        ripple = []
        for xr, zr in zip(self.x_rip[1:19], self.z_rip[1:19]):
            ripple.append(self.cage.get_ripple([xr, 0, zr]))

        ripple = np.array(ripple)

        assert np.max(np.abs(ripple - self.cariddi_ripple)) < 0.04

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


if __name__ == "__main__":
    pytest.main([__file__])
