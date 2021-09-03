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
"""
Created on Thu Nov 22 16:03:04 2018

@author: matti
"""

import os
import pytest
import time
from matplotlib import pyplot as plt
from BLUEPRINT.base.file import get_BP_path
from bluemira.base.look_and_feel import bluemira_print
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.equilibria.run import (
    AbExtraEquilibriumProblem,
    AbInitioEquilibriumProblem,
)

import tests


@pytest.mark.longrun
@pytest.mark.skipif(not tests.PLOTTING, reason="plotting disabled")
class TestAbInitioEquilibriumProblem:
    @pytest.mark.longrun
    def test_EUDEMO(self):  # noqa (N802)
        t = time.time()
        fp = get_BP_path("Geometry", subfolder="data/BLUEPRINT")
        tf = Loop.from_file(os.sep.join([fp, "TFreference.json"]))
        tf = tf.offset(0.5)
        a = AbInitioEquilibriumProblem(
            R_0=9,
            B_0=5.834,
            A=3.1,
            Ip=18.6679e6,
            betap=2.5,
            li=0.8,
            kappa=1.692,
            delta=0.373,
            r_cs=2.555,
            tk_cs=0.556,
            tfbnd=tf,
            n_PF=6,
            n_CS=5,
            eqtype="SN",
            rtype="Normal",
            profile=None,
        )

        fp = get_BP_path("eqdsk", subfolder="data/BLUEPRINT")
        fn = os.path.join(fp, "Equil_AR3d1_2015_04_v2_EOF_CSred_fine_final.eqdsk")
        b = AbExtraEquilibriumProblem(fn)
        b.solve()
        b.regrid()
        bluemira_print(f"Runtime: {time.time()-t:.2f} seconds")
        plt.close("all")
        a.plot()
        plt.show()
        # plt.close('all')


if __name__ == "__main__":
    pytest.main([__file__])
