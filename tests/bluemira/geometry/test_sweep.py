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

import numpy as np

from bluemira.geometry.tools import (
    make_polygon,
    make_circle,
    revolve_shape,
    sweep_shape,
    extrude_shape,
)
from bluemira.geometry.face import BluemiraFace


class TestSweep:
    def test_straight(self):
        path = make_polygon([[0, 0, 0], [0, 0, 1]])
        profile = make_polygon(
            [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], closed=True
        )

        extrusion = extrude_shape(BluemiraFace(profile), vec=(0, 0, 1))
        sweep = sweep_shape(profile, path, solid=True)

        assert np.isclose(extrusion.volume, sweep.volume)

    def test_circle(self):
        path = make_circle(start_angle=0, end_angle=180)
        profile = make_polygon(
            [[0.5, 0, -0.5], [1.5, 0, -0.5], [1.5, 0, 0.5], [0.5, 0, 0.5]], closed=True
        )

        revolution = revolve_shape(BluemiraFace(profile), degree=180)
        sweep = sweep_shape(profile, path, solid=True)

        assert np.isclose(revolution.volume, sweep.volume)


if __name__ == "__main__":
    import bluemira.geometry._freecadapi as cadapi
    import freecad
    import Part
    from bluemira.geometry.face import BluemiraFace
    from bluemira.geometry.wire import BluemiraWire
    from bluemira.geometry.shell import BluemiraShell
    from bluemira.geometry.solid import BluemiraSolid

    path = make_circle(start_angle=0, end_angle=180)
    profile = make_polygon(
        [[0.5, 0, -0.5], [1.5, 0, -0.5], [1.5, 0, 0.5], [0.5, 0, 0.5]], closed=True
    )

    fc_sweep = cadapi.sweep_shape(profile._shape, path._shape, solid=True)
    sweep = sweep_shape(profile, path, solid=True)

    faces = fc_sweep.Shells[0].Faces
    wires = [f.Wires for f in faces]
    bm_wires = [BluemiraWire(w) for w in wires]
    bm_faces = [BluemiraFace(w) for w in bm_wires]
    bm_shell = BluemiraShell(bm_faces)
    recon_fc_sweep = Part.Solid(Part.Shell(faces))
    recon_bm_sweep = BluemiraSolid(bm_shell)

    print(fc_sweep.Volume)
    print(recon_fc_sweep.Volume)
    print(recon_bm_sweep.volume)
    print(sweep.volume)
