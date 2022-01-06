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
Plasma CAD routines
"""
import numpy as np

from BLUEPRINT.base.palettes import BLUE
from BLUEPRINT.cad.cadtools import make_face, revolve
from BLUEPRINT.cad.component import ComponentCAD
from BLUEPRINT.geometry.loop import MultiLoop


class PlasmaCAD(ComponentCAD):
    """
    Plasma CAD constructor class

    Parameters
    ----------
    plasma: Systems::Plasma object
    """

    def __init__(self, plasma, **kwargs):
        palette = BLUE["PL"]
        super().__init__(
            "Plasma",
            plasma.geom,
            plasma.params.n_TF,
            palette=palette,
            n_colors=1,
            **kwargs
        )

    @staticmethod
    def build_single(plasma, n_TF, offset, width):
        """
        Build a single sector of plasma CAD.
        """
        angle = offset - 180 / n_TF

        p = plasma["LCFS"].rotate(angle, p1=[0, 0, 0], p2=[0, 0, 1], update=False)
        p.interpolate(200)
        p = make_face(p, spline=False)
        lcfs = revolve(p, None, width)
        p = plasma["Separatrix"].rotate(angle, p1=[0, 0, 0], p2=[0, 0, 1], update=False)

        if isinstance(p, MultiLoop):
            loops = p.loops
        else:
            loops = [p]

        seps = []
        for loop in loops:
            loop.interpolate(200)
            loop = make_face(loop, spline=False)
            sep = revolve(loop, None, width)
            seps.append(sep)
        return lcfs, seps

    def build(self, **kwargs):
        """
        Build the plasma CAD.
        """
        plasma, n_TF = self.args
        theta = np.zeros(2)
        dtheta = 360 / n_TF
        n_pattern = np.zeros(n_TF)  # left / right pattern
        n_pattern[1::2] = [i + 1 for i in range(len(n_pattern[1::2]))]
        n_pattern[2::2] = [-(i + 1) for i in range(len(n_pattern[2::2]))]
        for i, n in enumerate(n_pattern):
            theta[0] = np.min([theta[0], n * dtheta])
            theta[1] = np.max([theta[1], n * dtheta])
            lcfs, seps = self.build_single(plasma, n_TF, theta[0], (i + 1) * dtheta)
            self.add_shape(lcfs, name="Plasma_LCFS_{}".format(i), transparency=0.5)
            for sep in seps:
                self.add_shape(sep, name="Plasma_sep_{}".format(i), transparency=0.5)

    def build_neutronics(self, **kwargs):
        """
        Build the neutronics CAD for the plasma.
        """
        plasma, n_TF = self.args
        p = make_face(plasma["LCFS"])
        plasma = revolve(p, None, angle=360)
        self.add_shape(plasma, name="Plasma_0", transparency=0.5)
