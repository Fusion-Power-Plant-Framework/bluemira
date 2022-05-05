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
Building CAD routines
"""
import numpy as np

from BLUEPRINT.base.palettes import BLUE
from BLUEPRINT.cad.cadtools import (
    boolean_cut,
    boolean_fuse,
    extrude,
    make_compound,
    make_face,
    revolve,
)
from BLUEPRINT.cad.component import ComponentCAD
from BLUEPRINT.cad.mixins import PlugChopper


class RadiationCAD(PlugChopper, ComponentCAD):
    """
    CAD constructor class for the RadiationShield object

    Parameters
    ----------
    rad_shield: Systems::RadiationShield object
        The RadiationShield object for which to build the CAD
    """

    def __init__(self, rad_shield, **kwargs):
        self.part_info = {
            "tk_rs": rad_shield.params.tk_rs,
            "labyrinth_delta": rad_shield.params.rs_l_d,
            "labyrinth_gap": rad_shield.params.rs_l_gap,
            "n_labyrinth": rad_shield.params.n_rs_lab,
        }

        super().__init__(
            "Radiation shield",
            rad_shield.geom,
            rad_shield.params.n_TF,
            palette=[BLUE["RS"]],
            **kwargs
        )
        self._plugs = None

    def build_port_n_plugs(self, name, rad_port):
        """
        Builds the objects for the penetrations into the radiation shield, and
        the associated plugs
        """
        thickness = self.part_info["tk_rs"]
        n_labyrinth = int(self.part_info["n_labyrinth"])
        labyrinth_gap = self.part_info["labyrinth_gap"]
        total_delta = self.part_info["labyrinth_delta"]
        delta = total_delta / n_labyrinth

        loop0 = rad_port
        p_loop0 = rad_port.offset(-labyrinth_gap)
        if "Upper" in name:
            v = np.array([0, 0, thickness / n_labyrinth])
            o = np.array([0, 0, 0])
        else:
            # This is deal with the circle/square problem
            o = np.array([2, 0, 0])
            loop0 = loop0.translate(-o, update=False)
            p_loop0 = p_loop0.translate(-o, update=False)
            v = np.array([thickness / n_labyrinth, 0, 0])
        p, g = [loop0], [p_loop0]
        for i in range(n_labyrinth + 1)[1:]:
            if i == 1 or i == n_labyrinth + 1:
                u = o
            else:
                u = np.array([0, 0, 0])
            loop = p[-1].offset(delta)
            loop = loop.translate(v + u, update=False)
            p.append(loop)
            p_loop1 = loop.offset(-labyrinth_gap)
            g.append(p_loop1)
        pp, gg = [], []
        for i, (pi, gi) in enumerate(zip(p, g)):
            if i == 1 or i == n_labyrinth + 1:
                u = np.array([0, 0, 0])
            else:
                u = o
            pp.append(extrude(make_face(pi), vec=v + u))
            gg.append(extrude(make_face(gi), vec=v + u))
        ppp, ggg = pp[0], gg[0]
        for shape, shape_plug in zip(pp[1:], gg[1:]):
            ppp = boolean_fuse(ppp, shape)
            ggg = boolean_fuse(ggg, shape_plug)
        return ppp, ggg

    def build(self, **kwargs):  # define component shapes
        """
        Instantiated sector - not suitable for 360 neutronics calculation
        due to high valence centrepoints
        """
        rad_shield, n_TF = self.args
        rs = rad_shield["plates"].rotate(
            -180 / n_TF, p1=[0, 0, 0], p2=[0, 0, 1], update=False
        )
        rs = revolve(make_face(rs), None, angle=360 / n_TF)
        plugs = []
        for name, port in rad_shield["ports"].items():
            ppp, ggg = self.build_port_n_plugs(name, port)
            plugs.append(ggg)
            rs = boolean_cut(rs, ppp)
        self.add_shape(rs, name="Radiation shield")
        self._plugs = plugs

    def build_neutronics(self, **kwargs):
        """
        360 rotated build
        """
        rad_shield, n_TF = self.args
        face = make_face(rad_shield["plates"])
        rs = revolve(face, None, angle=360)

        cutin, cutout = self._make_cutters(rad_shield["plates"])
        plugs = []
        for name, port in rad_shield["ports"].items():
            ppp, ggg = self.build_port_n_plugs(name, port)
            ppp = make_compound(self.part_pattern(ppp, n_TF))
            rs = boolean_cut(rs, ppp)
            plug = make_compound(self.part_pattern(ggg, n_TF))
            plug = boolean_cut(plug, cutin)
            plug = boolean_cut(plug, cutout)
            plugs.append(plug)
        self.add_shape(rs, name="Radiation shield")
        self.plugs_for_neutronics(plugs)
