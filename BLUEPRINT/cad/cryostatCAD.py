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
Cryostat CAD routines
"""
from BLUEPRINT.base.palettes import BLUE
from BLUEPRINT.cad.cadtools import (
    boolean_cut,
    extrude,
    make_compound,
    make_face,
    revolve,
)
from BLUEPRINT.cad.component import ComponentCAD
from BLUEPRINT.cad.mixins import PlugChopper


class CryostatCAD(PlugChopper, ComponentCAD):
    """
    Cryostat CAD constructor class

    Parameters
    ----------
    cryostat: Systems::Cryostat object
        The cryostat object for which to build the CAD
    """

    def __init__(self, cryostat, **kwargs):
        super().__init__(
            "Cryostat",
            cryostat.geom,
            cryostat.params.n_TF,
            palette=[BLUE["CR"]],
            **kwargs
        )

    def build(self, **kwargs):
        """
        Build the CAD for the cryostat.
        """
        cryostat, n_TF = self.args
        cr = cryostat["plates"].rotate(
            -180 / n_TF, p1=[0, 0, 0], p2=[0, 0, 1], update=False
        )
        cr = revolve(make_face(cr), None, angle=360 / n_TF)
        for name, port in cryostat["ports"].items():
            pp = self.build_port(name, port)
            cr = boolean_cut(cr, pp)
        self.add_shape(cr, name="Cryostat vacuum vessel")

    def build_neutronics(self, **kwargs):
        """
        Build the neutronics CAD for the cryostat.
        """
        cryostat, n_TF = self.args
        cr = revolve(make_face(cryostat["plates"]), None, angle=360)
        plates = cryostat["plates"].copy()
        plates.reverse()
        cutin, cutout = self._make_cutters(plates)
        plugs = []
        for name, port in cryostat["ports"].items():
            pp = self.build_port(name, port)
            ppp = make_compound(self.part_pattern(pp, n_TF))
            cr = boolean_cut(cr, ppp)
            plug = boolean_cut(ppp, cutin)
            plug = boolean_cut(plug, cutout)
            plugs.append(plug)
        self.add_shape(cr, name="Cryostat vacuum vessel")
        self.plugs_for_neutronics(plugs)

    @staticmethod
    def build_port(name, port):
        """
        Build a CAD port for the cryostat.
        """
        if "Upper" in name:
            d, length = "z", 4
            p = make_face(port)
        else:
            d, length = "x", 4
            f = port.translate([-1, 0, 0], update=False)
            p = make_face(f)
        pp = extrude(p, length=length, axis=d)
        return pp
