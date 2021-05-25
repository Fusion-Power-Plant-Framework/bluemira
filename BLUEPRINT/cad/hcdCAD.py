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

"""
HCD CAD routines
"""
from BLUEPRINT.base.palettes import BLUE
from BLUEPRINT.cad.component import ComponentCAD
from BLUEPRINT.cad.cadtools import make_face, make_axis, revolve


class NBIoCAD(ComponentCAD):
    """
    Neutral Beam Injector CAD building class.
    """

    def __init__(self, neutral_beam, **kwargs):
        ComponentCAD.__init__(
            self,
            "Neutral neam injector",
            neutral_beam.geom,
            neutral_beam.n,
            palette=[BLUE["HCD"]],
            **kwargs
        )

    def build(self, **kwargs):
        """
        Build the CAD for the NBI.
        """
        neutral_beam, n = self.args
        for seg in neutral_beam["feed 3D CAD"]:
            segment = neutral_beam["feed 3D CAD"][seg]
            path = segment["path"]
            face = make_face(segment["profile1"])
            ax = make_axis(path["rotation axis"][0], (0, 0, 1))
            shape = revolve(face, ax, path["angle"])
            self.add_shape(shape, name=seg)


class ECDoCAD(ComponentCAD):
    """
    Electron Cyclotron Drive CAD building class.
    """

    def __init__(self, e_cyclotron, **kwargs):
        ComponentCAD.__init__(
            self,
            "Electron cyclotron",
            e_cyclotron.geom,
            e_cyclotron.n,
            palette=[BLUE["HCD"]],
            **kwargs
        )

    def build(self, **kwargs):
        """
        Build the CAD for the ECD.
        """
        e_cyclotron, n = self.args
        for seg in e_cyclotron["feed 3D CAD"]:
            segment = e_cyclotron["feed 3D CAD"][seg]
            path = segment["path"]
            bb_face = make_face(segment["profile1"])
            ax = make_axis(path["rotation axis"][0], (0, 0, 1))
            shape = revolve(bb_face, ax, path["angle"])
            self.add_shape(shape, name=seg)


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
