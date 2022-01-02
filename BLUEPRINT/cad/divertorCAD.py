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
Divertor CAD routines
"""
from BLUEPRINT.base.error import CADError
from BLUEPRINT.base.palettes import BLUE
from BLUEPRINT.cad.cadtools import make_axis, make_mixed_face, revolve
from BLUEPRINT.cad.component import ComponentCAD
from BLUEPRINT.geometry.loop import Loop, MultiLoop


class DivertorCAD(ComponentCAD):
    """
    Divertor CAD constructor class

    Parameters
    ----------
    divertor: Systems::Divertor object
        The divertor object for which to build the CAD
    """

    def __init__(self, divertor, **kwargs):
        super().__init__(
            "Divertor",
            divertor.geom,
            divertor.params.n_TF,
            palette=[BLUE["DIV"]],
            **kwargs,
        )

    def build(self, **kwargs):
        """
        Build the CAD for the divertors.
        """
        divertor, self.n_TF = self.args
        for name, segment in divertor["feed 3D CAD"].items():
            # if name == 'CDS':
            path = segment["path"]
            # div = make_face(segment['profile1'], spline=True)

            # Catch multiple divertors
            if isinstance(segment["profile1"], Loop):  # Single Null
                loops = [segment["profile1"]]
            elif isinstance(segment["profile1"], MultiLoop):  # Double Null
                loops = segment["profile1"].loops
            else:
                raise CADError(
                    f"Unknown object of type: {type(segment['profile1'])}"
                    "in DivertorCAD"
                )

            for i, loop in enumerate(loops):
                div = make_mixed_face(loop)
                angle = path["angle"]
                ax = make_axis(path["rotation axis"][0], [0, 0, 1])
                shape = revolve(div, ax, angle)
                self.add_shape(shape, name=name + f"_{str(i+1)}")

    def build_neutronics(self, **kwargs):
        """
        Build the neutronics CAD for the divertors.
        """
        self.build(**kwargs)
        self.component_pattern(self.n_TF)
