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
Central column shield CAD routines
"""
from BLUEPRINT.base.palettes import BLUE
from BLUEPRINT.cad.cadtools import make_face, revolve, rotate_shape
from BLUEPRINT.cad.component import ComponentCAD
from BLUEPRINT.geometry.loop import Loop


class CentralColumnShieldCAD(ComponentCAD):
    """
    Central column shield CAD constructor class

    Parameters
    ----------
    ccs: CentralColumnShield

        Exects that geom dictionary to be populated (see below)

    ccs.geom : dict
        Dictionary to specify 2D geometry

         - ccs.geom["2D profile"] : Loop

            central column shield 2D profile

    kwargs: dict
        Keyword arguments as for :class:`BLUEPRINT.cad.component.ComponentCAD`

    Attributes
    ----------
    profile: Shell
        2D profile of the central column shield
    n_TF : int
        number of TF coils
    """

    def __init__(self, ccs, **kwargs):

        from BLUEPRINT.systems.centralcolumnshield import CentralColumnShield

        # Check the passed system is the correct type
        if not isinstance(ccs, CentralColumnShield):
            raise TypeError(
                "CentralColumnShieldCAD requires a CentralColumnShield as input"
            )

        # Fetch the 2D profile from geom
        self.profile = ccs.geom["2D profile"]
        self.n_TF = ccs.params.n_TF

        if not isinstance(self.profile, Loop):
            raise TypeError("2D profile key does not map to a Loop object")
        super().__init__("Central column shield", palette=BLUE["CCS"], **kwargs)

    def build(self, **kwargs):
        """
        Build the CAD for the central column shield
        Invoked automatically during :code:`__init__`
        """
        # Make OCC face BLUEPRINT for loop
        # (mixed method is compromise between spliny and non-spliny)
        face = make_face(self.profile)

        # Rotate the 2-D shape
        segment = rotate_shape(face, None, -180 / self.n_TF)

        # Revolve about z-axis to get a segment
        ccs_cad = revolve(segment, None, 360 / self.n_TF)

        # Save
        self.add_shape(ccs_cad)
