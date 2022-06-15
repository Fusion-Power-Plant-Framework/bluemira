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
First wall CAD routines
"""
from BLUEPRINT.base.palettes import BLUE
from BLUEPRINT.cad.cadtools import make_face, revolve, rotate_shape
from BLUEPRINT.cad.component import ComponentCAD


class FirstWallCAD(ComponentCAD):
    """
    FirstWall CAD constructor class

    Parameters
    ----------
    firstwall: ReactorSystem
        Expects one of:

         - :class:`BLUEPRINT.systems.firstwall.FirstWallSN`
         - :class:`BLUEPRINT.systems.firstwall.FirstWallDN`

        Exects that firstwall.geom dictionary to be populated (see below)

    firstwall.geom : dict
        Dictionary to specify 2D geometry

    kwargs: dict
        Keyword arguments as for :class:`BLUEPRINT.cad.component.ComponentCAD`

    Attributes
    ----------
    plot_loops : list
        List of Loops containing 2D profiles of components to be plotted
    n_TF : int
        number of TF coils
    """

    def __init__(self, firstwall, **kwargs):

        from BLUEPRINT.systems.firstwall import FirstWallDN, FirstWallSN

        # Check the passed system is the correct type
        if not isinstance(firstwall, (FirstWallSN, FirstWallDN)):
            raise TypeError(
                "FirstWallCAD requires either FirstWallSN or FirstWallDN as argument"
            )

        # Fetch the 2D profile from geom
        plot_names = firstwall.xz_plot_loop_names
        self.plot_loops = []
        for name in plot_names:
            obj = firstwall.geom[name]
            self.plot_loops.append(obj)

        self.n_TF = firstwall.params.n_TF

        super().__init__("Reactor first wall", palette=BLUE["FW"], **kwargs)

    def build(self, **kwargs):
        """
        Build the CAD for the first wall.
        Invoked automatically during :code:`__init__`
        """
        # Make OCC faces
        shapes = []
        for loop in self.plot_loops:
            face = make_face(loop)
            shapes.append(face)

        for shape in shapes:
            # Rotate the 2-D shape
            shape_rot = rotate_shape(shape, None, -180 / self.n_TF)

            # Revolve about z-axis to get a segment
            segment = revolve(shape_rot, None, 360 / self.n_TF)

            # Save
            self.add_shape(segment)
