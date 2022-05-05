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
Blanket CAD routines
"""
from BLUEPRINT.base.palettes import BLUE
from BLUEPRINT.cad.cadtools import (
    make_axis,
    make_face,
    make_mixed_face,
    revolve,
    rotate_shape,
)
from BLUEPRINT.cad.component import ComponentCAD


class BlanketCAD(ComponentCAD):
    """
    Blanket CAD constructor class

    Parameters
    ----------
    blanket: Systems::BreedingBlanket object
        The breeding blanket object for which to build the CAD
    """

    def __init__(self, blanket, **kwargs):
        super().__init__(
            "Breeding blanket",
            blanket.geom,
            blanket.params.n_TF,
            palette=BLUE["BB"],
            **kwargs,
        )
        self.n_TF = None

    def build(self, **kwargs):
        """
        Build the CAD for the blankets.
        """
        blanket, self.n_TF = self.args
        sub_names = ["fw", "bz", "manifold", "bss"]
        for name, seg in blanket["feed 3D CAD"].items():
            ax = make_axis(seg["path"]["rotation axis"][0], [0, 0, 1])
            for part, sub in zip(seg["parts"][:3], sub_names[:3]):
                # First few are squarey
                face = make_face(part)
                shape = revolve(face, ax, seg["path"]["angle"])
                self.add_shape(shape, name=name + "_" + sub)

            # Back supporting structure is curvy and can be multiple Loops
            for i, part in enumerate(seg["parts"][3:]):
                face = make_mixed_face(part)
                shape = revolve(face, ax, seg["path"]["angle"])
                self.add_shape(shape, name=name + "_" + f"bss_{i}")

    def build_parallel(self):
        """
        Build parallel blanket segmentation
        """
        blanket, self.n_TF = self.args
        blanket["FW cut"].close()

        for seg in ["LIBS", "RIBS"]:
            segment = blanket["feed 3D CAD"][seg]
            path = segment["path"]
            angle = path["angle"]

            ax = make_axis(path["rotation axis"][0], [0, 0, 1])
            bb_face = make_mixed_face(segment["body"])
            shape = revolve(bb_face, ax, angle)
            self.add_shape(shape, name=seg + "_body")
            bb_fw = make_face(segment["fw"])
            shape = revolve(bb_fw, ax, angle)
            self.add_shape(shape, name=seg + "_fw")

    def build_neutronics(self, **kwargs):
        """
        Build the neutronics CAD for the blankets.
        """
        self.build(**kwargs)
        self.component_pattern(self.n_TF)


# This was copy-pasted from firstwall. TODO: inherit from common class
class STBlanketCAD(ComponentCAD):
    """
    STBreedingBlanket CAD constructor class

    Parameters
    ----------
    blanket: ReactorSystem
        Exects that blanket.geom dictionary to be populated.

    kwargs: dict
        Keyword arguments as for :class:`BLUEPRINT.cad.component.ComponentCAD`

    Attributes
    ----------
    plot_loops : list
        List of Loops containing 2D profiles of components to be plotted
    n_TF : int
        number of TF coils
    """

    def __init__(self, blanket, **kwargs):

        # Fetch the 2D profile from geom
        plot_names = blanket.xz_plot_loop_names
        self.plot_loops = []
        for name in plot_names:
            obj = blanket.geom[name]
            self.plot_loops.append(obj)

        self.n_TF = blanket.params.n_TF

        super().__init__("Breeding blanket", palette=BLUE["BB"], **kwargs)

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
