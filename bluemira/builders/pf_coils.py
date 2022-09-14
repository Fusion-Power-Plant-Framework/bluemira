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
Builder for the PF coils
"""

from bluemira.base.components import Component, PhysicalComponent
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.parameterisations import PictureFrame
from bluemira.geometry.tools import make_circle, make_face, offset_wire, revolve_shape


class PFCoilBuilder:
    """
    Builder for a single PF coil
    """

    def __init__(self, coil, r_corner, tk_insulation, tk_casing, ctype):
        self.coil = coil
        self.r_corner = r_corner
        self.tk_insulation = tk_insulation
        self.tk_casing = tk_casing
        self.ctype = ctype

    def build_xy(self):
        """
        Build the x-y representation of a PF coil.
        """
        r_in = self.coil.x - self.coil.dx
        r_out = self.coil.x + self.coil.dx
        c1 = make_circle(r_out)
        c2 = make_circle(r_in)

        wp = PhysicalComponent("winding pack", make_face([c1, c2]))
        idx = 0 if self.ctype == "CS" else 1
        wp.plot_options.face_options["color"] = BLUE_PALETTE["PF"][idx]

        r_in -= self.tk_insulation
        c3 = make_circle(r_in)
        inner_ins = PhysicalComponent("inner", make_face([c2, c3]))
        inner_ins.plot_options.face_options["color"] = BLUE_PALETTE["PF"][3]

        r_out += self.tk_insulation
        c4 = make_circle(r_out)
        outer_ins = PhysicalComponent("outer_ins", make_face([c4, c1]))
        outer_ins.plot_options.face_options["color"] = BLUE_PALETTE["PF"][3]

        ins = Component(
            name="ground insulation",
            children=[inner_ins, outer_ins],
        )

        r_in -= self.tk_casing
        c5 = make_circle(r_in)
        inner_cas = PhysicalComponent("inner", make_face([c3, c5]))
        inner_cas.plot_options.face_options["color"] = BLUE_PALETTE["PF"][2]

        r_out += self.tk_casing
        c6 = make_circle(r_out)
        outer_cas = PhysicalComponent("outer", make_face([c6, c4]))
        outer_cas.plot_options.face_options["color"] = BLUE_PALETTE["PF"][2]
        casing = Component(
            "casing",
            children=[inner_cas, outer_cas],
        )

        component = Component(self.coil.name, children=[wp, ins, casing])
        return component

    def build_xz(self):
        """
        Build the x-z representation of a PF coil.
        """
        x_in = self.coil.x - self.coil.dx
        x_out = self.coil.x + self.coil.dx
        z_up = self.coil.z + self.coil.dz
        z_down = self.coil.z - self.coil.dz

        shape = PictureFrame(
            {
                "x1": {"value": x_in, "fixed": True},
                "x2": {"value": x_out, "fixed": True},
                "z1": {"value": z_up, "fixed": True},
                "z2": {"value": z_down, "fixed": True},
                "ri": {"value": self.r_corner, "fixed": True},
                "ro": {"value": self.r_corner, "fixed": True},
            }
        ).create_shape()
        wp = PhysicalComponent("winding pack", make_face(shape))
        idx = 0 if self.ctype == "CS" else 1
        wp.plot_options.face_options["color"] = BLUE_PALETTE["PF"][idx]

        ins_shape = offset_wire(shape, self.tk_insulation)

        ins = PhysicalComponent("ground insulation", make_face([ins_shape, shape]))
        ins.plot_options.face_options["color"] = BLUE_PALETTE["PF"][3]
        cas_shape = offset_wire(ins_shape, self.tk_casing)

        casing = PhysicalComponent("casing", make_face([cas_shape, ins_shape]))
        casing.plot_options.face_options["color"] = BLUE_PALETTE["PF"][2]
        return Component(self.coil.name, children=[wp, ins, casing])

    def build_xyz(self, degree: float = 360.0):
        """
        Build the x-y-z representation of a PF coil.

        Parameters
        ----------
        degree: float
            The angle [°] around which to build the components, by default 360.0.

        Returns
        -------
        component: Component
            The component grouping the results in 3D (xyz).
        """
        # I doubt this is floating-point safe to collisions...
        c_xz = self.build_xz()
        components = []
        c: PhysicalComponent
        for c in c_xz.children:
            shape = revolve_shape(c.shape, degree=degree)
            c_xyz = PhysicalComponent(c.name, shape)
            c_xyz.display_cad_options.color = c.plot_options.face_options["color"]
            components.append(c_xyz)

        return Component(self.coil.name, children=components)
