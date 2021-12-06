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

from bluemira.base.components import PhysicalComponent, Component

from bluemira.geometry.tools import revolve_shape, make_circle, offset_wire
from bluemira.geometry.wire import BluemiraWire
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import PictureFrame


class PFCoilBuilder:
    def __init__(self, coil, r_corner, tk_insulation, tk_casing):
        self.coil = coil
        self.r_corner = r_corner
        self.tk_insulation = tk_insulation
        self.tk_casing = tk_casing

    def build_xy(self):
        r_in = self.coil.x - self.coil.dx
        r_out = self.coil.x + self.coil.dx
        c1 = make_circle(r_out)
        c2 = make_circle(r_in)

        wp = PhysicalComponent("winding pack", BluemiraFace([c1, c2]))

        r_in -= self.tk_insulation
        c3 = make_circle(r_in)
        inner_ins = BluemiraFace([c2, c3])

        r_out += self.tk_insulation
        c4 = make_circle(r_out)
        outer_ins = BluemiraFace([c4, c1])

        ins = Component(
            name="ground insulation",
            children=[
                PhysicalComponent("inner", inner_ins),
                PhysicalComponent("outer", outer_ins),
            ],
        )

        r_in -= self.tk_casing
        c5 = make_circle(r_in)
        inner_cas = BluemiraFace([c3, c5])

        r_out += self.tk_casing
        c6 = make_circle(r_out)
        outer_cas = BluemiraFace([c6, c4])
        casing = Component(
            "casing",
            children=[
                PhysicalComponent("inner", inner_cas),
                PhysicalComponent("outer", outer_cas),
            ],
        )

        component = Component(self.coil.name, children=[wp, ins, casing])
        component.plot_options.plane = "xy"  # :/
        return component

    def build_xz(self):
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
        wp = PhysicalComponent("winding pack", BluemiraFace(shape))

        ins_shape = offset_wire(shape, self.tk_insulation)

        ins = PhysicalComponent("ground insulation", BluemiraFace([ins_shape, shape]))

        cas_shape = offset_wire(ins_shape, self.tk_casing)

        casing = PhysicalComponent("casing", BluemiraFace([cas_shape, ins_shape]))
        return Component(self.coil.name, children=[wp, ins, casing])

    def build_xyz(self):
        # I doubt this is floating-point safe to collisions...
        c_xz = self.build_xz()
        components = []
        for c in c_xz.children:
            shape = revolve_shape(c.shape, degree=360)
            c_xyz = PhysicalComponent(c.name, shape)
            components.append(c_xyz)

        return Component(self.coil.name, children=components)


class PFCoilsetBuilder:
    def __init__(self, coilset, r_corner, tk_insulation, tk_casing):
        self.coilset = coilset
        self.r_corner = r_corner
        self.tk_insulation = tk_insulation
        self.tk_casing = tk_casing
        self.sub_components = []
        for coil in self.coilset.coils.values():
            sub_comp = PFCoilBuilder(
                coil, self.r_corner, self.tk_insulation, self.tk_casing
            )
            self.sub_components.append(sub_comp)

    def build_xy(self):
        xy_comps = []
        for comp in self.sub_components:
            xy_comps.append(comp.build_xy())
        component = Component("PF coils", children=xy_comps)
        component.plot_options.plane = "xy"
        return component

    def build_xz(self):
        xz_comps = []
        for comp in self.sub_components:
            xz_comps.append(comp.build_xz())
        component = Component("PF coils", children=xz_comps)
        return component

    def build_xyz(self):
        xyz_comps = []
        for comp in self.sub_components:
            xyz_comps.append(comp.build_xyz())
        component = Component("PF coils", children=xyz_comps)
        return component


if __name__ == "__main__":
    from bluemira.equilibria.coils import Coil, CoilSet

    coil = Coil(4, 4, 10e6, dx=0.5, dz=0.8, name="PF_1")
    coil2 = Coil(10, 10, 1e5, dx=0.1, dz=0.5, name="PF_2")
    coilset = CoilSet([coil, coil2])
    builder = PFCoilBuilder(coil, 0.05, 0.05, 0.1)

    c_xy = builder.build_xy()
    c_xz = builder.build_xz()
    c_xyz = builder.build_xyz()

    c_xy.plot_2d()
    c_xz.plot_2d()
    c_xyz.show_cad()

    builder = PFCoilsetBuilder(coilset, 0.05, 0.05, 0.1)
    c_xy = builder.build_xy()
    c_xz = builder.build_xz()
    c_xyz = builder.build_xyz()

    c_xy.plot_2d()
    c_xz.plot_2d()
    c_xyz.show_cad()
