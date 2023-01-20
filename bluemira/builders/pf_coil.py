# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
from dataclasses import dataclass
from typing import Dict, List, Union

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.equilibria.coils import Coil
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import PictureFrame
from bluemira.geometry.tools import make_circle, offset_wire, revolve_shape
from bluemira.geometry.wire import BluemiraWire


@dataclass
class PFCoilBuilderParams(ParameterFrame):
    """
    Parameters for the `PFCoilBuilder` class.
    """

    tk_insulation: Parameter[float]
    tk_casing: Parameter[float]
    ctype: Parameter[str]


class PFCoilBuilder(Builder):
    """
    Builder for a single PF coil.
    """

    CASING = "Casing"
    GROUND_INSULATION = "Ground Insulation"
    INNER = "Inner"
    OUTER_INS = "Outer Ins"
    WINDING_PACK = "Winding Pack"

    param_cls = PFCoilBuilderParams

    def __init__(
        self,
        params: Union[PFCoilBuilderParams, Dict],
        build_config: Dict,
        xz_cross_section: BluemiraWire,
    ):
        super().__init__(params, build_config)
        self.xz_cross_section = xz_cross_section

    def build(self) -> Component:
        """
        Build the PFCoil component.
        """
        return self.component_tree(
            xz=self.build_xz(self.xz_cross_section),
            xy=self.build_xy(self.xz_cross_section),
            xyz=self.build_xyz(self.xz_cross_section),
        )

    def build_xy(self, shape: BluemiraWire) -> List[PhysicalComponent]:
        """
        Build the xy cross-section of the PF coil.
        """
        r_in = shape.bounding_box.x_min
        r_out = shape.bounding_box.x_max
        c1 = make_circle(r_out)
        c2 = make_circle(r_in)

        wp = PhysicalComponent(self.WINDING_PACK, BluemiraFace([c1, c2]))
        idx = 0 if self.params.ctype.value == "CS" else 1
        wp.plot_options.face_options["color"] = BLUE_PALETTE["PF"][idx]

        r_in -= self.params.tk_insulation.value
        c3 = make_circle(r_in)
        inner_ins = PhysicalComponent(self.INNER, BluemiraFace([c2, c3]))
        inner_ins.plot_options.face_options["color"] = BLUE_PALETTE["PF"][3]

        r_out += self.params.tk_insulation.value
        c4 = make_circle(r_out)
        outer_ins = PhysicalComponent(self.OUTER_INS, BluemiraFace([c4, c1]))
        outer_ins.plot_options.face_options["color"] = BLUE_PALETTE["PF"][3]

        ins = Component(name=self.GROUND_INSULATION, children=[inner_ins, outer_ins])

        r_in -= self.params.tk_casing.value
        c5 = make_circle(r_in)
        inner_cas = PhysicalComponent(self.INNER, BluemiraFace([c3, c5]))
        inner_cas.plot_options.face_options["color"] = BLUE_PALETTE["PF"][2]

        r_out += self.params.tk_casing.value
        c6 = make_circle(r_out)
        outer_cas = PhysicalComponent(self.OUTER_INS, BluemiraFace([c6, c4]))
        outer_cas.plot_options.face_options["color"] = BLUE_PALETTE["PF"][2]
        casing = Component(self.CASING, children=[inner_cas, outer_cas])

        return [wp, ins, casing]

    def build_xz(self, shape: BluemiraWire) -> List[PhysicalComponent]:
        """
        Build the xz cross-section of the PF coil.
        """
        wp = PhysicalComponent(self.WINDING_PACK, BluemiraFace(shape))
        idx = 0 if self.params.ctype.value == "CS" else 1
        wp.plot_options.face_options["color"] = BLUE_PALETTE["PF"][idx]

        ins_shape = offset_wire(shape, self.params.tk_insulation.value)
        ins = PhysicalComponent(self.GROUND_INSULATION, BluemiraFace([ins_shape, shape]))
        ins.plot_options.face_options["color"] = BLUE_PALETTE["PF"][3]
        cas_shape = offset_wire(ins_shape, self.params.tk_casing.value)

        casing = PhysicalComponent(self.CASING, BluemiraFace([cas_shape, ins_shape]))
        casing.plot_options.face_options["color"] = BLUE_PALETTE["PF"][2]
        return [wp, ins, casing]

    def build_xyz(
        self, shape: BluemiraWire, degree: float = 360.0
    ) -> List[PhysicalComponent]:
        """
        Build the xyz representation of the PF coil.

        Parameters
        ----------
        shape: BluemiraWire
            The xz cross-section shape of the coil.
        degree: float
            The angle [°] around which to build the components, by default 360.0.

        Returns
        -------
        component: Component
            The component grouping the results in 3D (xyz).
        """
        # I doubt this is floating-point safe to collisions...
        xz_components = self.build_xz(shape)
        components = []
        for c in xz_components:
            shape = revolve_shape(c.shape, degree=degree)
            c_xyz = PhysicalComponent(c.name, shape)
            c_xyz.display_cad_options.color = c.plot_options.face_options["color"]
            components.append(c_xyz)

        return components


@dataclass
class PFCoilPictureFrameParams(ParameterFrame):
    """
    Parameters for the `PFCoilPictureFrame` designer.
    """

    r_corner: Parameter[float]


class PFCoilPictureFrame(Designer):
    """
    Designer for the shape of a PF coil in the xz plane using a
    PictureFrame parameterisation.
    """

    param_cls = PFCoilPictureFrameParams

    def __init__(self, params: Union[PFCoilPictureFrameParams, Dict], coil: Coil):
        super().__init__(params)
        self.coil = coil

    def run(self) -> BluemiraWire:
        """
        Run the design step, outputting the PictureFrame shape as a wire.
        """
        x_in = self.coil.x - self.coil.dx
        x_out = self.coil.x + self.coil.dx
        z_up = self.coil.z + self.coil.dz
        z_down = self.coil.z - self.coil.dz
        return PictureFrame(
            {
                "x1": {"value": x_in, "fixed": True},
                "x2": {"value": x_out, "fixed": True},
                "z1": {"value": z_up, "fixed": True},
                "z2": {"value": z_down, "fixed": True},
                "ri": {"value": self.params.r_corner.value, "fixed": True},
                "ro": {"value": self.params.r_corner.value, "fixed": True},
            }
        ).create_shape()
