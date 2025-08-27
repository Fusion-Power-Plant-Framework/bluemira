# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Builder for the PF coils
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.builders.tools import apply_component_display_options, get_n_sectors
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.equilibria.coils import Coil, CoilType
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import PictureFrame
from bluemira.geometry.tools import make_circle, offset_wire, revolve_shape

if TYPE_CHECKING:
    from bluemira.base.builder import BuildConfig
    from bluemira.base.parameter_frame.typed import ParameterFrameLike
    from bluemira.geometry.wire import BluemiraWire


@dataclass
class PFCoilBuilderParams(ParameterFrame):
    """
    Parameters for the `PFCoilBuilder` class.
    """

    n_TF: Parameter[int]

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

    param_cls: type[PFCoilBuilderParams] = PFCoilBuilderParams
    params: PFCoilBuilderParams

    def __init__(
        self,
        params: ParameterFrameLike,
        build_config: BuildConfig,
        xz_cross_section: BluemiraWire,
    ):
        super().__init__(params, build_config, verbose=False)
        self.xz_cross_section = xz_cross_section

    def build(self) -> Component:
        """
        Build the PFCoil component.
        """  # noqa: DOC201
        return self.component_tree(
            xz=self.build_xz(self.xz_cross_section),
            xy=self.build_xy(self.xz_cross_section),
            xyz=self.build_xyz(self.xz_cross_section, degree=0),
        )

    def build_xy(self, shape: BluemiraWire) -> list[PhysicalComponent]:
        """
        Build the xy cross-section of the PF coil.

        Returns
        -------
        :
            the winding pack, insulation and casing
        """
        r_in = shape.bounding_box.x_min
        r_out = shape.bounding_box.x_max
        c1 = make_circle(r_out)
        c2 = make_circle(r_in)

        wp = PhysicalComponent(self.WINDING_PACK, BluemiraFace([c1, c2]))

        idx = CoilType(self.params.ctype.value).value - 1
        apply_component_display_options(wp, color=BLUE_PALETTE["PF"][idx])

        r_in -= self.params.tk_insulation.value
        c3 = make_circle(r_in)
        inner_ins = PhysicalComponent(self.INNER, BluemiraFace([c2, c3]))
        apply_component_display_options(inner_ins, color=BLUE_PALETTE["PF"][3])

        r_out += self.params.tk_insulation.value
        c4 = make_circle(r_out)
        outer_ins = PhysicalComponent(self.OUTER_INS, BluemiraFace([c4, c1]))
        apply_component_display_options(outer_ins, color=BLUE_PALETTE["PF"][3])

        ins = Component(name=self.GROUND_INSULATION, children=[inner_ins, outer_ins])

        r_in -= self.params.tk_casing.value
        c5 = make_circle(r_in)
        inner_cas = PhysicalComponent(self.INNER, BluemiraFace([c3, c5]))
        apply_component_display_options(inner_cas, color=BLUE_PALETTE["PF"][2])

        r_out += self.params.tk_casing.value
        c6 = make_circle(r_out)
        outer_cas = PhysicalComponent(self.OUTER_INS, BluemiraFace([c6, c4]))
        apply_component_display_options(outer_cas, color=BLUE_PALETTE["PF"][2])

        casing = Component(self.CASING, children=[inner_cas, outer_cas])

        return [wp, ins, casing]

    def build_xz(self, shape: BluemiraWire) -> list[PhysicalComponent]:
        """
        Build the xz cross-section of the PF coil.

        Returns
        -------
        :
            the winding pack, insulation and casing
        """
        wp = PhysicalComponent(
            self.WINDING_PACK,
            BluemiraFace(shape),
            material=self.get_material(self.WINDING_PACK),
        )
        idx = CoilType(self.params.ctype.value).value - 1
        apply_component_display_options(wp, color=BLUE_PALETTE["PF"][idx])

        ins_shape = offset_wire(shape, self.params.tk_insulation.value)
        ins = PhysicalComponent(
            self.GROUND_INSULATION,
            BluemiraFace([ins_shape, shape]),
            material=self.get_material(self.GROUND_INSULATION),
        )
        apply_component_display_options(ins, color=BLUE_PALETTE["PF"][3])

        cas_shape = offset_wire(ins_shape, self.params.tk_casing.value)
        casing = PhysicalComponent(
            self.CASING,
            BluemiraFace([cas_shape, ins_shape]),
            material=self.get_material(self.CASING),
        )
        apply_component_display_options(casing, color=BLUE_PALETTE["PF"][2])
        return [wp, ins, casing]

    def build_xyz(
        self, shape: BluemiraWire, degree: float = 360.0
    ) -> list[PhysicalComponent]:
        """
        Build the xyz representation of the PF coil.

        Parameters
        ----------
        shape:
            The xz cross-section shape of the coil.
        degree:
            The angle [°] around which to build the components, by default 360.0.

        Returns
        -------
        :
            The component grouping the results in 3D (xyz).
        """
        sector_degree, n_sectors = get_n_sectors(self.params.n_TF.value, degree)

        # I doubt this is floating-point safe to collisions...
        xz_components = self.build_xz(shape)
        components = []
        for c in xz_components:
            shape = revolve_shape(c.shape, degree=sector_degree * n_sectors)
            c_xyz = PhysicalComponent(c.name, shape, material=c.material)
            apply_component_display_options(
                c_xyz, color=c.plot_options.face_options["color"]
            )
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

    param_cls: type[PFCoilPictureFrameParams] = PFCoilPictureFrameParams
    params: PFCoilPictureFrameParams

    def __init__(self, params: ParameterFrameLike, coil: Coil):
        super().__init__(params, verbose=False)
        self.coil = coil

    def run(self) -> BluemiraWire:
        """
        Run the design step

        Returns
        -------
        :
            The PictureFrame shape as a wire.
        """
        x_in = self.coil.x - self.coil.dx
        x_out = self.coil.x + self.coil.dx
        z_up = self.coil.z + self.coil.dz
        z_down = self.coil.z - self.coil.dz
        return PictureFrame({
            "x1": {"value": x_in, "fixed": True},
            "x2": {"value": x_out, "fixed": True},
            "z1": {"value": z_up, "fixed": True},
            "z2": {"value": z_down, "fixed": True},
            "ri": {"value": self.params.r_corner.value, "fixed": True},
            "ro": {"value": self.params.r_corner.value, "fixed": True},
        }).create_shape()
