# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Plasma builder.
"""

from dataclasses import dataclass

from bluemira.base.builder import Builder, ComponentManager
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.builders.tools import apply_component_display_options, get_n_sectors
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_circle, revolve_shape
from bluemira.geometry.wire import BluemiraWire


class Plasma(ComponentManager):
    """
    Wrapper around a plasma component tree.
    """

    def lcfs(self) -> BluemiraWire:
        """Return a wire representing the last-closed flux surface."""
        return (
            self.component()
            .get_component("xz")
            .get_component(PlasmaBuilder.LCFS)
            .shape.boundary[0]
        )


@dataclass
class PlasmaBuilderParams(ParameterFrame):
    """
    Plasma builder parameters
    """

    n_TF: Parameter[int]


class PlasmaBuilder(Builder):
    """
    Builder for a poloidally symmetric plasma.
    """

    LCFS = "LCFS"

    param_cls: type[PlasmaBuilderParams] = PlasmaBuilderParams

    def __init__(
        self,
        params: ParameterFrame | dict,
        build_config: dict,
        xz_lcfs: BluemiraWire,
    ):
        super().__init__(params, build_config)
        self.xz_lcfs = xz_lcfs

    def build(self) -> Component:
        """
        Build the plasma component.
        """
        return self.component_tree(
            xz=[self.build_xz(self.xz_lcfs)],
            xy=[self.build_xy(self.xz_lcfs)],
            xyz=[self.build_xyz(self.xz_lcfs, degree=0)],
        )

    def build_xz(self, lcfs: BluemiraWire) -> PhysicalComponent:
        """
        Build the x-z components of the plasma.

        Parameters
        ----------
        lcfs:
            LCFS wire
        """
        face = BluemiraFace(lcfs, self.name)
        component = PhysicalComponent(self.LCFS, face)
        apply_component_display_options(
            component, color=BLUE_PALETTE["PL"], transparency=0.3
        )
        return component

    def build_xy(self, lcfs: BluemiraWire) -> PhysicalComponent:
        """
        Build the x-y components of the plasma.

        Parameters
        ----------
        lcfs:
            LCFS wire
        """
        inner = make_circle(lcfs.bounding_box.x_min)
        outer = make_circle(lcfs.bounding_box.x_max)
        face = BluemiraFace([outer, inner], self.name)
        component = PhysicalComponent(self.LCFS, face)
        apply_component_display_options(
            component, color=BLUE_PALETTE["PL"], transparency=0.3
        )
        return component

    def build_xyz(self, lcfs: BluemiraWire, degree: float = 360.0) -> PhysicalComponent:
        """
        Build the x-y-z components of the plasma.

        Parameters
        ----------
        lcfs:
            LCFS wire
        degree:
            degrees to sweep the shape
        """
        sector_degree, n_sectors = get_n_sectors(self.params.n_TF.value, degree)

        solid = revolve_shape(
            BluemiraFace(lcfs),
            direction=(0, 0, 1),
            degree=sector_degree * n_sectors,
            label=self.LCFS,
        )
        component = PhysicalComponent(self.LCFS, solid)
        apply_component_display_options(
            component, color=BLUE_PALETTE["PL"], transparency=0.3
        )
        return component
