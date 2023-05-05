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
Cryostat builder
"""
from dataclasses import dataclass
from typing import Dict, List, Tuple, Type, Union

import numpy as np

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.builders.tools import (
    apply_component_display_options,
    build_sectioned_xyz,
    make_circular_xy_ring,
)
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_polygon


@dataclass
class CryostatDesignerParams(ParameterFrame):
    """
    Cryostat designer parameters
    """

    g_cr_ts: Parameter[float]


@dataclass
class CryostatBuilderParams(ParameterFrame):
    """
    Cryostat builder parameters
    """

    g_cr_ts: Parameter[float]
    n_TF: Parameter[int]
    tk_cr_vv: Parameter[float]
    # TODO add to Parameter default = 5 chickens
    well_depth: Parameter[float]
    x_g_support: Parameter[float]
    # TODO add to Parameter default = 2
    x_gs_kink_diff: Parameter[float]
    # TODO add to Parameter default (z gravity support) = -15 chickens
    z_gs: Parameter[float]


class CryostatDesigner(Designer[Tuple[float, float]]):
    """
    Designer for the cryostat
    """

    param_cls: Type[CryostatDesignerParams] = CryostatDesignerParams

    def __init__(
        self,
        params: Union[ParameterFrame, Dict],
        cryo_ts_xz: BluemiraFace,
    ):
        super().__init__(params)
        self.cryo_ts_xz = cryo_ts_xz

    def run(self) -> Tuple[float, float]:
        """
        Cryostat designer run method
        """
        bound_box = self.cryo_ts_xz.bounding_box
        z_max = bound_box.z_max
        x_max = bound_box.x_max
        x_out = x_max + self.params.g_cr_ts.value
        z_top = z_max + self.params.g_cr_ts.value
        return x_out, z_top


class CryostatBuilder(Builder):
    """
    Builder for the cryostat
    """

    CRYO = "Cryostat VV"
    param_cls: Type[CryostatBuilderParams] = CryostatBuilderParams

    def __init__(
        self,
        params: Union[ParameterFrame, Dict, None],
        build_config: Dict,
        x_out: float,
        z_top: float,
    ):
        super().__init__(params, build_config)
        self.x_out = x_out
        self.z_top = z_top

    def build(self) -> Component:
        """
        Build the cryostat component.
        """
        xz_cryostat = self.build_xz(self.x_out, self.z_top)
        xz_cross_section: BluemiraFace = xz_cryostat.get_component_properties("shape")
        return self.component_tree(
            xz=[xz_cryostat],
            xy=[self.build_xy(self.x_out)],
            xyz=self.build_xyz(xz_cross_section, degree=0),
        )

    def build_xz(self, x_out: float, z_top: float) -> PhysicalComponent:
        """
        Build the x-z components of the cryostat.

        Parameters
        ----------
        x_out:
            x coordinate extremity
        z_top:
            z coordinate extremity

        Notes
        -----
        Only designed for an inward kink, outward kinks will fail
        """
        x_in = 0
        x_gs_kink = self.params.x_g_support.value - self.params.x_gs_kink_diff.value
        if x_gs_kink > x_out:
            raise ValueError(
                "Outward kinks not supported x_g_support-x_gs_kink_diff > x_out"
            )
        z_mid = self.params.z_gs.value - self.params.g_cr_ts.value
        z_bot = z_mid - self.params.well_depth.value
        tk = self.params.tk_cr_vv.value

        x_inner = np.array([x_in, x_out, x_out, x_gs_kink, x_gs_kink, x_in])
        z_inner = np.array([z_top, z_top, z_mid, z_mid, z_bot, z_bot])

        x_outer = np.array([x_in, x_gs_kink, x_gs_kink, x_out, x_out, x_in])
        x_outer[1:-1] += tk

        z_outer = np.array([z_bot, z_bot, z_mid, z_mid, z_top, z_top])
        z_outer[:4] -= tk
        z_outer[4:] += tk

        x = np.concatenate([x_inner, x_outer])
        z = np.concatenate([z_inner, z_outer])

        cryostat_vv = PhysicalComponent(
            self.CRYO, BluemiraFace(make_polygon({"x": x, "y": 0, "z": z}, closed=True))
        )
        apply_component_display_options(cryostat_vv, color=BLUE_PALETTE["CR"][0])
        return cryostat_vv

    def build_xy(self, x_out: float) -> PhysicalComponent:
        """
        Build the x-y components of the cryostat.

        Parameters
        ----------
        x_out:
            x coordinate extremity
        """
        cryostat_vv = PhysicalComponent(
            self.CRYO, make_circular_xy_ring(x_out, x_out + self.params.tk_cr_vv.value)
        )
        apply_component_display_options(cryostat_vv, color=BLUE_PALETTE["CR"][0])
        return cryostat_vv

    def build_xyz(
        self, xz_cross_section: BluemiraFace, degree=360
    ) -> List[PhysicalComponent]:
        """
        Build the x-y-z components of the cryostat.

        Parameters
        ----------
        xz_cross_section:
            xz cross section of cryostat
        degree:
            Revolution degree
        """
        return build_sectioned_xyz(
            xz_cross_section,
            self.CRYO,
            self.params.n_TF.value,
            BLUE_PALETTE["CR"][0],
            degree,
        )
