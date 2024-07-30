# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Cryostat builder
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from bluemira.base.builder import BuildConfig, Builder
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

if TYPE_CHECKING:
    from bluemira.base.parameter_frame.typing import ParameterFrameLike


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


class CryostatDesigner(Designer[tuple[float, float]]):
    """
    Designer for the cryostat
    """

    param_cls: type[CryostatDesignerParams] = CryostatDesignerParams

    def __init__(
        self,
        params: ParameterFrameLike,
        cryo_ts_xz: BluemiraFace,
    ):
        super().__init__(params)
        self.cryo_ts_xz = cryo_ts_xz

    def run(self) -> tuple[float, float]:
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
    params: CryostatBuilderParams
    param_cls: type[CryostatBuilderParams] = CryostatBuilderParams

    def __init__(
        self,
        params: ParameterFrameLike,
        build_config: BuildConfig,
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

        Raises
        ------
        ValueError
            Only internal kink supported

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
    ) -> list[PhysicalComponent]:
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
