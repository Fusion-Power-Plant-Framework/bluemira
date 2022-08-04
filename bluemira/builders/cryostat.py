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
Cryostat builder
"""
from typing import Dict, List, Type, Union

import numpy as np

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.designer import Designer
from bluemira.base.parameter import Parameter, ParameterFrame
from bluemira.builders.tools import circular_pattern_component
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_circle, make_polygon, revolve_shape


class Cryostat:
    """
    Cryostat Component Manager TODO
    """

    def __init__(self, component: Component):
        super().__init__()
        self._component = component

    def component(self) -> Component:
        """
        Return component
        """
        return self._component


class CryostatDesignerParams(ParameterFrame):
    """
    Cryostat designer parameters
    """

    x_g_support: Parameter[float]
    x_gs_kink_diff: Parameter[float]  # TODO add to Parameter default = 2
    g_cr_ts: Parameter[float]
    tk_cr_vv: Parameter[float]
    well_depth: Parameter[float]  # TODO add to Parameter default = 5 chickens
    z_gs: Parameter[
        float
    ]  # TODO add to Parameter default (z gravity support) = -15 chickens


class CryostatBuilderParams(ParameterFrame):
    """
    Cryostat builder parameters
    """

    n_TF: Parameter[int]


class CryostatDesigner(Designer[BluemiraFace]):
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

    def run(self) -> tuple[BluemiraFace]:
        """
        Cryostat designer run method
        """
        return self.run_xz(), self.run_xy()

    def run_xz(self) -> BluemiraFace:
        """
        Creates minimal xz for cryostat
        """
        x_in = 0
        x_out, z_top = self._get_extrema()

        x_gs_kink = self.params.x_g_support - self.params.x_gs_kink_diff
        z_mid = self.params.z_gs - self.params.g_cr_ts
        z_bot = z_mid - self.params.well_depth
        tk = self.params.tk_cr_vv.value

        x_inner = [x_in, x_out, x_out, x_gs_kink, x_gs_kink, x_in]
        z_inner = [z_top, z_top, z_mid, z_mid, z_bot, z_bot]

        x_outer = [x_in, x_gs_kink, x_gs_kink, x_out, x_out, x_in]
        x_outer[1:-1] += tk

        z_outer = [z_bot, z_bot, z_mid, z_mid, z_top, z_top]
        z_outer[:4] -= tk
        z_outer[4:] += tk

        x = np.concatenate([x_inner, x_outer])
        z = np.concatenate([z_inner, z_outer])

        return BluemiraFace(make_polygon({"x": x, "y": 0, "z": z}, closed=True))

    def run_xy(self) -> BluemiraFace:
        """
        Creates minimal xy for cryostat
        """
        r_in, _ = self._get_extrema()
        r_out = r_in + self.params.tk_cr_vv
        inner = make_circle(radius=r_in)
        outer = make_circle(radius=r_out)

        return BluemiraFace([outer, inner])

    def _get_extrema(self) -> tuple[float]:
        bound_box = self.cryo_ts_xz.bounding_box
        z_max = bound_box.z_max
        x_max = bound_box.x_max
        x_out = x_max + self.params.g_cr_ts
        z_top = z_max + self.params.g_cr_ts
        return x_out, z_top


class CryostatBuilder(Builder):
    """
    Builder for the cryostat
    """

    CRYO = "Cryostat VV"
    param_cls: Type[CryostatBuilderParams] = CryostatBuilderParams

    def build(self) -> Cryostat:
        """
        Build the cryostat component.
        """
        component = super().build()

        self._xz_cross_section, self._xy_cross_section = self.designer.run()

        component.add_child(Component("xz", children=[self.build_xz()]))
        component.add_child(Component("xy", children=[self.build_xy()]))
        component.add_child(Component("xyz", children=self.build_xyz()))
        return Cryostat(component)

    def build_xz(self) -> PhysicalComponent:
        """
        Build the x-z components of the cryostat.
        """
        cryostat_vv = PhysicalComponent(self.CRYO, self._xz_cross_section)
        cryostat_vv.plot_options.face_options["color"] = BLUE_PALETTE["CR"][0]
        return cryostat_vv

    def build_xy(self) -> PhysicalComponent:
        """
        Build the x-y components of the cryostat.
        """
        cryostat_vv = PhysicalComponent(self.CRYO, self._xy_cross_section)
        cryostat_vv.plot_options.face_options["color"] = BLUE_PALETTE["CR"][0]
        return cryostat_vv

    def build_xyz(self, degree=360) -> List[PhysicalComponent]:
        """
        Build the x-y-z components of the cryostat.
        """
        sector_degree = 360 / self.params.n_TF.value
        n_sectors = max(1, int(degree // int(sector_degree)))

        shape = revolve_shape(
            self._xz_cross_section,
            base=(0, 0, 0),
            direction=(0, 0, 1),
            degree=sector_degree,
        )

        cryostat_vv = PhysicalComponent(self.CRYO, shape)
        cryostat_vv.display_cad_options.color = BLUE_PALETTE["CR"][0]
        return circular_pattern_component(
            cryostat_vv, n_sectors, degree=sector_degree * n_sectors
        )
