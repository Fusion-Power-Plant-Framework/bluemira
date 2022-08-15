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
from typing import Dict, List, Tuple, Type, Union

import numpy as np

from bluemira.base.builder import Builder, ComponentManager
from bluemira.base.components import PhysicalComponent
from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import NewParameter as Parameter
from bluemira.base.parameter_frame import NewParameterFrame as ParameterFrame
from bluemira.base.parameter_frame import parameter_frame
from bluemira.builders.tools import circular_pattern_component, get_n_sectors
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_circle, make_polygon, revolve_shape


class Cryostat(ComponentManager):
    """
    Wrapper around a cryostat component tree.
    """


@parameter_frame
class CryostatDesignerParams:
    """
    Cryostat designer parameters
    """

    g_cr_ts: Parameter[float]


@parameter_frame
class CryostatBuilderParams:
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

    def run(self) -> Tuple[float]:
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

    def build(self) -> Cryostat:
        """
        Build the cryostat component.
        """
        x_out, z_top = self.designer.run()
        xz_cryostat = self.build_xz(x_out, z_top)
        xz_cross_section = xz_cryostat.get_component_properties("shape")

        return Cryostat(
            self.component_tree(
                xz=[xz_cryostat],
                xy=[self.build_xy(x_out)],
                xyz=self.build_xyz(xz_cross_section),
            )
        )

    def build_xz(self, x_out: float, z_top: float) -> PhysicalComponent:
        """
        Build the x-z components of the cryostat.

        Parameters
        ----------
        x_out: float
            x coordinate extremity
        z_top: float
            z coordinate extremity
        """
        x_in = 0
        x_gs_kink = self.params.x_g_support.value - self.params.x_gs_kink_diff.value
        z_mid = self.params.z_gs.value - self.params.g_cr_ts.value
        z_bot = z_mid - self.params.well_depth.value
        tk = self.params.tk_cr_vv.value

        x_inner = [x_in, x_out, x_out, x_gs_kink, x_gs_kink, x_in]
        z_inner = [z_top, z_top, z_mid, z_mid, z_bot, z_bot]

        x_outer = np.array([x_in, x_gs_kink, x_gs_kink, x_out, x_out, x_in])
        x_outer[1:-1] += tk

        z_outer = np.array([z_bot, z_bot, z_mid, z_mid, z_top, z_top])
        z_outer[:4] -= tk
        z_outer[4:] += tk

        x = np.concatenate([x_inner, x_outer])
        z = np.concatenate([z_inner, z_outer])

        xz_cross_section = BluemiraFace(
            make_polygon({"x": x, "y": 0, "z": z}, closed=True)
        )

        cryostat_vv = PhysicalComponent(self.CRYO, xz_cross_section)
        cryostat_vv.plot_options.face_options["color"] = BLUE_PALETTE["CR"][0]
        return cryostat_vv

    def build_xy(self, x_out: float) -> PhysicalComponent:
        """
        Build the x-y components of the cryostat.

        Parameters
        ----------
        x_out: float
            x coordinate extremity
        """
        r_out = x_out + self.params.tk_cr_vv.value
        inner = make_circle(radius=x_out)
        outer = make_circle(radius=r_out)

        cryostat_vv = PhysicalComponent(self.CRYO, BluemiraFace([outer, inner]))
        cryostat_vv.plot_options.face_options["color"] = BLUE_PALETTE["CR"][0]
        return cryostat_vv

    def build_xyz(
        self, xz_cross_section: BluemiraFace, degree=360
    ) -> List[PhysicalComponent]:
        """
        Build the x-y-z components of the cryostat.

        Parameters
        ----------
        xz_cross_section: BluemiraFace
            xz cross section of cryostat
        """
        sector_degree, n_sectors = get_n_sectors(self.params.n_TF.value, degree)

        shape = revolve_shape(
            xz_cross_section,
            base=(0, 0, 0),
            direction=(0, 0, 1),
            degree=sector_degree,
        )

        cryostat_vv = PhysicalComponent(self.CRYO, shape)
        cryostat_vv.display_cad_options.color = BLUE_PALETTE["CR"][0]
        return circular_pattern_component(
            cryostat_vv, n_sectors, degree=sector_degree * n_sectors
        )
