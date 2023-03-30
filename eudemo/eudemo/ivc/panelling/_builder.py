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
"""Definition of panelling builder for EUDEMO."""

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.builders.tools import get_n_sectors, pattern_revolved_silhouette
from bluemira.geometry.tools import extrude_shape, make_polygon, sweep_shape
from bluemira.geometry.wire import BluemiraWire


@dataclass
class PanellingBuilderParams(ParameterFrame):
    """Parameters for :class:`.PanellingBuilder`."""

    n_TF: Parameter[int]


class PanellingBuilder(Builder):
    param_cls = PanellingBuilderParams
    params: PanellingBuilderParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame],
        panel_joints: np.ndarray,
        # build_config: Optional[Dict] = None,
    ):
        super().__init__(params, None)
        self.panel_joints = panel_joints

    def build(self) -> Component:
        xz = self.build_xz()
        return self.component_tree(
            xz=[xz],
            xy=[],
            xyz=self.build_xyz(xz.shape, degree=0),
        )

    def build_xz(self) -> PhysicalComponent:
        poly_panels = make_polygon(self.panel_joints, label="panels")
        return PhysicalComponent("panels", poly_panels)

    def build_xyz(self, xz: BluemiraWire, degree: float = 360.0) -> PhysicalComponent:
        sector_degree, n_sectors = get_n_sectors(self.params.n_TF.value, degree)
        panel = extrude_shape(xz, (0, 1, 0))
        # shapes = pattern_revolved_silhouette(xz, )


if __name__ == "__main__":
    from functools import lru_cache

    import matplotlib.pyplot as plt
    import numpy as np

    from bluemira.display import plot_2d
    from bluemira.equilibria.shapes import JohnerLCFS
    from bluemira.geometry.tools import boolean_cut, make_polygon
    from eudemo.ivc.panelling import PanellingDesigner

    def cut_wire_below_z(wire, proportion: float):
        """Cut a wire below z that is 'proportion' of the height of the wire."""
        bbox = wire.bounding_box
        z_cut_coord = proportion * (bbox.z_max - bbox.z_min) + bbox.z_min
        cutting_box = np.array(
            [
                [bbox.x_min - 1, 0, bbox.z_min - 1],
                [bbox.x_min - 1, 0, z_cut_coord],
                [bbox.x_max + 1, 0, z_cut_coord],
                [bbox.x_max + 1, 0, bbox.z_min - 1],
                [bbox.x_min - 1, 0, bbox.z_min - 1],
            ]
        )
        pieces = boolean_cut(wire, [make_polygon(cutting_box, closed=True)])
        return pieces[np.argmax([p.center_of_mass[2] for p in pieces])]

    @lru_cache(maxsize=None)
    def make_cut_johner():
        """
        Make a wall shape and cut it below a (fictional) x-point.

        As this is for testing, we just use a JohnerLCFS with a slightly
        larger radius than default, then cut it below a z-coordinate that
        might be the x-point in an equilibrium.
        """
        johner_wire = JohnerLCFS(var_dict={"r_0": {"value": 10.5}}).create_shape()
        return cut_wire_below_z(johner_wire, 1 / 4)

    designer = PanellingDesigner(
        params={
            "fw_a_max": {"value": 25, "unit": "degrees"},
            "fw_dL_min": {"value": 0.25, "unit": "m"},
        },
        wall_boundary=make_cut_johner(),
    )
    panel_coords = designer.run()

    builder = PanellingBuilder({"n_TF": {"value": 16}}, panel_coords)
