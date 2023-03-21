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
"""Designer for wall panelling."""

from dataclasses import dataclass
from typing import Dict, Optional, Union

import numpy as np

from bluemira.base.designer import Designer
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.geometry.wire import BluemiraWire
from bluemira.utilities.optimiser import Optimiser
from eudemo.ivc.panelling._opt_problem import PanellingOptProblem
from eudemo.ivc.panelling._paneller import Paneller


@dataclass
class PanellingDesignerParams(ParameterFrame):
    """Parameters for :class:`.PanellingDesigner`."""

    fw_a_max: Parameter[float]
    """The maximum angle of rotation between adjacent panels [degrees]."""
    fw_dL_min: Parameter[float]
    """The minimum length for an individual panel [m]."""


class PanellingDesigner(Designer[np.ndarray]):
    r"""
    Design the shape for panels of the first wall.

    The panel design's objective is to minimise the cumulative panel
    length (minimising the amount of material required), whilst
    constraining the maximum angle of rotation between adjacent panels,
    and the minimum length of a panel.

    Parameters
    ----------
    params
        The parameters for the panelling design problem. See
        :class:`.PanellingDesignerParams` for the required parameters.
    wall_boundary
        The boundary of the first wall to build the panels around. Note
        that this designer constructs panels around the *outside* of the
        given wall boundary; i.e., the panels enclose the wall.
    build_config
        Configuration options for the designer:

        * algorithm: str
            The optimisation algorithm to use (default: ``'SLSQP'``\).
        * opt_conditions: Dict[str, Union[float, int]]
            The stopping conditions for the optimiser
            (default: ``{"max_eval": 400, "ftol_rel": 1e-4}``\).

    """

    param_cls = PanellingDesignerParams
    params: PanellingDesignerParams

    def __init__(
        self,
        params: Union[Dict, PanellingDesignerParams, ParameterFrame],
        wall_boundary: BluemiraWire,
        build_config: Optional[Dict] = None,
    ):
        super().__init__(params, build_config)
        self.wall_boundary = wall_boundary

    def run(self) -> np.ndarray:
        paneller = self._make_paneller()
        optimiser = Optimiser(
            self.build_config.get("algorithm", "SLSQP"),
            opt_conditions=self.build_config.get(
                "opt_conditions", {"max_eval": 400, "ftol_rel": 1e-4}
            ),
        )
        opt_problem = PanellingOptProblem(paneller, optimiser)
        x_opt = opt_problem.optimise()
        return paneller.joints(x_opt)

    def mock(self) -> np.ndarray:
        paneller = self._make_paneller()
        return paneller.joints(paneller.x0)

    def _make_paneller(self) -> Paneller:
        """Make a :class:`.Paneller` with this designer's params."""
        boundary = self.wall_boundary.discretize(byedges=True)
        return Paneller(
            boundary.xyz[[0, 2], :],
            self.params.fw_a_max.value,
            self.params.fw_dL_min.value,
        )
