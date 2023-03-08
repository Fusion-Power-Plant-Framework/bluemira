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

# from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.plane import BluemiraPlane, BluemiraFace
from bluemira.geometry.tools import make_polygon, slice_shape
from bluemira.utilities.optimiser import Optimiser
from eudemo.maintenance.upper_port import UpperPortOP
from eudemo.equatorial_port import EquatorialPortDesigner
import math
import pytest


class TestUpperPortOP:
    def test_dummy_blanket_port_opt(self):
        params = {
            "c_rm": {"value": 0.02, "unit": "m"},
            "R_0": {"value": 9, "unit": "m"},
            "bb_min_angle": {"value": 70, "unit": "degrees"},
            "tk_bb_ib": {"value": 0.8, "unit": "m"},
            "tk_bb_ob": {"value": 1.1, "unit": "m"},
        }
        bb = make_polygon(
            {
                "x": [5, 6, 6, 11, 11, 12, 12, 5],
                "y": 0,
                "z": [-5, -5, 5, 5, -5, -5, 6, 6],
            },
            closed=True,
        )
        bb = BluemiraFace(bb)
        optimiser = Optimiser(
            "SLSQP", opt_conditions={"max_eval": 1000, "ftol_rel": 1e-8}
        )

        design_problem = UpperPortOP(params, optimiser, bb)

        solution = design_problem.optimise()

        assert design_problem.opt.check_constraints(solution)


class TestEquatorialPortDesigner:
    def setup_method(self) -> None:
        params = {
            "ep_height": {"value": 0, "unit": "m"},
        }

        self.designer = EquatorialPortDesigner(params, None, 0, 0)

    @pytest.mark.parametrize(
        "xi, xo, zh", zip([2.0, 3.0, 1.0], [9.0, 9.0, 4.0], [5.0, 4.0, 2.0])
    )
    def test_ep_designer(self, xi, xo, zh):
        param_values = {"ep_height": zh}
        self.designer.params.update_values(param_values)
        self.designer.x_ib = xi
        self.designer.x_ob = xo
        output = self.designer.execute()

        x = (xi, xo, xo, xi)
        z = (-zh / 2, -zh / 2, zh / 2, zh / 2)
        expectation = BluemiraFace(make_polygon({"x": x, "y": 0, "z": z}, closed=True))
        
        assert math.isclose(output.length, expectation.length)
        assert math.isclose(output.area, expectation.area)
        assert math.isclose(output.volume, expectation.volume)
