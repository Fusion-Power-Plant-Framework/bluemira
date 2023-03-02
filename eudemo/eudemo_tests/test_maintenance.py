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

import numpy as np
import pytest

from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import make_polygon
from bluemira.utilities.optimiser import Optimiser
from eudemo.maintenance.lower_port import LowerPortDesigner
from eudemo.maintenance.upper_port import UpperPortOP


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
            Coordinates(
                {
                    "x": [5, 6, 6, 11, 11, 12, 12, 5],
                    "y": 0,
                    "z": [-5, -5, 5, 5, -5, -5, 6, 6],
                }
            ),
            closed=True,
        )
        bb = BluemiraFace(bb)
        optimiser = Optimiser(
            "SLSQP", opt_conditions={"max_eval": 1000, "ftol_rel": 1e-8}
        )

        design_problem = UpperPortOP(params, optimiser, bb)

        solution = design_problem.optimise()

        assert design_problem.opt.check_constraints(solution)


class TestLowerPortDesigner:
    def setup_method(self):
        params = {
            "lower_port_angle": {"value": 0, "unit": "degrees"},
            "divertor_padding": {"value": 0, "unit": "m"},
        }
        divertor_shaped_box = make_polygon(
            Coordinates(
                {
                    "x": [0, 1, 1, 0],
                    "z": [1, 1, 0, 0],
                }
            ),
            closed=True,
        )
        self.designer = LowerPortDesigner(
            params,
            {},
            BluemiraFace(divertor_shaped_box),
            x_wall_inner=10,
            x_wall_outer=20,
            x_extrema=30,
        )

    @pytest.mark.parametrize("angle", [-90, 90, 45.1])
    def test_lower_port_angle_ValueError(self, angle):
        self.designer.params.lower_port_angle.value = angle

        with pytest.raises(ValueError):
            self.designer.execute()

    @pytest.mark.parametrize("angle, z_max", zip([-45, 45, 0, 33], [10, -10, 0, -6.494]))
    def test_lower_port_angle(self, angle, z_max):
        self.designer.params.lower_port_angle.value = angle

        _, traj = self.designer.execute()

        arr = traj.discretize(ndiscr=99)

        np.testing.assert_allclose(arr[2][-1], z_max, rtol=2e-5)
        np.testing.assert_allclose(arr[0][-1], self.designer.x_extrema)

        assert any(arr[0] >= self.designer.x_wall_outer)
        np.testing.assert_allclose(
            arr[2][np.where(arr[0] >= self.designer.x_wall_outer)],
            arr[2][-1],
        )
        assert any(arr[0] <= self.designer.x_wall_inner)
        np.testing.assert_allclose(
            arr[2][np.where(arr[0] <= self.designer.x_wall_inner)],
            arr[2][0],
        )

    def test_lower_port_padding(self):
        port, _ = self.designer.execute()

        self.designer.params.divertor_padding.value = 2

        port2, _ = self.designer.execute()

        assert np.isclose(port.length, port2.length - 8)
        assert np.isclose(
            port.area,
            port2.area
            - 4
            * (self.designer.x_extrema - self.designer.divertor_xz.center_of_mass[0]),
        )
