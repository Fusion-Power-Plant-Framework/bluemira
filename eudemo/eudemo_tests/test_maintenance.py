# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short, I. Chiang
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
Tests for EU-DEMO Maintenance
"""
import math

import pytest

from bluemira.geometry.face import BluemiraFace, BluemiraWire
from bluemira.geometry.tools import make_polygon
from bluemira.utilities.optimiser import Optimiser
from eudemo.maintenance.equatorial_port import (
    CastellationBuilder,
    EquatorialPortBuilder,
    EquatorialPortDesigner,
)
from eudemo.maintenance.upper_port import UpperPortOP


class TestUpperPortOP:
    """Test Upper Port"""

    def test_dummy_blanket_port_opt(self):
        """Test Upper Port Optimiser"""
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
    """Tests the Equatorial Port Designer"""

    def setup_method(self) -> None:
        """Set-up Equatorial Port Designer"""
        params = {
            "ep_height": {"value": 0, "unit": "m"},
        }

        self.designer = EquatorialPortDesigner(params, None, 0, 0, 0)

    @pytest.mark.parametrize(
        "xi, xo, zh", zip([2.0, 3.0, 1.0], [9.0, 9.0, 4.0], [5.0, 4.0, 2.0])
    )
    def test_ep_designer(self, xi, xo, zh):
        """Test Equatorial Port Designer"""
        param_values = {"ep_height": zh}
        self.designer.params.update_values(param_values)
        self.designer.x_ib = xi
        self.designer.x_ob = xo
        output = self.designer.execute()
        x_len = xo - xi
        z_len = zh

        assert math.isclose(output.length, 2 * (x_len + z_len))
        assert math.isclose(output.area, x_len * z_len)


class TestEquatorialPortBuilder:
    """Tests the Equatorial Port Builder"""

    def setup_method(self) -> None:
        """Set-up to Equatorial Port Builder"""
        params = {
            "ep_height": {"value": 0, "unit": "m"},
            "cst_r_corner": {"value": 0, "unit": "m"},
        }
        y = (0, 1, 1, 0)
        z = (-1, -1, 1, 1)
        face = BluemiraFace(make_polygon({"x": 0, "y": y, "z": z}, closed=True))
        self.builder = EquatorialPortBuilder(params, {}, face, 1.0, 0.1)

    @pytest.mark.parametrize(
        "xi, xo, z, y, th",
        zip(
            [2.0, 3.0, 1.0],  # x_inboard
            [9.0, 9.0, 4.0],  # x_outboard
            [5.0, 4.0, 2.0],  # z_height
            [3.0, 2.0, 1.0],  # y_widths
            [-0.5, -0.5, -0.25],  # thickness
            # expected volumes: [63, 42, 5.25]
        ),
    )
    def test_ep_builder(self, xi, xo, z, y, th):
        """Test Equatorial Port Builder"""
        param_values = {"ep_height": z, "cst_r_corner": 0}
        self.builder.params.update_values(param_values)
        y_tup = (y / 2.0, -y / 2.0, -y / 2.0, y / 2.0)
        z_tup = (-z / 2.0, -z / 2.0, z / 2.0, z / 2.0)
        yz_profile = BluemiraWire(
            make_polygon({"x": xi, "y": y_tup, "z": z_tup}, closed=True)
        )
        length = xo - xi
        self.builder.length = length
        self.builder.outer = yz_profile
        self.builder.offset = th
        output = self.builder.build()
        out_port = output.get_component("xyz").get_component("Equatorial Port 1")
        expectation = length * (-2 * (th * (y + z + (2 * th))))

        assert math.isclose(out_port.shape.volume, expectation)


class TestCastellationBuilder:
    """Tests the Castellation Builder"""

    def setup_method(self) -> None:
        """Set-up Castellation Builder"""
        params = {
            "n_components": {"value": 10, "unit": ""},
            "cst_r_corner": {"value": 0, "unit": "m"},
        }
        length = 1
        y = (0, 1, 1, 0)
        z = (-1, -1, 1, 1)
        face = BluemiraFace(make_polygon({"x": 0, "y": y, "z": z}, closed=True))
        vector = (1, 0, 0)
        x_offs = [0]
        c_offs = [0]

        self.builder = CastellationBuilder(
            params, {}, length, face, vector, x_offs, c_offs
        )

    @pytest.mark.parametrize(
        "xi, xo, zh, yw, vec, x_offsets, c_offsets, exp_v",
        zip(
            [2.0, 3.0, 1.0],  # x_inboard
            [9.0, 9.0, 4.0],  # x_outboard
            [5.0, 4.0, 2.0],  # z_height
            [3.0, 2.0, 1.0],  # y_widths
            [(1, 0, 0), (1, 0, 0), (1, 0, 0.5)],  # extrusion vectors
            [[3.0], [2.0, 4.0], [1.0]],  # x castellation_positions
            [[1.0], [1.0, 1.0], [0.5]],  # y/z castellation_offsets
            [185.0, 160.0, 12.521980674],  # volume check value of Eq. Ports
        ),
    )
    def test_cst_builder(self, xi, xo, zh, yw, vec, x_offsets, c_offsets, exp_v):
        """Test Castellation Builder"""
        y = (yw / 2.0, -yw / 2.0, -yw / 2.0, yw / 2.0)
        z = (-zh / 2.0, -zh / 2.0, zh / 2.0, zh / 2.0)
        yz_profile = BluemiraFace(make_polygon({"x": xi, "y": y, "z": z}, closed=True))

        self.builder.length = xo - xi
        self.builder.face = yz_profile
        self.builder.vec = vec
        self.builder.off = x_offsets
        self.builder.cst = c_offsets
        output = self.builder.build()
        out_cst = output.get_component("xyz").get_component("Castellation 1")
        if out_cst is None:
            out_cst = output.get_component("xyz").get_component("Castellation")
        assert math.isclose(out_cst.shape.volume, exp_v)
