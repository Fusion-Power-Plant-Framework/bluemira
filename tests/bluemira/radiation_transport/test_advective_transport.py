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

import os
import numpy as np
import pytest

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from BLUEPRINT.base.parameter import ParameterFrame
from bluemira.geometry._deprecated_loop import Loop
from bluemira.radiation_transport.error import AdvectionTransportError
from bluemira.radiation_transport.advective_transport import ChargedParticleSolver


TEST_PATH = get_bluemira_path(
    "bluemira/radiation_transport/test_data", subfolder="tests"
)
EQ_PATH = get_bluemira_path("equilibria", subfolder="data")


class TestChargedParticleAPIinputs:
    def test_bad_fractions(self):
        params = ParameterFrame(
            [
                [
                    "f_outer_target",
                    "Fraction of SOL power deposited on the outer target(s)",
                    0.75,
                    "N/A",
                    None,
                    "Input",
                ],
                [
                    "f_inner_target",
                    "Fraction of SOL power deposited on the inner target(s)",
                    0.5,
                    "N/A",
                    None,
                    "Input",
                ],
            ]
        )

        with pytest.raises(AdvectionTransportError):
            ChargedParticleSolver(params, None)

        params = ParameterFrame(
            [
                ["f_upper_target", "Power fraction", 0.1, "N/A", None, "Input"],
                ["f_lower_target", "Power fraction", 0.5, "N/A", None, "Input"],
            ]
        )
        with pytest.raises(AdvectionTransportError):
            ChargedParticleSolver(params, None)

    def test_bad_inputs(self):
        pass


class TestChargedParticleRecursionSN:
    @classmethod
    def setup_class(cls):
        eq_name = "EU-DEMO_EOF.json"
        filename = os.sep.join([EQ_PATH, eq_name])
        eq = Equilibrium.from_eqdsk(filename)
        fw_name = "first_wall.json"
        filename = os.sep.join([TEST_PATH, fw_name])
        fw = Loop.from_file(filename)
        # fmt: off
        cls.params = ParameterFrame([
        ["fw_p_sol_near", "near scrape-off layer power", 50, "MW", None, "Input"],
        ["fw_p_sol_far", "far scrape-off layer power", 50, "MW", None, "Input"],
        ["fw_lambda_q_near", "Lambda q near SOL at the outboard", 0.05, "m", None, "Input"],
        ["fw_lambda_q_far", "Lambda q far SOL at the outboard", 0.05, "m", None, "Input"],
        ["fw_lambda_q_near_ib", "Lambda q near SOL at the inboard", 0.05, "m", None, "Input"],
        ["fw_lambda_q_far_ib", "Lambda q far SOL at the inboard", 0.05, "m", None, "Input"],
        ["f_outer_target", "Fraction of SOL power deposited on the outer target(s)", 0.75, "N/A", None, "Input"],
        ["f_inner_target", "Fraction of SOL power deposited on the inner target(s)", 0.25, "N/A", None, "Input"],
        ["f_upper_target", "Fraction of SOL power deposited on the upper targets. DN only", 0.5, "N/A", None, "Input"],
        ["f_lower_target", "Fraction of SOL power deposited on the lower target, DN only", 0.5, "N/A", None, "Input"],
    ])
        # fmt: on

        solver = ChargedParticleSolver(cls.params, eq, dx_mp=0.001)
        x, z, hf = solver.analyse(fw)
        cls.x, cls.z, cls.hf = np.array(x), np.array(z), np.array(hf)
        cls.solver = solver

    def test_recursion(self):

        assert np.isclose(np.max(self.hf), 5.379, rtol=1e-2)
        assert np.argmax(self.hf) == 0
        assert np.isclose(np.sum(self.hf), 399, rtol=1e-2)

    def test_n_intersections(self):
        """
        Because it is a single null, we expect the same number of flux surfaces LFS and
        HFS.
        """
        assert 2 * len(self.solver.flux_surfaces) == len(self.x)

    def test_integrals(self):
        n_fs = len(self.solver.flux_surfaces)
        x_lfs = self.x[:n_fs]
        x_hfs = self.x[n_fs:]
        z_lfs = self.z[:n_fs]
        z_hfs = self.z[n_fs:]
        hf_lfs = self.hf[:n_fs]
        hf_hfs = self.hf[n_fs:]

        dx_lfs = x_lfs[:-1] - x_lfs[1:]
        dz_lfs = z_lfs[:-1] - z_lfs[1:]
        d_lfs = np.hypot(dx_lfs, dz_lfs)
        q_lfs = sum(hf_lfs[:-1] * d_lfs * (x_lfs[:-1] + 0.5 * abs(dx_lfs)))

        dx_hfs = x_hfs[:-1] - x_hfs[1:]
        dz_hfs = z_hfs[:-1] - z_hfs[1:]
        d_hfs = np.hypot(dx_hfs, dz_hfs)
        q_hfs = sum(hf_hfs[:-1] * d_hfs * (x_hfs[:-1] + 0.5 * abs(dx_hfs)))
        true_total = self.params["fw_p_sol_near"] + self.params["fw_p_sol_far"]
        assert np.isclose(q_lfs + q_hfs, true_total, rtol=2e-2)


class TestChargedParticleRecursionDN:
    @classmethod
    def setup_class(cls):
        eq_name = "DN-DEMO_eqref.json"
        filename = os.sep.join([EQ_PATH, eq_name])
        eq = Equilibrium.from_eqdsk(filename)
        fw_name = "DN_fw_shape.json"
        filename = os.sep.join([TEST_PATH, fw_name])
        fw = Loop.from_file(filename)
        # fmt: off
        cls.params = ParameterFrame([
        ["fw_p_sol_near", "near scrape-off layer power", 90, "MW", None, "Input"],
        ["fw_p_sol_far", "far scrape-off layer power", 50, "MW", None, "Input"],
        ["fw_lambda_q_near", "Lambda q near SOL at the outboard", 0.003, "m", None, "Input"],
        ["fw_lambda_q_far", "Lambda q far SOL at the outboard", 0.1, "m", None, "Input"],
        ["fw_lambda_q_near_ib", "Lambda q near SOL at the inboard", 0.003, "m", None, "Input"],
        ["fw_lambda_q_far_ib", "Lambda q far SOL at the inboard", 0.1, "m", None, "Input"],
        ["f_outer_target", "Fraction of SOL power deposited on the outer target(s)", 0.9, "N/A", None, "Input"],
        ["f_inner_target", "Fraction of SOL power deposited on the inner target(s)", 0.1, "N/A", None, "Input"],
        ["f_upper_target", "Fraction of SOL power deposited on the upper targets. DN only", 0.5, "N/A", None, "Input"],
        ["f_lower_target", "Fraction of SOL power deposited on the lower target, DN only", 0.5, "N/A", None, "Input"],
                            ])
        # fmt: on

        solver = ChargedParticleSolver(cls.params, eq, dx_mp=0.001)
        x, z, hf = solver.analyse(fw)
        cls.x, cls.z, cls.hf = np.array(x), np.array(z), np.array(hf)
        cls.solver = solver

    def test_recursion(self):
        assert np.isclose(np.max(self.hf), 85.194, rtol=1e-2)
        assert np.isclose(np.sum(self.hf), 830.6, rtol=1e-2)
