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

from bluemira.base.file import get_bluemira_path
from bluemira.base.parameter import ParameterFrame
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.geometry._deprecated_loop import Loop
from bluemira.radiation_transport.advective_transport import ChargedParticleSolver
from bluemira.radiation_transport.radiation_profile import Radiation, TwoPointModelTools

TEST_PATH = get_bluemira_path("radiation_transport/test_data", subfolder="tests")
EQ_PATH = get_bluemira_path("equilibria", subfolder="data")


class TestRadiation:
    @classmethod
    def setup_class(cls):
        eq_name = "DN-DEMO_eqref.json"
        filename = os.sep.join([EQ_PATH, eq_name])
        eq = Equilibrium.from_eqdsk(filename)
        fw_name = "DN_fw_shape.json"
        filename = os.sep.join([TEST_PATH, fw_name])
        cls.fw = Loop.from_file(filename)

        p_solver_params = ParameterFrame()
        solver = ChargedParticleSolver(p_solver_params, eq, dx_mp=0.001)
        x, z, hf = solver.analyse(cls.fw)
        cls.x, cls.z, cls.hf = np.array(x), np.array(z), np.array(hf)
        cls.solver = solver

        # fmt: off
        plasma_params = ParameterFrame([
            ["kappa", "Elongation", 3, "dimensionless", None, "Input"],
        ])
        # fmt: on

        # fmt: off
        rad_params = ParameterFrame([
            ["p_sol", "power entering the SoL", 300e6, "W", None, "Input"],
        ])
        # fmt: on

        cls.rad = Radiation(cls.solver, plasma_params)
        cls.tpm = TwoPointModelTools(cls.solver, plasma_params, rad_params)

    def test_flux_tube_pol_t(self):
        flux_tube = self.solver.eq.get_flux_surface(0.99)
        te = self.rad.flux_tube_pol_t(flux_tube, 100, True)
        assert te[0] == te[-1]
        assert len(te) == len(flux_tube)

    def test_key_temperature(self):
        t_u, q_u = self.tpm.upstream_temperature(self.fw)
        t_tar = self.tpm.target_temperature(q_u, t_u)
        t_x = self.tpm.x_point_temperature(q_u, t_u, self.fw)
        assert t_u < 5e-1
        assert t_tar < t_u * 1e-1
        assert t_u > t_x > t_tar
