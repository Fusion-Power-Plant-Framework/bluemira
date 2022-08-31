# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2022 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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

import json
import os
from typing import Dict

import numpy as np
import pytest
from EUDEMO_builders.equilibria import UnconstrainedTikhonovSolver as UTSolver
from EUDEMO_builders.equilibria import UnconstrainedTikhonovSolverParams
from EUDEMO_builders.equilibria.tools import estimate_kappa95

from bluemira.equilibria import Equilibrium


class TestUnconstrainedTikhonovSolver:

    DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

    @classmethod
    def setup_class(cls):
        cls.param_dict = cls._read_json(os.path.join(cls.DATA_DIR, "params.json"))

    def test_params_converted_to_parameter_frame(self):
        solver = UTSolver(self.param_dict)

        assert isinstance(solver.params, UnconstrainedTikhonovSolverParams)

    @pytest.mark.longrun
    def test_solver_converges_on_run(self):
        solver = UTSolver(self.param_dict, {"plot_optimisation": True})

        eq = solver.execute(solver.run_mode_cls.RUN)

        assert eq.get_LCFS()
        # check parameters have been updated
        assert solver.params != UnconstrainedTikhonovSolverParams.from_dict(
            self.param_dict
        )

    def test_solver_reads_file_in_read_mode(self):
        eqdsk = os.path.join(os.path.dirname(__file__), "data", "equlibrium_eqdsk.json")
        solver = UTSolver(self.param_dict, {"file_path": eqdsk})

        eq = solver.execute(solver.run_mode_cls.READ)

        ref_eq = Equilibrium.from_eqdsk(eqdsk)
        assert eq.analyse_plasma() == ref_eq.analyse_plasma()

    def test_solver_outputs_equilibrium_in_mock_mode(self):
        solver = UTSolver(self.param_dict)

        eq = solver.execute(solver.run_mode_cls.MOCK)

        assert isinstance(eq, Equilibrium)
        expected_n_coils = sum(self.param_dict[p]["value"] for p in ["n_PF", "n_CS"])
        assert len(eq.coilset.coils) == expected_n_coils

    @staticmethod
    def _read_json(file_path: str) -> Dict:
        with open(file_path, "r") as f:
            return json.load(f)


class TestKappaLaw:
    """
    As per the conclusions of the CREATE report 2L4NMJ
    """

    @pytest.mark.parametrize(
        "A, m_s, expected",
        [
            [3.6, 0.3, 1.58],
            [3.1, 0.3, 1.68],
            [2.6, 0.3, 1.73],
            [3.6, 0, 1.66],
            [3.1, 0, 1.77],
            [2.6, 0, 1.80],
        ],
    )
    def test_kappa(self, A, m_s, expected):
        k95 = estimate_kappa95(A, m_s)
        np.testing.assert_allclose(k95, expected, rtol=5e-3)
