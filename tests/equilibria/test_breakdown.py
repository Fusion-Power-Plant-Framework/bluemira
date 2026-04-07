# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from pathlib import Path

import numpy as np
import pytest

from bluemira.equilibria.constants import BLUEMIRA_DEFAULT_COCOS
from bluemira.equilibria.equilibrium import Breakdown
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.optimisation.constraints import (
    CoilFieldConstraints,
    CoilForceConstraints,
)
from bluemira.equilibria.optimisation.problem._breakdown import (
    BreakdownCOP,
    OutboardBreakdownZoneStrategy,
)
from tests.equilibria.setup_methods import _coilset_setup


class TestBreakdown:
    @classmethod
    def setup_class(cls):
        _coilset_setup(cls, materials=True)
        grid = Grid(2, 16.0, -9.0, 9.0, 100, 100)
        cls.breakdown = Breakdown(cls.coilset, grid)

        # Coil constraints
        PF_Fz_max = 450e6  # [N]
        CS_Fz_sum = 300e6  # [N]
        CS_Fz_sep = 350e6  # [N]

        field_constraints = CoilFieldConstraints(
            cls.coilset, cls.coilset.b_max, tolerance=1e-6
        )
        force_constraints = CoilForceConstraints(
            cls.coilset, PF_Fz_max, CS_Fz_sum, CS_Fz_sep, tolerance=1e-6
        )

        R_0 = 8.938
        A = 3.1
        max_currents = cls.coilset.get_max_current(0.0)
        bd_opt_problem = BreakdownCOP(
            cls.breakdown,
            OutboardBreakdownZoneStrategy(R_0, A, 0.225),
            opt_algorithm="COBYLA",
            opt_conditions={"max_eval": 3000, "ftol_rel": 1e-6},
            max_currents=max_currents,
            B_stray_max=1e-3,
            B_stray_con_tol=1e-6,
            n_B_stray_points=10,
            constraints=[field_constraints, force_constraints],
        )

        cls.coilset = bd_opt_problem.optimise(x0=max_currents).coilset

    def test_breakdown_flux(self):
        bd_flux_vs = self.breakdown.breakdown_psi * 2 * np.pi
        assert bd_flux_vs > 245

    @pytest.mark.parametrize("file_format", ["json", "eqdsk"])
    def test_eqdsk_write_read(self, file_format, tmp_path):
        new_file_name = f"test_breakdown.{file_format}"
        new_file_path = Path(tmp_path, new_file_name)
        self.breakdown.to_eqdsk(
            new_file_name,
            directory=tmp_path,
            filetype=file_format,
            to_cocos=BLUEMIRA_DEFAULT_COCOS,
        )
        new_bd = Breakdown.from_eqdsk(
            new_file_path, from_cocos=BLUEMIRA_DEFAULT_COCOS, force_symmetry=False
        )
        bd_flux_vs = new_bd.breakdown_psi * 2 * np.pi
        recalced_flux_vs = new_bd.psi(*new_bd.breakdown_point) * 2 * np.pi
        assert bd_flux_vs > 245
        assert recalced_flux_vs > 245
