# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from setup_methods import _coilset_setup

from bluemira.equilibria.equilibrium import Breakdown
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.optimisation.problem import (
    BreakdownCOP,
    OutboardBreakdownZoneStrategy,
)


class TestBreakdown:
    @classmethod
    def setup_class(cls):
        _coilset_setup(cls, materials=True)
        grid = Grid(3, 20, -10, 10, 65, 65)
        cls.bd = Breakdown(cls.coilset, grid)
        cls.coilset.fix_sizes()
        cls.coilset.discretisation = 0.3

        problem = BreakdownCOP(
            cls.coilset,
            cls.bd,
            OutboardBreakdownZoneStrategy(9.0, 3.1, 0.225),
            B_stray_max=0.003,
            B_stray_con_tol=1e-6,
            n_B_stray_points=10,
            max_currents=cls.coilset.get_max_current(0.0),
        )
