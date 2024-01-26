# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import matplotlib.pyplot as plt

from bluemira.equilibria.equilibrium import Breakdown
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.optimisation.constraints import (
    CoilFieldConstraints,
    CoilForceConstraints,
)
from bluemira.equilibria.optimisation.problem import (
    BreakdownCOP,
    OutboardBreakdownZoneStrategy,
)
from tests.equilibria.setup_methods import _coilset_setup


class TestBreakdown:
    @classmethod
    def setup_class(cls):
        _coilset_setup(cls, materials=True)
        grid = Grid(3, 20, -10, 10, 65, 65)
        cls.bd = Breakdown(cls.coilset, grid)
        cls.coilset.fix_sizes()
        cls.coilset.discretisation = 0.3
        strategy = OutboardBreakdownZoneStrategy(9.0, 3.1, 0.225)
        cls.problem = BreakdownCOP(
            cls.coilset,
            cls.bd,
            strategy,
            B_stray_max=0.003,
            B_stray_con_tol=1e-6,
            n_B_stray_points=11,
            max_currents=cls.coilset.get_max_current(0.0),
            constraints=[
                CoilForceConstraints(cls.coilset, 450, 300, 350),
                CoilFieldConstraints(cls.coilset, cls.coilset.b_max),
            ],
            opt_algorithm="COBYLA",
            opt_conditions={"ftol_rel": 1e-10, "max_eval": 1000},
        )
        cls.bd.set_breakdown_point(*strategy.breakdown_point)

    def test_solve_breakdown(self):
        cs_coils = self.coilset.get_coiltype("CS")
        cs_coils.current = cs_coils.get_max_current()
        self.problem.optimise()
        f, ax = plt.subplots()
        self.bd.plot(ax=ax)
        self.bd.coilset.plot(ax=ax, label=True)
        plt.show()
