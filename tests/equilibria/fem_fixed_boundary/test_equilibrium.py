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

import pytest

from bluemira.equilibria.fem_fixed_boundary.equilibrium import (
    PlasmaFixedBoundaryParams,
    _create_plasma_xz_cross_section,
    _interpolate_profile,
    _run_transport_solver,
    _solve_GS_problem,
    _update_delta_kappa_iteration_err,
    solve_transport_fixed_boundary,
)


def test_update_delta_kappa_it_err():

    delta, kappa, iter_err = _update_delta_kappa_iteration_err(
        PlasmaFixedBoundaryParams(1, 3, 5, 7, 9, 11), 10, 8, 6, 4, 2
    )

    assert delta == 10.8
    assert kappa == pytest.approx(6.666666)
    assert iter_err == 0.5
