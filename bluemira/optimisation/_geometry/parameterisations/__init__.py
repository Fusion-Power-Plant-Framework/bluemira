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

from bluemira.geometry.parameterisations import PrincetonD, SextupleArc, TripleArc
from bluemira.optimisation._geometry.parameterisations import (
    princeton_d,
    sextuple_arc,
    triple_arc,
)

INEQ_CONSTRAINT_REGISTRY = {
    PrincetonD: [
        {
            "f_constraint": princeton_d.f_ineq_constraint,
            "df_constraint": princeton_d.df_ineq_constraint,
            "tolerance": princeton_d.tol(),
        }
    ],
    TripleArc: [
        {
            "f_constraint": triple_arc.f_ineq_constraint,
            "df_constraint": triple_arc.df_ineq_constraint,
            "tolerance": triple_arc.tol(),
        }
    ],
    SextupleArc: [
        {
            "f_constraint": sextuple_arc.f_ineq_constraint,
            "df_constraint": sextuple_arc.df_ineq_constraint,
            "tolerance": sextuple_arc.tol(),
        }
    ],
}
