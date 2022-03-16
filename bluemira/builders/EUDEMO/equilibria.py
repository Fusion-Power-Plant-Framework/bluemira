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

"""
Equilibrium objects for EU-DEMO design
"""

import numpy as np

from bluemira.equilibria.eq_constraints import (
    DivertorLegCalculator,
    FieldNullConstraint,
    MagneticConstraintSet,
    PsiBoundaryConstraint,
)
from bluemira.equilibria.shapes import flux_surface_johner


class EUDEMOSingleNullConstraints(DivertorLegCalculator, MagneticConstraintSet):
    """
    Parameterised family of magnetic constraints for a typical EU-DEMO-like single
    null equilibrium.
    """

    def __init__(
        self,
        R_0,
        Z_0,
        A,
        kappa_u,
        kappa_l,
        delta_u,
        delta_l,
        psi_u_neg,
        psi_u_pos,
        psi_l_neg,
        psi_l_pos,
        div_l_ib,
        div_l_ob,
        psibval,
        lower=True,
        n=100,
    ):
        constraints = []
        f_s = flux_surface_johner(
            R_0,
            Z_0,
            R_0 / A,
            kappa_u,
            kappa_l,
            delta_u,
            delta_l,
            psi_u_neg,
            psi_u_pos,
            psi_l_neg,
            psi_l_pos,
            n=200,
        )

        if lower:
            arg_x = np.argmin(f_s.z)
        else:
            arg_x = np.argmax(f_s.z)

        x_point = [f_s.x[arg_x], f_s.z[arg_x]]

        constraints = [FieldNullConstraint(*x_point)]

        f_s.interpolate(n)
        x_s, z_s = f_s.x, f_s.z

        constraints.append(PsiBoundaryConstraint(x_s, z_s, psibval))

        x_leg1, z_leg1 = self.calc_divertor_leg(
            x_point, 50, div_l_ob, int(n / 10), loc="lower", pos="outer"
        )

        x_leg2, z_leg2 = self.calc_divertor_leg(
            x_point, 40, div_l_ib, int(n / 10), loc="lower", pos="inner"
        )

        x_legs = np.append(x_leg1, x_leg2)
        z_legs = np.append(z_leg1, z_leg2)
        constraints.append(PsiBoundaryConstraint(x_legs, z_legs, psibval))

        super().__init__(constraints)


class EUDEMODoubleNullConstraints(DivertorLegCalculator, MagneticConstraintSet):
    """
    Parameterised family of magnetic constraints for a typical EU-DEMO-like double
    null equilibrium.
    """

    def __init__(
        self,
        R_0,
        Z_0,
        A,
        kappa,
        delta,
        psi_neg,
        psi_pos,
        div_l_ib,
        div_l_ob,
        psibval,
        n=400,
    ):
        super().__init__()
        f_s = flux_surface_johner(
            R_0,
            Z_0,
            R_0 / A,
            kappa,
            kappa,
            delta,
            delta,
            psi_neg,
            psi_pos,
            psi_neg,
            psi_pos,
            n=200,
        )

        arg_xl = np.argmin(f_s.z)
        arg_xu = np.argmax(f_s.z)
        constraints = [
            FieldNullConstraint(f_s.x[arg_xl], f_s.z[arg_xl]),
            FieldNullConstraint(f_s.x[arg_xu], f_s.z[arg_xu]),
        ]
        f_s.interpolate(n)
        x_s, z_s = f_s.x, f_s.z

        constraints.append(PsiBoundaryConstraint(x_s, z_s, psibval))

        super().__init__(constraints)
