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

import os

import numpy as np

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.opt_constraints import (
    IsofluxConstraint,
    MagneticConstraintSet,
    PsiBoundaryConstraint,
)
from bluemira.equilibria.opt_problems import TikhonovCurrentCOP


# @pytest.mark.longrun
class TestWeightedConstraints:
    def test_constraint_weights(self):
        """
        Checks that supplied weights are applied to the applied constraints.
        """

        # Generate a test equilibrium
        path = get_bluemira_path("equilibria/test_data", subfolder="tests")
        fn = os.sep.join([path, "DN-DEMO_eqref.json"])
        eq = Equilibrium.from_eqdsk(fn)

        # Test that both default weights and custom weights can be applied
        for apply_weights in (True, False):
            # Create dummy constraints. Isoflux Constraint is an example of a
            # relative constraint with weights applied, whereas PsiBoundaryConstraint
            # is absolute.
            x_iso = np.array([0.6, 1.0, 1.5, 1.0])
            z_iso = np.array([0.0, 4.3, 0.1, -4.0])
            w_iso = np.array([3.0, 1.0, 10.0, 0.0])

            x_psi = np.array([0.0, 9.0, 15.0])
            z_psi = np.array([0.0, 8.0, 8.0])
            w_psi = np.array([2.0, 0.5, 3.0])

            if apply_weights:
                constraint_set = MagneticConstraintSet(
                    [
                        IsofluxConstraint(
                            x_iso, z_iso, ref_x=0.5, ref_z=0.5, weights=w_iso
                        ),
                        PsiBoundaryConstraint(
                            x_psi, z_psi, target_value=0.0, weights=w_psi
                        ),
                    ]
                )
                weights = np.concatenate([w_iso, w_psi])
            else:
                constraint_set = MagneticConstraintSet(
                    [
                        IsofluxConstraint(x_iso, z_iso, ref_x=0.5, ref_z=0.5),
                        PsiBoundaryConstraint(x_psi, z_psi, target_value=0.0),
                    ]
                )
                weights = np.ones(len(constraint_set))

            # Populate constraint set based on test equilibrium
            constraint_set(eq)

            # Test that weights have been applied

            problem = TikhonovCurrentCOP(
                eq.coilset,
                eq,
                constraint_set,
                gamma=1e-8,
            )
            problem.optimise(fixed_coils=True)

            assert np.allclose(
                problem._objective._args["b_vec"], weights * constraint_set.b
            )
            for (i, weight) in enumerate(weights):
                assert np.allclose(
                    problem._objective._args["a_mat"][i, :],
                    weight * constraint_set.A[i, :],
                )
