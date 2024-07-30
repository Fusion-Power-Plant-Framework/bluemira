# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from pathlib import Path

import numpy as np
from eqdsk.models import Sign

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.constraints import (
    IsofluxConstraint,
    MagneticConstraintSet,
    PsiBoundaryConstraint,
)
from bluemira.equilibria.optimisation.problem import TikhonovCurrentCOP


class TestWeightedConstraints:
    def test_constraint_weights(self):
        """
        Checks that supplied weights are applied to the applied constraints.
        """

        # Generate a test equilibrium
        path = get_bluemira_path("equilibria/test_data", subfolder="tests")
        fn = Path(path, "DN-DEMO_eqref.json")
        eq = Equilibrium.from_eqdsk(fn, from_cocos=3, qpsi_sign=Sign.NEGATIVE)

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
                constraint_set = MagneticConstraintSet([
                    IsofluxConstraint(x_iso, z_iso, ref_x=0.5, ref_z=0.5, weights=w_iso),
                    PsiBoundaryConstraint(x_psi, z_psi, target_value=0.0, weights=w_psi),
                ])
                weights = np.concatenate([w_iso, w_psi])
            else:
                constraint_set = MagneticConstraintSet([
                    IsofluxConstraint(x_iso, z_iso, ref_x=0.5, ref_z=0.5),
                    PsiBoundaryConstraint(x_psi, z_psi, target_value=0.0),
                ])
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

            _, w_a_mat, w_b_vec = problem.targets.get_weighted_arrays()
            assert np.allclose(w_b_vec, weights * constraint_set.b)
            assert np.allclose(w_a_mat, weights[:, None] * constraint_set.A)
