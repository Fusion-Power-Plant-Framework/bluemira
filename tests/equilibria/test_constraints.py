# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.constraints import (
    CoilFieldConstraints,
    CoilForceConstraints,
    IsofluxConstraint,
    MagneticConstraintSet,
    PsiBoundaryConstraint,
)
from bluemira.equilibria.optimisation.problem import TikhonovCurrentCOP


class TestConstraintMechanics:
    @classmethod
    def setup_class(cls):
        # Generate a test equilibrium
        path = get_bluemira_path("equilibria/test_data", subfolder="tests")
        fn = Path(path, "DN-DEMO_eqref.json")
        cls._eq = Equilibrium.from_eqdsk(fn, from_cocos=3, qpsi_positive=False)

        # Test isoflux constraint locations and weights
        cls.x_iso = np.array([0.6, 1.0, 1.5, 1.0])
        cls.z_iso = np.array([0.0, 4.3, 0.1, -4.0])
        cls.w_iso = np.array([3.0, 1.0, 10.0, 0.0])

        # Test psi constraint locations and weights
        cls.x_psi = np.array([0.0, 9.0, 15.0])
        cls.z_psi = np.array([0.0, 8.0, 8.0])
        cls.w_psi = np.array([2.0, 0.5, 3.0])

        # Test force and field constraint values
        cls.B_max = 11.5
        cls.PF_Fz_max = 450e6
        cls.CS_Fz_sum_max = 300e6
        cls.CS_Fz_sep_max = 250e6

        cls.pf_names = cls._eq.coilset.get_coiltype("PF").name
        cls.cs_names = cls._eq.coilset.get_coiltype("CS").name

    def setup_method(self):
        self.eq = deepcopy(self._eq)

    def teardown_method(self):
        del self.eq

    @pytest.mark.parametrize("control", [False, True])
    def test_control_coils(self, control):
        """
        Checks that the number of coils used matches the control coils.
        Default is all coils are control coils.
        """
        if control:
            # Test coilset with n control coils < n coils
            self.eq.coilset.control = [*self.pf_names[:3], *self.cs_names[:3]]

        # Field
        field_c = CoilFieldConstraints(self.eq.coilset, self.B_max)
        field_c.prepare(self.eq)
        assert np.shape(field_c._args["ax_mat"]) == (
            len(self.eq.coilset.control),
            len(self.eq.coilset.control),
        )
        assert np.shape(field_c._args["az_mat"]) == (
            len(self.eq.coilset.control),
            len(self.eq.coilset.control),
        )
        assert len(field_c._args["bxp_vec"]) == len(self.eq.coilset.control)
        assert len(field_c._args["bzp_vec"]) == len(self.eq.coilset.control)

        # Force
        force_c = CoilForceConstraints(
            self.eq.coilset, self.PF_Fz_max, self.CS_Fz_sum_max, self.CS_Fz_sep_max
        )
        force_c.prepare(self.eq)
        assert np.shape(force_c._args["a_mat"]) == (
            len(self.eq.coilset.control),
            len(self.eq.coilset.control),
            2,
        )
        assert len(force_c._args["b_vec"]) == len(self.eq.coilset.control)

        # Set
        mcs = MagneticConstraintSet([
            IsofluxConstraint(self.x_iso, self.z_iso, ref_x=0.5, ref_z=0.5),
            PsiBoundaryConstraint(self.x_psi, self.z_psi, target_value=0.0),
        ])
        mcs.__call__(equilibrium=self.eq)
        assert np.shape(mcs.A) == (
            len(self.x_psi) + len(self.x_iso),
            len(self.eq.coilset.control),
        )

    @pytest.mark.parametrize("apply_weights", [True, False])
    def test_constraint_weights(self, apply_weights):
        """
        Checks that supplied weights (default weights and custom)
        are applied to constraints.
        """
        # Create dummy constraints. Isoflux Constraint is an example
        # of a relative constraint with weights applied, whereas
        # PsiBoundaryConstraint is absolute
        if apply_weights:
            constraint_set = MagneticConstraintSet([
                IsofluxConstraint(
                    self.x_iso, self.z_iso, ref_x=0.5, ref_z=0.5, weights=self.w_iso
                ),
                PsiBoundaryConstraint(
                    self.x_psi, self.z_psi, target_value=0.0, weights=self.w_psi
                ),
            ])
            weights = np.concatenate([self.w_iso, self.w_psi])
        else:
            constraint_set = MagneticConstraintSet([
                IsofluxConstraint(self.x_iso, self.z_iso, ref_x=0.5, ref_z=0.5),
                PsiBoundaryConstraint(self.x_psi, self.z_psi, target_value=0.0),
            ])
            weights = np.ones(len(constraint_set))

        # Populate constraint set based on test equilibrium
        constraint_set(self.eq)

        # Test that weights have been applied
        problem = TikhonovCurrentCOP(
            self.eq.coilset,
            self.eq,
            constraint_set,
            gamma=1e-8,
        )
        problem.optimise(fixed_coils=True)
        _, w_a_mat, w_b_vec = problem.targets.get_weighted_arrays()
        assert np.allclose(w_b_vec, weights * constraint_set.b)
        assert np.allclose(w_a_mat, weights[:, None] * constraint_set.A)
