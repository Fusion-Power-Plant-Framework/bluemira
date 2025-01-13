# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from pathlib import Path

import numpy as np
import pytest

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.constraint_funcs import (
    AxBConstraint,
    CoilForceConstraint,
    CurrentMidplanceConstraint,
    FieldConstraintFunction,
    L2NormConstraint,
)
from bluemira.optimisation._tools import approx_derivative

TEST_PATH = get_bluemira_path("equilibria/test_data", subfolder="tests")


class TestSimpleABConstraintFuntions:
    @classmethod
    def setup_class(cls):
        cls.vector = np.array([2, 2])
        cls.a_mat = np.array([[1, 0], [0, 1]])
        cls.b_vec = np.array([1, 1])
        cls.value = [2.0, 0.2]
        cls.scale = [1.0, 0.1]

    def constraint_setup(self, v, s, constraint):
        return constraint(
            a_mat=self.a_mat,
            b_vec=self.b_vec,
            value=v,
            scale=s,
        )

    def test_AxBConstraint(self):
        f_res = [-1.0, -1.0]
        df_res = [1.0, 0.1]
        for v, s, f, df in zip(self.value, self.scale, f_res, df_res, strict=False):
            axb = self.constraint_setup(v, s, AxBConstraint)
            test_f = axb.f_constraint(self.vector)
            test_df = axb.df_constraint(self.vector)
            assert isinstance(test_f, np.ndarray)
            assert isinstance(test_df, np.ndarray)
            assert all(test_f == f)
            assert all(np.diag(test_df) == df)

    def test_L2NormConstraint(self):
        f_res = [0.0, 1.08]
        # df_res = [2.0, -0.16]
        for v, s, f in zip(self.value, self.scale, f_res, strict=False):
            axb = self.constraint_setup(v, s, L2NormConstraint)
            test_f = axb.f_constraint(self.vector)
            test_df = axb.df_constraint(self.vector)
            assert isinstance(test_f, float)
            assert isinstance(test_df, np.ndarray)
            assert test_f == pytest.approx(f)


class TestEquilibriumInput:
    @classmethod
    def setup_class(cls):
        cls.eq = Equilibrium.from_eqdsk(Path(TEST_PATH, "eqref_OOB.json"), from_cocos=7)
        cls.coilset = cls.eq.coilset
        cls.vector = cls.coilset.current
        cls.scale = 1.0

    def test_field_constraint_function(self):
        x_test = [4.0, 4.0, np.array([4.0, 4.0])]
        z_test = [8.0, 6.0, np.array([8.0, 6.0])]
        B_max = [5.0, 5.0, np.array([5.0, 5.0])]
        const = [-2.08141466, -1.96647235, np.array([-2.08141466, -1.96647235])]
        for x, z, b, c in zip(x_test, z_test, B_max, const, strict=False):
            ax_mat = self.coilset.Bx_response(x, z, control=True)
            az_mat = self.coilset.Bz_response(x, z, control=True)
            bxp_vec = np.atleast_1d(self.eq.Bx(x, z))
            bzp_vec = np.atleast_1d(self.eq.Bz(x, z))
            fcf = FieldConstraintFunction(
                ax_mat=ax_mat,
                az_mat=az_mat,
                bxp_vec=bxp_vec,
                bzp_vec=bzp_vec,
                B_max=b,
                scale=self.scale,
            )
            l_c = 1 if isinstance(c, float) else len(c)
            assert len(fcf.f_constraint(self.vector)) == l_c
            assert fcf.f_constraint(self.vector) == pytest.approx(c)
            assert fcf.df_constraint(self.vector) == pytest.approx(
                approx_derivative(fcf.f_constraint, self.vector)
            )

    def test_current_midplane_constraint(self):
        ib_bool = [True, True, False, False]
        radius = [
            6.2912260811273745,
            5.891226081127375,
            11.79685790168689,
            11.99685790168689,
        ]
        constraint = [0.2, -0.2, 0.1, -0.1]
        for b, r, c in zip(ib_bool, radius, constraint, strict=False):
            cmc = CurrentMidplanceConstraint(
                eq=self.eq,
                radius=r,
                scale=self.scale,
                inboard=b,
            )
            assert cmc.f_constraint(self.vector) == pytest.approx(c)

    def test_coil_force_constraint(self):
        a_mat = self.coilset.control_F(self.coilset)
        b_vec = np.zeros((self.coilset.n_coils(), 2))
        non_zero = np.nonzero(self.vector)[0]
        b_vec[non_zero] = (
            self.coilset.F(self.eq)[non_zero] / self.vector[non_zero][:, None]
        )
        cfc = CoilForceConstraint(
            a_mat=a_mat,
            b_vec=b_vec,
            n_PF=self.coilset.n_coils("PF"),
            n_CS=self.coilset.n_coils("CS"),
            PF_Fz_max=450e6,
            CS_Fz_sum_max=300e6,
            CS_Fz_sep_max=350e6,
            scale=self.scale,
        )

        test_f_constraint = cfc.f_constraint(self.vector)
        test_df_constraint = cfc.df_constraint(self.vector)
        ref_f_constraint = np.array([
            3.73612632e07,
            1.32148390e08,
            3.01621694e08,
            3.70187813e08,
            1.47613187e08,
            9.98255464e08,
            8.78847911e08,
            -3.21248969e08,
            -1.17035624e09,
            -1.26766468e09,
            -1.50838972e09,
        ])
        approx_df_constraint = approx_derivative(cfc.f_constraint, self.vector)
        assert len(test_f_constraint) == len(self.vector)
        assert len(test_df_constraint[0, :]) == len(self.vector)
        assert len(test_df_constraint[:, 0]) == len(self.vector)
        assert [
            test == pytest.approx(ref)
            for test, ref in zip(test_f_constraint, ref_f_constraint, strict=False)
        ]
        for test, appx in zip(test_df_constraint, approx_df_constraint, strict=False):
            for t, a in zip(test, appx, strict=False):
                assert t == pytest.approx(a, rel=0.01)
