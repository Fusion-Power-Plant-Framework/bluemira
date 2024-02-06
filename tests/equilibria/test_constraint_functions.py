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
    L2NormConstraint,
)

TEST_PATH = get_bluemira_path("equilibria/test_data", subfolder="tests")


class TestSimpleConstraintFuntions:
    @classmethod
    def setup_class(cls):
        cls.vector = np.array([2, 2])
        cls.a_mat = np.array([[1, 0], [0, 1]])
        cls.b_vec = np.array([1, 1])
        cls.value = [2.0, 0.2]
        cls.scale = [1.0, 0.1]

    def constrint_setup(self, v, s, constraint):
        con_setup = constraint(
            a_mat=self.a_mat,
            b_vec=self.b_vec,
            value=v,
            scale=s,
        )
        print(v, s)
        return con_setup

    def test_AxBConstraint(self):
        f_res = [-1.0, -1.0]
        df_res = [1.0, 0.1]
        for v, s, f, df in zip(self.value, self.scale, f_res, df_res):
            axb = self.constrint_setup(v, s, AxBConstraint)
            test_f = axb.f_constraint(self.vector)
            test_df = axb.df_constraint(self.vector)
            assert isinstance(test_f, np.ndarray)
            assert isinstance(test_df, np.ndarray)
            assert all(test_f == f)
            assert all(np.diag(test_df) == df)

    def test_L2NormConstraint(self):
        f_res = [0.0, 1.08]
        # df_res = [2.0, -0.16]
        for v, s, f in zip(self.value, self.scale, f_res):
            axb = self.constrint_setup(v, s, L2NormConstraint)
            test_f = axb.f_constraint(self.vector)
            test_df = axb.df_constraint(self.vector)
            assert isinstance(test_f, float)
            assert isinstance(test_df, np.ndarray)
            assert test_f == pytest.approx(f)


def test_FieldConstraintFunction():
    pass


def test_CurrentMidplanceConstraint():
    eq_name = "eqref_OOB.json"
    eq = Equilibrium.from_eqdsk(Path(TEST_PATH, eq_name))
    pass


def test_CoilForceConstraint():
    pass
