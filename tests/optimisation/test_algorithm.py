# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
import pytest

from bluemira.optimisation._algorithm import Algorithm


class TestAlgorithm:
    @pytest.mark.parametrize("alg", list(Algorithm))
    def test_algorithm_can_be_made_from_name(self, alg):
        assert Algorithm[alg.name] == alg

    def test_direct_l_can_be_made_from_str_with_hyphen(self):
        assert Algorithm("DIRECT-L") == Algorithm.DIRECT_L

    def test_ValueError_given_unknown_algorithm_str(self):
        with pytest.raises(ValueError):  # noqa: PT011
            Algorithm("NOT_AN_ALG")
