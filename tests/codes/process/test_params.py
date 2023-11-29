# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import pytest

from bluemira.codes.process.params import ParameterMapping, ProcessSolverParams
from tests.codes.process.utilities import PARAM_FILE


class TestProcessSolverParams:
    params = ProcessSolverParams.from_json(PARAM_FILE)

    @pytest.mark.parametrize("param", params, ids=lambda p: p.name)
    def test_mapping_defined_for_param(self, param):
        assert isinstance(self.params.mappings[param.name], ParameterMapping)
