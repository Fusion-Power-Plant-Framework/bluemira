# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from pathlib import Path

import pytest

from bluemira.codes.plasmod.params import ParameterMapping, PlasmodSolverParams

PARAM_FILE = Path(Path(__file__).parent, "data", "params.json").as_posix()


class TestProcessSolverParams:
    params = PlasmodSolverParams.from_json(PARAM_FILE)

    @pytest.mark.parametrize("param", params, ids=lambda p: p.name)
    def test_mapping_defined_for_param(self, param):
        assert isinstance(self.params.mappings[param.name], ParameterMapping)
