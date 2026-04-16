# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import pytest

cadapi = pytest.importorskip("bluemira.codes._cadqueryapi")

from tests.codes._shared.backend_api_tests import BackendApiTestsBase  # noqa: E402


class TestCadqueryapi(BackendApiTestsBase):
    cadapi = cadapi
