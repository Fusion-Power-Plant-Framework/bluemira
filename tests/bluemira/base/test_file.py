# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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

import pytest
from bluemira.base.file import get_bluemira_path, try_get_bluemira_path


@pytest.mark.parametrize(
    "path,subfolder,allow_missing,expect_none",  # noqa(N802)
    [
        ("bluemira", "tests", False, False),  # NOTE: Change "bluemira" to e.g. "base"
        ("bluemira", "tests", True, False),  # NOTE: Change "bluemira" to e.g. "base"
        ("spam", "tests", True, True),
        ("spam", "ham", True, True),
    ],
)
def test_try_get_bluemira_path(path, subfolder, allow_missing, expect_none):
    output_path = try_get_bluemira_path(
        path, subfolder=subfolder, allow_missing=allow_missing
    )
    if expect_none:
        assert output_path is None
    else:
        assert output_path == get_bluemira_path(path, subfolder=subfolder)


@pytest.mark.parametrize(
    "path,subfolder",  # noqa(N802)
    [
        ("spam", "tests"),
        ("spam", "ham"),
    ],
)
def test_try_get_bluemira_path_raises(path, subfolder):
    with pytest.raises(ValueError):
        try_get_bluemira_path(path, subfolder=subfolder, allow_missing=False)


if __name__ == "__main__":
    pytest.main([__file__])
