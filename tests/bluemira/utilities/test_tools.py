# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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
import os
import numpy as np
import json
from bluemira.base.file import get_bluemira_path
from bluemira.utilities.tools import NumpyJSONEncoder, is_num


class TestNumpyJSONEncoder:
    def test_childclass(self):
        fp = get_bluemira_path("bluemira/utilities/test_data", subfolder="tests")
        fn = os.sep.join([fp, "testJSONEncoder.json"])
        d = {"x": np.array([1, 2, 3.4, 4]), "y": [1, 3], "z": 3, "a": "aryhfdhsdf"}
        with open(fn, "w") as file:
            json.dump(d, file, cls=NumpyJSONEncoder)
        with open(fn, "r") as file:
            dd = json.load(file)
        for k, v in d.items():
            for kk, vv in dd.items():
                if k == kk:
                    if isinstance(v, np.ndarray):
                        assert v.tolist() == vv
                    else:
                        assert v == vv


def test_is_num():
    vals = [0, 34.0, 0.0, -0.0, 34e183, 28e-182, np.pi, np.inf]
    for v in vals:
        assert is_num(v) is True

    vals = [True, False, np.nan, object()]
    for v in vals:
        assert is_num(v) is False


if __name__ == "__main__":
    pytest.main([__file__])
