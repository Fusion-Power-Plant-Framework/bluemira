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
import json
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")
READ_DIR = os.path.join(DATA_DIR, "read")
RUN_DIR = os.path.join(DATA_DIR, "run")
FAKE_PROCESS_DICT = {  # Fake for the output of PROCESS's `get_dicts()`
    "DICT_DESCRIPTIONS": {"some_property": "its description"}
}


class FakeMFile:
    """
    A fake of PROCESS's MFile class.

    It replicates the :code:`.data` attribute with some PROCESS results
    data. This allows us to test the logic in our API without having
    PROCESS installed.
    """

    with open(os.path.join(DATA_DIR, "mfile_data.json"), "r") as f:
        data = json.load(f)

    def __init__(self, filename):
        self.filename = filename
