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
import io

from bluemira.codes.plasmod.params import PlasmodInputs


class TestPlasmodInputs:
    def test_write_writes_a_line_for_every_parameter(self):
        stream = io.StringIO()
        params = PlasmodInputs()

        params.write(stream)

        stream.seek(0)
        output = stream.read()
        lines = [line for line in output.split("\n") if line.strip()]
        params_dict = vars(params)
        assert len(lines) == len(params_dict)
        for param in params_dict:
            assert any([param in line for line in lines])
