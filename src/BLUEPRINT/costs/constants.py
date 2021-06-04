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

"""
Constants for the cost module.
"""

RELATIVE_COST_FRACTIONS = {
    "RS": 1,
    "VV": 10,
    "TS": 5,
    "CR": 3,
    "BB": 70,
    "DIV": 90,
    "TF wp": 100,
    "TF case": 15,
    "coil structures": 15,
    "PF": 50,
    "CS": 80,
}

AUXILIARY_COST_FRACTIONS = {
    "RS": 2.5,
    "VV": 2,
    "TS": 2,
    "CR": 1.5,
    "BB": 1.7,
    "DIV": 1.5,
    "TF wp": 2.1,
    "TF case": 1.05,
    "coil structures": 1.05,
    "PF": 2.1,
    "CS": 2.1,
}


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
