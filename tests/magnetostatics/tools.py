# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from bluemira.geometry.coordinates import Coordinates


def make_xs_from_bd(b, d):
    return Coordinates(
        {
            "x": [-b, b, b, -b],
            "y": 0,
            "z": [-d, -d, d, d],
        }
    )
