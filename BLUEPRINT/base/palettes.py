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

"""
Some more aesthetic standardisation.. color palettes
"""
import numpy as np
import seaborn as sns

from BLUEPRINT.utilities.colortools import make_rgb_alpha

BLUE = dict(
    zip(
        ["BB", "DIV", "VV", "TS", "TF", "PF", "CR", "RS"],
        [sns.color_palette("Blues_r", 9)[i] for i in [3, 7, 6, 8, 0, 1, 2, 5]],
    )
)
BLUE["TF"] = [BLUE["TF"], make_rgb_alpha(BLUE["TF"], 0.5)]
BLUE["PF"] = [BLUE["PF"]]
BLUE["CS"] = [make_rgb_alpha(BLUE["PF"][0], 0.5)]
BLUE["BB"] = [
    BLUE["BB"],
    make_rgb_alpha(BLUE["BB"], 0.7),
    make_rgb_alpha(BLUE["BB"], 0.6),
    make_rgb_alpha(BLUE["BB"], 0.5),
    make_rgb_alpha(BLUE["BB"], 0.4),
    make_rgb_alpha(BLUE["BB"], 0.3),
    make_rgb_alpha(BLUE["BB"], 0.2),
    make_rgb_alpha(BLUE["BB"], 0.1),
]
BLUE["FW"] = BLUE["BB"]
BLUE["CCS"] = BLUE["BB"]
BLUE["ATEC"] = [
    make_rgb_alpha(BLUE["PF"][0], 0.2),
    make_rgb_alpha(BLUE["PF"][0], 0.4),
    make_rgb_alpha(BLUE["PF"][0], 0.6),
]
BLUE["PL"] = [(0.80078431, 0.54, 0.80078431)]
BLUE["HCD"] = list(np.array([178, 34, 34]) / 255)  # TODO: include a HCD blue


LU = sns.color_palette(
    [
        "#B36305",
        "#E32017",
        "#FFD300",
        "#00782A",
        "#00A4A7",
        "#F3A9BB",
        "#A0A5A9",
        "#9B0056",
        "#000000",
        "#EE7C0E",
        "#003688",
        "#84B817",
        "#0098D4",
        "#95CDBA",
    ]
)


LONDON = {
    "Northern": "#000000",
    "Waterloo and City": "#95CDBA",
    "Piccadilly": "#003688",
    "Central": "#E32017",
    "District": "#00782A",
    "DLR": "#00A4A7",
    "Hammersmith and City": "#F3A9BB",
    "Jubilee": "#A0A5A9",
    "Metropolitan": "#9B0056",
    "Overground": "#EE7C0E",
    "Tramlink": "#84B817",
    "Victoria": "#0098D4",
    "Bakerloo": "#B36305",
    "Circle": "#FFD300",
}


# This is specifically NOT the MATLAB color palette.
B_PAL_MAP = {
    "blue": "#0072c2",
    "orange": "#d85319",
    "yellow": "#f0b120",
    "purple": "#7d2f8e",
    "green": "#75ac30",
    "cyan": "#4cbdf0",
    "red": "#a21430",
    "pink": "#f77ec7",
    "grey": "#a8a495",
}


B_PALETTE = sns.color_palette(list(B_PAL_MAP.values()))
