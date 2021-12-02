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
Colour palettes
"""

from typing import Union
import numpy as np
import seaborn as sns
import matplotlib.colors as colors


class ColorPalette:
    def __init__(self, palette_map):
        self._dict = palette_map
        self._palette = sns.color_palette(list(palette_map.values()))

    def __setitem__(self, idx_or_key: Union[int, str], value):
        if isinstance(idx_or_key, int):
            self._palette[idx_or_key] = value
            key = list(self._dict)[idx_or_key]
            self._dict[key] = value

        elif isinstance(idx_or_key, str):
            self._dict[idx_or_key] = value
            idx = list(self._dict).index(idx_or_key)
            self._palette[idx] = value

    def __getitem__(self, idx_or_key: Union[int, str]):
        if isinstance(idx_or_key, int):
            return self._palette[idx_or_key]
        elif isinstance(idx_or_key, str):
            return self._dict[idx_or_key]

    def _repr_html(self):
        s = 55
        n = len(self)

        sub_pals = []
        for k, v in self._dict.items():
            if isinstance(v, ColorPalette):
                sub_pals.append(v)

        if sub_pals:
            m = max([len(sp) for sp in sub_pals])
        else:
            m = 1

        html = f'<svg  width="{n * s}" height="{m * s}">'

        for i, c in enumerate(self._palette):
            if isinstance(c, ColorPalette):
                for j, sc in enumerate(c):
                    sc = colors.to_hex(sc)
                    html += (
                        f'<rect x="{i * s}" y="{j * s}" width="{s}" height="{s}" style="fill:{sc};'
                        'stroke-width:2;stroke:rgb(255,255,255)"/>'
                    )
            else:
                c = colors.to_hex(c)
                html += (
                    f'<rect x="{i * s}" y="0" width="{s}" height="{s}" style="fill:{c};'
                    'stroke-width:2;stroke:rgb(255,255,255)"/>'
                )

        html += "</svg>"
        return html

    def __repr__(self):
        from IPython.core.interactiveshell import InteractiveShell

        if not InteractiveShell.initialized():
            return self._palette.__repr__()

        from IPython.core.display import display, HTML

        display(HTML(self._repr_html()))
        return ""

    def __len__(self):
        return len(self._dict)


def make_rgb_alpha(rgb, alpha, background_rgb=(1, 1, 1)):
    """
    Adds a transparency to a RGB color tuple

    Parameters
    ----------
    rgb: tuple(float, float, float) 0<=float<=1
        Tuple of RGB floats
    alpha: 0<=float<=1
        Transparency as a fraction
    background_rgb: tuple(float, float, float) 0<=float<=1
        Background colour (default = white)

    Returns
    -------
    rgba: tuple(float, float, float) 0<=float<=1
        The RGB tuple accounting for transparency
    """
    rgb = colors.to_rgb(rgb)
    return [alpha * c1 + (1 - alpha) * c2 for (c1, c2) in zip(rgb, background_rgb)]


def make_alpha_palette(color, n_colors, background_rgb=(1, 1, 1)):
    color_name = colors.to_hex(color)
    alphas = np.linspace(0, 1, n_colors + 1)[1:-1][::-1]
    color_values = [color] + [make_rgb_alpha(color, a, background_rgb) for a in alphas]
    palette_map = {f"{color_name}_{i}": color for i, color in enumerate(color_values)}
    return ColorPalette(palette_map)


# This is specifically NOT the MATLAB color palette.
BLUEMIRA_PALETTE = ColorPalette(
    {
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
)


LONDON_PALETTE = ColorPalette(
    {
        "black": "#000000",
        "blue": "#003688",
        "red": "#E32017",
        "green": "#00782A",
        "purple": "#9B0056",
        "light_blue": "#0098D4",
        "orange": "#EE7C0E",
        "yellow": "#FFD300",
        "pink": "#F3A9BB",
        "grey": "#A0A5A9",
        "brown": "#B36305",
        "turquoise": "#95CDBA",
    }
)


BLUE_PALETTE = ColorPalette(
    {
        "BB": "#4a98c9",  # Breeding blanket
        "DIV": "#d0e1f2",  # Divertor
        "VV": "#b7d4ea",  # Vacuum vessel
        "TS": "#e3eef9",  # Thermal shield
        "TF": "#084a91",  # Toroidal field system
        "PF": "#1764ab",  # Poloidal field system
        "CR": "#2e7ebc",  # Cryostat vacuum vessel
        "RS": "#94c4df",  # Radiation shield
        "PL": "#cc8acc",  # Plasma
    }
)

BLUE_PALETTE["BB"] = make_alpha_palette(BLUE_PALETTE["BB"], 8)
BLUE_PALETTE["PF"] = make_alpha_palette(BLUE_PALETTE["PF"], 2)
BLUE_PALETTE["TF"] = make_alpha_palette(BLUE_PALETTE["TF"], 3)
