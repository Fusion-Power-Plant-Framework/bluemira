# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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

from itertools import cycle
from typing import Tuple, Union

import matplotlib.colors as colors
import numpy as np
import seaborn as sns


class ColorPalette:
    """
    Color palette object, wrapping some seaborn functionality.

    Parameters
    ----------
    palette_map: Dict[str: Any]
        Dictionary of color names to any object matplotlib will recognise as a color
    """

    def __init__(self, palette_map):
        self._dict = palette_map
        color_list = []
        for v in palette_map.values():
            if isinstance(v, (str, tuple)):
                color_list.append(v)
            else:
                color_list.extend(v._palette)
        self._palette = sns.color_palette(color_list)
        self._cycle = cycle(color_list)

    def keys(self):
        """Keys of ColorPalette"""
        return self._dict.keys()

    def __next__(self):
        """
        Get the next color in the ColorPalette
        """
        return next(self._cycle)

    def __setitem__(self, idx_or_key: Union[int, str], value):
        """
        Set an item in the ColorPalette by index or key

        Parameters
        ----------
        idx_or_key: Union[int, str]
            Index or key of the ColorPalette
        value: Union[ColorType, ColorPalette]
            The value to set. Note that this can be another ColorPalette
        """
        if isinstance(idx_or_key, int):
            self._palette[idx_or_key] = value
            key = list(self._dict)[idx_or_key]
            self._dict[key] = value

        elif isinstance(idx_or_key, str):
            self._dict[idx_or_key] = type(self)({idx_or_key: value})
            idx = list(self._dict).index(idx_or_key)
            self._palette[idx] = type(self)({idx_or_key: value})

    def __getitem__(self, idx_or_key: Union[int, str]):
        """
        Get an item in the ColorPalette by index or key

        Parameters
        ----------
        idx_or_key: Union[int, str]
            Index or key of the ColorPalette

        Returns
        -------
        value: Union[ColorType, ColorPalette]
            The value. Note that this can be another ColorPalette
        """
        if isinstance(idx_or_key, int):
            return self._palette[idx_or_key]
        elif isinstance(idx_or_key, str):
            item = self._dict[idx_or_key]
            return (
                item
                if isinstance(item, type(self))
                else type(self)({idx_or_key: self._dict[idx_or_key]})
            )

    def _repr_html(self) -> str:
        def html_rect(x: int, y: int, size: int, fill: str) -> str:
            return (
                f'<rect x="{x}" y="{y}" width="{size}" height="{size}" style="fill:{fill};'
                'stroke-width:2;stroke:rgb(255,255,255)"/>'
            )

        s = 55
        n = len(self)
        sub_pals = [v for v in self._dict.values() if isinstance(v, type(self))]
        m = max([len(sp) for sp in sub_pals]) if sub_pals else 1

        html = f'<svg  width="{n * s}" height="{m * s}">'

        for i, c in enumerate(self._palette):
            if isinstance(c, type(self)):
                for j, sc in enumerate(c):
                    html += html_rect(i * s, j * s, s, colors.to_hex(sc))
            else:
                html += html_rect(i * s, 0, s, colors.to_hex(c))

        html += "</svg>"
        return html

    def __repr__(self) -> str:
        """
        Create a representation of the ColorPalette
        """
        try:
            g_ipy = get_ipython()
            if "terminal" in str(type(g_ipy)) or g_ipy is None:
                return self._repr_colour_str()
        except NameError:
            return self._repr_colour_str()

        from IPython.core.display import HTML, display

        display(HTML(self._repr_html()))
        return ""

    def _repr_colour_str(self) -> str:
        """Create colourful representation in terminal"""
        if isinstance(self._palette, list):
            string = ""
            for en, pp in enumerate(self._palette):
                if isinstance(pp, tuple):
                    string += background_colour_string(colors.to_hex(pp))
                elif isinstance(pp, type(self)):
                    string += f"{pp._repr_colour_str()}\n"
        elif isinstance(self._palette, tuple):
            string = background_colour_string(colors.to_hex(self._palette))
        else:
            string = background_colour_string(self._palette)

        return string

    def __len__(self) -> int:
        """Get the length of the ColorPalette"""
        return len(self._dict)


def background_colour_string(hexstring: str) -> str:
    """Create ANSI background colour string for hex colour"""
    hexstring = hexstring.strip("#")
    a, b, c = (1, 2, 3) if len(hexstring) == 3 else (2, 4, 6)
    return (
        f"\033[48:2::{int(hexstring[:a], 16)}:"
        f"{int(hexstring[a:b], 16)}:"
        f"{int(hexstring[b:c], 16)}m \033[49m"
    )


def make_rgb_alpha(
    rgb: Tuple[float, ...],
    alpha: float,
    background_rgb: Tuple[float, ...] = (1.0, 1.0, 1.0),
) -> Tuple[float, ...]:
    """
    Adds a transparency to a RGB color tuple

    Parameters
    ----------
    rgb:
        Tuple of 3 RGB floats  (0<=float<=1)
    alpha:
        Transparency as a fraction  (0<=float<=1)
    background_rgb
        3 RGB floats (0<=float<=1), background colour (default = white)

    Returns
    -------
    The RGB tuple accounting for transparency
    """
    return tuple(alpha * c1 + (1 - alpha) * c2 for (c1, c2) in zip(rgb, background_rgb))


def make_alpha_palette(color, n_colors: int, background_rgb="white") -> ColorPalette:
    """
    Make a palette from a color by varying alpha.

    Parameters
    ----------
    color: Any
        Palette base color. Anything matplotlib will recognise as a color
    n_colors
        Numer of colors to make in the palette
    background_rgb: Any
        Background color. Anything matplotlib will recognise as a color

    Returns
    -------
    Colour palette from the base color. The first color is the base color
    """
    if isinstance(color, ColorPalette):
        color = color._palette[0]

    color_name = colors.to_hex(color)
    color_rgb = colors.to_rgb(color)
    background_rgb = colors.to_rgb(background_rgb)

    color_values = [color_rgb] + [
        make_rgb_alpha(color_rgb, alpha, background_rgb)
        for alpha in np.linspace(0, 1, n_colors + 1)[1:-1][::-1]
    ]
    return ColorPalette(
        {f"{color_name}_{i}": col_val for i, col_val in enumerate(color_values)}
    )


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


BLUE_PALETTE["VV"] = make_alpha_palette(BLUE_PALETTE["VV"], 2)
BLUE_PALETTE["BB"] = make_alpha_palette(BLUE_PALETTE["BB"], 8)
BLUE_PALETTE["DIV"] = make_alpha_palette(BLUE_PALETTE["DIV"], 5)
BLUE_PALETTE["PF"] = make_alpha_palette(BLUE_PALETTE["PF"], 4)
BLUE_PALETTE["TF"] = make_alpha_palette(BLUE_PALETTE["TF"], 3)
BLUE_PALETTE["CR"] = make_alpha_palette(BLUE_PALETTE["CR"], 2)
BLUE_PALETTE["RS"] = make_alpha_palette(BLUE_PALETTE["RS"], 2)
BLUE_PALETTE["TS"] = make_alpha_palette(BLUE_PALETTE["TS"], 2)
