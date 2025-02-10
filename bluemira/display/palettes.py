# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Colour palettes
"""

from __future__ import annotations

from itertools import cycle, zip_longest
from typing import TYPE_CHECKING

import numpy as np
import seaborn as sns
from matplotlib import colors

from bluemira.base.constants import ANSI_COLOR, EXIT_COLOR

if TYPE_CHECKING:
    from matplotlib.typing import ColorType


class ColorPalette:
    """
    Color palette object, wrapping some seaborn functionality.

    Parameters
    ----------
    palette_map:
        Dictionary of color names to any object matplotlib will recognise as a color
    """

    def __init__(self, palette_map: dict[str, ColorType]):
        self._dict = palette_map
        color_list = []
        for v in palette_map.values():
            if isinstance(v, str | tuple):
                color_list.append(v)
            else:
                color_list.extend(v._palette)
        self._palette = sns.color_palette(color_list)
        self._cycle = cycle(color_list)

    def keys(self):
        """
        Returns
        -------
        :
            Keys of ColorPalette
        """
        return self._dict.keys()

    def __next__(self):
        """
        Returns
        -------
        :
            the next color in the ColorPalette
        """
        return next(self._cycle)

    def __setitem__(self, idx_or_key: int | str, value: ColorType | ColorPalette):
        """
        Set an item in the ColorPalette by index or key

        Parameters
        ----------
        idx_or_key:
            Index or key of the ColorPalette
        value:
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

    def __getitem__(self, idx_or_key: int | str) -> ColorType | ColorPalette | None:
        """
        Get an item in the ColorPalette by index or key

        Parameters
        ----------
        idx_or_key:
            Index or key of the ColorPalette

        Returns
        -------
        value:
            The value. Note that this can be another ColorPalette
        """
        if isinstance(idx_or_key, int):
            return self._palette[idx_or_key]
        if isinstance(idx_or_key, str):
            item = self._dict[idx_or_key]
            return (
                item
                if isinstance(item, type(self))
                else type(self)({idx_or_key: self._dict[idx_or_key]})
            )
        return None

    def _hex_horizontal(self) -> list[str] | list[list[str]]:
        """
        Returns
        -------
        :
            A list of strings representing a horizontal row of hex colors,
            or a list of lists of strings where each inner list represents
            a row of hex colors.
        """
        _hex = self.as_hex()
        if isinstance(_hex, str):
            _hex = [_hex]
        elif any(isinstance(h, list) for h in _hex):
            for i, h in enumerate(_hex):
                if isinstance(h, str):
                    _hex[i] = [h]
            _hex = list(map(list, zip_longest(*_hex, fillvalue="  ")))
        return _hex

    def _repr_html(self) -> str:
        def html_str(_hex: list[str] | list[list[str]], y: int = 0) -> str:
            """
            Returns
            -------
            :
                an HTML string of rectangles representing hex colors.
            """
            string = ""
            x = 0
            for col in _hex:
                if isinstance(col, list):
                    string += html_str(_hex=col, y=y)
                    y += 1
                else:
                    if col != "  ":
                        string += (
                            f'<rect x="{x * s}" y="{y * s}"'
                            f' width="{s}" height="{s}" style="fill:{col};'
                            'stroke-width:2;stroke:rgb(255,255,255)"/>'
                        )
                    x += 1

            return string

        s = 55
        hex_str = self._hex_horizontal()
        m = len(hex_str) if any(isinstance(h, list) for h in hex_str) else 1
        colours = html_str(hex_str)
        return f'<svg  width="{(len(self)) * s}" height="{m * s}">{colours}</svg>'

    def _repr_colour_str(self, _hex: list[str] | list[list[str]]) -> str:
        """

        Returns
        -------
        :
            colourful representation in terminal
        """
        string = ""
        for col in _hex:
            if isinstance(col, list):
                string += self._repr_colour_str(_hex=col)
            elif col != "  ":
                string += background_colour_string(col, sqlen=2)
            else:
                string += col
        return f"{string}\n"

    def __repr__(self) -> str:
        """

        Returns
        -------
        :
            a representation of the ColorPalette
        """
        try:
            g_ipy = get_ipython()
            if "terminal" in str(type(g_ipy)) or g_ipy is None:
                return self._repr_colour_str(self._hex_horizontal()).strip(" \n")
        except NameError:
            return self._repr_colour_str(self._hex_horizontal()).strip(" \n")

        from IPython.core.display import HTML, display  # noqa: PLC0415

        display(HTML(self._repr_html()))
        return ""

    def __len__(self) -> int:
        """
        Returns
        -------
        :
            the length of the ColorPalette
        """
        return len(self._palette)

    def as_hex(self) -> list[str] | list[list[str]] | str:
        """

        Returns
        -------
        :
            the hex representation of the palette
        """
        hex_list = []
        if isinstance(self._palette, list):
            for pp in self._palette:
                if isinstance(pp, tuple):
                    hex_list.append(colors.to_hex(pp))
                elif isinstance(pp, type(self)):
                    hex_list.append(pp.as_hex())
        elif isinstance(self._palette, tuple):
            hex_list.append(colors.to_hex(self._palette))
        else:
            hex_list.append(self._palette)
        return hex_list[0] if len(hex_list) == 1 else hex_list


def background_colour_string(hexstring: str, sqlen=2) -> str:
    """
    Returns
    -------
    :
        ANSI background colour string for hex colour
    """
    hexstring = hexstring.strip("#")
    a, b, c = (1, 2, 3) if len(hexstring) == 3 else (2, 4, 6)  # noqa: PLR2004
    return (
        f"\033[48:2::{int(hexstring[:a], 16)}:"
        f"{int(hexstring[a:b], 16)}:"
        f"{int(hexstring[b:c], 16)}m{' ' * sqlen}\033[49m"
    )


def make_rgb_alpha(
    rgb: tuple[float, ...],
    alpha: float,
    background_rgb: tuple[float, ...] = (1.0, 1.0, 1.0),
) -> tuple[float, ...]:
    """
    Adds a transparency to a RGB color tuple

    Parameters
    ----------
    rgb:
        Tuple of 3 RGB floats  (0<=float<=1)
    alpha:
        Transparency as a fraction  (0<=float<=1)
    background_rgb:
        3 RGB floats (0<=float<=1), background colour (default = white)

    Returns
    -------
    The RGB tuple accounting for transparency
    """
    return tuple(
        alpha * c1 + (1 - alpha) * c2
        for (c1, c2) in zip(rgb, background_rgb, strict=False)
    )


def make_alpha_palette(
    color, n_colors: int, background_rgb: ColorType = "white"
) -> ColorPalette:
    """
    Make a palette from a color by varying alpha.

    Parameters
    ----------
    color: Any
        Palette base color. Anything matplotlib will recognise as a color
    n_colors:
        Numer of colors to make in the palette
    background_rgb:
        Background color. Anything matplotlib will recognise as a color

    Returns
    -------
    :
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
    return ColorPalette({
        f"{color_name}_{i}": col_val for i, col_val in enumerate(color_values)
    })


def _print_colour(string: str, colour: str) -> str:
    """
    Create text to print. NOTE: Does not call print command

    Parameters
    ----------
    string:
        The text to colour
    colour:
        The colour to make the colour-string for

    Returns
    -------
    :
        The string with ANSI colour decoration
    """
    return f"{ANSI_COLOR[colour]}{string}{EXIT_COLOR}"


# This is specifically NOT the MATLAB color palette.
BLUEMIRA_PALETTE = ColorPalette({
    "blue": "#0072c2",
    "orange": "#d85319",
    "yellow": "#f0b120",
    "purple": "#7d2f8e",
    "green": "#75ac30",
    "cyan": "#4cbdf0",
    "red": "#a21430",
    "pink": "#f77ec7",
    "grey": "#a8a495",
})


LONDON_PALETTE = ColorPalette({
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
})


BLUE_PALETTE = ColorPalette({
    "BB": "#4a98c9",  # Breeding blanket
    "DIV": "#d0e1f2",  # Divertor
    "VV": "#b7d4ea",  # Vacuum vessel
    "TS": "#e3eef9",  # Thermal shield
    "TF": "#084a91",  # Toroidal field system
    "PF": "#1764ab",  # Poloidal field system
    "CR": "#2e7ebc",  # Cryostat vacuum vessel
    "RS": "#94c4df",  # Radiation shield
    "PL": "#cc8acc",  # Plasma
})


BLUE_PALETTE["VV"] = make_alpha_palette(BLUE_PALETTE["VV"], 2)
BLUE_PALETTE["BB"] = make_alpha_palette(BLUE_PALETTE["BB"], 8)
BLUE_PALETTE["DIV"] = make_alpha_palette(BLUE_PALETTE["DIV"], 5)
BLUE_PALETTE["PF"] = make_alpha_palette(BLUE_PALETTE["PF"], 4)
BLUE_PALETTE["TF"] = make_alpha_palette(BLUE_PALETTE["TF"], 3)
BLUE_PALETTE["CR"] = make_alpha_palette(BLUE_PALETTE["CR"], 2)
BLUE_PALETTE["RS"] = make_alpha_palette(BLUE_PALETTE["RS"], 2)
BLUE_PALETTE["TS"] = make_alpha_palette(BLUE_PALETTE["TS"], 2)
