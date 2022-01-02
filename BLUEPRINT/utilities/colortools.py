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
Color utilities
"""
import matplotlib.colors as col
import numpy as np
import seaborn as sns
from matplotlib.colors import hex2color


def facecolor_kwargs(**kwargs):
    """
    Deals with the facecolor kwarg
    returns a color cycle
    """
    if "facecolor" in kwargs:
        if len(np.shape(kwargs["facecolor"])) == 1:
            if type(kwargs["facecolor"][0]) is str:
                colors = [col.to_rgb(kwargs["facecolor"][0])]
            else:
                colors = kwargs["facecolor"]
        else:
            colors = kwargs["facecolor"]
    elif "facecolor" not in kwargs:
        colors = ["grey"]
    return colors


def color_kwargs(**kwargs):
    """
    Handles color kwargs for plotting purposes.
    Must deal with seaborn, bokeh, and matplotlib
    Must handle color cycles as well as point colors
    Must handle facecolor, edgecolor, and color kwargs
    """
    if "color" in kwargs:  # Deprecated 'color' kwargs ==> 'facecolor'
        kwargs["facecolor"] = kwargs["color"]
        del kwargs["color"]
    if "facecolor" in kwargs:
        kwargs["facecolor"] = facecolor_kwargs(**kwargs)
    if "edgecolor" in kwargs:
        pass
    elif "palette" and "n" in kwargs:
        p = sns.color_palette(kwargs["palette"], kwargs["n"])
        colors = [p[i] for i in range(kwargs["n"])]
        kwargs["facecolor"] = colors
    return kwargs


def map_palette(paldict, mapping_dict):
    """
    Palette mapping utility.
    """

    def convert(value):
        if isinstance(value, str):
            return mapping_dict[value]
        else:
            return value

    paldict_n = {}
    for k, v in paldict.items():
        v = convert(v)
        if isinstance(v, list):
            vnew = []
            for val in v:
                vnew.append(convert(val))
            v = vnew
        paldict_n[k] = v
    return paldict_n


def force_rgb(col_dict):
    """
    Force a color dictionary to RGB values.

    Parameters
    ----------
    col_dict: dict
        The dictionary of color values (hexstr, tuple, array, list)

    Returns
    -------
    c: dict
        The dictionary with RGB color values
    """

    def convert(value):
        if isinstance(value, str):
            return hex2color(value)
        else:
            return value

    c = {}
    for k, v in col_dict.items():
        v = convert(v)
        if isinstance(v, list):
            vnew = []
            for val in v:
                vnew.append(convert(val))
            v = vnew
        c[k] = v
    return c


def col_0_to_256(color):
    """
    Conversion from matplotlib rgb color tuple (0->1) to classical (0->255)

    Parameters
    ----------
    color: tuple(3)
        The RGB tuple in 0 <-> 255 units

    Returns
    -------
    col: tuple(3)
        The RGB tuple in 0 <-> 1 units
    """
    return [int(i * 255) for i in color]


def make_rgb_alpha(rgb, alpha, background_rgb=None):
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
    if background_rgb is None:
        background_rgb = [1, 1, 1]
    if isinstance(rgb, str):
        rgb = hex2color(rgb)
    return [alpha * c1 + (1 - alpha) * c2 for (c1, c2) in zip(rgb, background_rgb)]


# NOTE: This is what happens when you don't sort things properly..
# you duplicate your own shit.
def shift_rgb_color(rgb, alpha):
    """
    Tints or shades an RGB color tuple

    Parameters
    ----------
    rgb: tuple(3)
        The RGB color tuple in matplotlib units (0->1)
    alpha: float -1 < f < 1
        The factor by which to shift the RGB color:
            Negative values will shade the color.
            Positive values will tint the color.

    Returns
    -------
    rgb_new: tuple(3)
        The shifter RGB color tuple
    """
    if isinstance(rgb, str):
        rgb = hex2color(rgb)
    if alpha < -1 or alpha > 1:
        from bluemira.base.look_and_feel import bluemira_warn

        bluemira_warn("shift_RGBcolor: alpha ! belong [-1, 1]")
        alpha = np.clip(alpha, -1, 1)
    rgb = np.array(rgb)
    if alpha <= 0:
        v = 0
    else:
        v = 1
    rgb_new = rgb + (v - rgb) * abs(alpha)
    return tuple(rgb_new)
