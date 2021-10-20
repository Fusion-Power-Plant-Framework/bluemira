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
Mostly just aesthetic and ambiance functions
"""
from itertools import cycle
import numpy as np
import subprocess  # noqa (S404)
import seaborn as sns
from matplotlib.colors import hex2color

KEY_TO_PLOT = False


def color_kwargs(**kwargs):
    """
    Handle matplotlib color keyword arguments.

    Parameters
    ----------
    kwargs

    Returns
    -------
    colors: cycle
        The cycle of colors to use in plotting
    """
    if "color" in kwargs:
        if len(np.shape(kwargs["color"])) == 1:
            if type(kwargs["color"][0]) is str:
                colors = [hex2color(kwargs["color"][0])]
            else:
                colors = [kwargs["color"]]
        else:
            colors = kwargs["color"]
    elif "palette" and "n" in kwargs:
        p = sns.color_palette(kwargs["palette"], kwargs["n"])
        colors = [p[i] for i in range(kwargs["n"])]
    elif "color" not in kwargs:
        colors = ["grey"]
    colors = cycle(colors)
    return colors


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
