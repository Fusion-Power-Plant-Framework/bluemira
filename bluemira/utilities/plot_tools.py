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
A collection of plotting tools.
"""

import os
import re
from typing import Union

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import PathPatch3D

import bluemira.display.error as bm_display_error
from bluemira.base.components import Component
from bluemira.base.constants import GREEK_ALPHABET, GREEK_ALPHABET_CAPS
from bluemira.base.file import get_bluemira_path
from bluemira.geometry.coordinates import check_ccw, rotation_matrix_v1v2
from bluemira.geometry.placement import BluemiraPlacement

__all__ = [
    "str_to_latex",
    "make_gif",
    "save_figure",
    "coordinates_to_path",
    "Plot3D",
    "BluemiraPathPatch3D",
]


def gsymbolify(string):
    """
    Convert a string to a LaTEX printable greek letter if detected.

    Parameters
    ----------
    string: str
        The string to add Greek symbols to

    Returns
    -------
    string: str
        The modified string. Returns input if no changes made
    """
    if string in GREEK_ALPHABET or string in GREEK_ALPHABET_CAPS:
        return "\\" + string
    else:
        return string


def str_to_latex(string):
    """
    Create a new string which can be printed in LaTEX nicely.

    Parameters
    ----------
    string: str
        The string to be converted

    Returns
    -------
    string: str
        The mathified string

    'I_m_p' ==> '$I_{m_{p}}$'
    """
    s = string.split("_")
    s = [gsymbolify(sec) for sec in s]
    ss = "".join(["_" + "{" + lab for i, lab in enumerate(s[1:])])
    return "$" + s[0] + ss + "}" * (len(s) - 1) + "$"


def make_gif(folder, figname, formatt="png", clean=True):
    """
    Make a GIF image from a set of images with similar names in a folder.
    Figures are sorted in increasing order based on a trailing number, e.g.
    'figure_A[1, 2, 3, ..].png'
    Cleans up the temporary figure files (deletes!)
    Creates a GIF file in the folder directory

    Parameters
    ----------
    folder: str
        Full path folder name
    figname: str
        Figure name prefix
    formatt: str (default = 'png')
        Figure filename extension
    clean: bool (default = True)
        Delete figures after completion?
    """
    ims = []
    for filename in os.listdir(folder):
        if filename.startswith(figname) and filename.endswith(formatt):

    find_digit = re.compile("(\\d+)")
    ims = sorted(ims, key=lambda x: int(find_digit.findall(x)[-1]))
    images = [imageio.imread(fp) for fp in ims]
    if clean:
        for fp in ims:
            os.remove(fp)
    gifname = os.path.join(folder, figname) + ".gif"
    kwargs = {"duration": 0.5, "loop": 3}
    imageio.mimsave(gifname, images, "GIF-FI", **kwargs)


def save_figure(fig, name, save=False, folder=None, dpi=600, formatt="png", **kwargs):
    """
    Saves a figure to the directory if save flag active
    """
    if save is True:
        if folder is None:
            folder = get_bluemira_path("plots", subfolder="data")
        name = os.sep.join([folder, name]) + "." + formatt
        if os.path.isfile(name):
            os.remove(name)  # f.savefig will otherwise not overwrite
        fig.savefig(name, dpi=dpi, bbox_inches="tight", format=formatt, **kwargs)


def ring_coding(n):
    """
    The codes will be all "LINETO" commands, except for "MOVETO"s at the
    beginning of each subpath
    """
    codes = np.ones(n, dtype=Path.code_type) * Path.LINETO
    codes[0] = Path.MOVETO
    return codes


def coordinates_to_path(x, z):
    """
    Convert coordinates to path vertices.
    """
    if not check_ccw(x, z):
        x = x[::-1]
        z = z[::-1]
    vertices = np.array([x, z]).T
    codes = ring_coding(len(x))
    return Path(vertices, codes)


def set_component_view(comp: Component, placement: Union[str, BluemiraPlacement]):
    if placement not in ["xy", "xz", "yz"] and not isinstance(
        placement, BluemiraPlacement
    ):
        raise bm_display_error.DisplayError(
            f"Not a valid view {placement} - select either xy, xz, yz, "
            f"or a BluemiraPlacement"
        )

    comp.plot_options.view = placement
    for child in comp.children:
        set_component_view(child, placement)


class Plot3D(Axes3D):
    """
    Cheap and cheerful
    """

    def __init__(self):
        fig = plt.figure(figsize=[14, 14])
        super().__init__(fig, auto_add_to_figure=False)
        fig.add_axes(self)
        # \n to offset labels from axes
        self.set_xlabel("\n\nx [m]")
        self.set_ylabel("\n\ny [m]")
        self.set_zlabel("\n\nz [m]")


class BluemiraPathPatch3D(PathPatch3D):
    """
    Class for a 3-D PathPatch which can actually be filled properly!

    Parameters
    ----------
    path: matplotlib path::Path object
        The path object to plot in 3-D
    normal: iterable(3)
        The 3-D normal vector of the face
    translation: iterable(3)
        Translation vector to apply to the face
    color: str
        The color to plot the fill
    """

    # Thank you StackOverflow
    # https://stackoverflow.com/questions/18228966/how-can-matplotlib-2d-patches-be-transformed-to-3d-with-arbitrary-normals
    def __init__(self, path, normal, translation=None, color="b", **kwargs):
        # Initialise the patch first, or we can get into nasty recursive
        # calls in __getattr__
        self._patch2d = PathPatch(path, color=color, **kwargs)

        Patch.__init__(self, **kwargs)

        if translation is None:
            translation = [0, 0, 0]

        self._path2d = path
        self._code3d = path.codes
        self._facecolor3d = self._patch2d.get_facecolor

        r_matrix = rotation_matrix_v1v2(normal, (0, 0, 1))
        t_matrix = np.array(translation)

        points = path.vertices

        new_points = np.array([np.dot(r_matrix, np.array([x, y, 0])) for x, y in points])

        self._segment3d = new_points + t_matrix

    def __getattr__(self, key):
        """
        Transfer the key getattr to underlying PathPatch object.
        """
        return getattr(self._patch2d, key)
