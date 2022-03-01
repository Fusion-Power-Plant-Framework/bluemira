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
Generic plot utilities, figure and gif operations
"""
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import PathPatch3D
from scipy.interpolate import interp1d

from bluemira.base.file import get_bluemira_path
from bluemira.geometry.coordinates import rotation_matrix_v1v2


def makegif(folder, figname, formatt="png", clean=True):
    """
    Makes a GIF image from a set of images with similar names in a folder
    Cleans up the temporary figure files (deletes!)
    Creates a GIF file in the folder directory

    Parameters
    ----------
    folder: str
        Full path folder name
    figname: str
        Figure name prefix. E.g. 'figure_A'[1, 2, 3, ..]
    formatt: str (default = 'png')
        Figure filename extension
    clean: bool (default = True)
        Delete figures after completion?
    """
    ims = []
    for filename in os.listdir(folder):
        if filename.startswith(figname):
            if filename.endswith(formatt):
                fp = os.path.join(folder, filename)
                ims.append(fp)
    ims = sorted(ims)
    images = [imageio.imread(fp) for fp in ims]
    if clean:
        for fp in ims:
            os.remove(fp)
    gifname = os.path.join(folder, figname) + ".gif"
    kwargs = {"duration": 0.5, "loop": 3}
    imageio.mimsave(gifname, images, "GIF-FI", **kwargs)


def savefig(f, name, save=False, folder=None, dpi=600, formatt="png", **kwargs):
    """
    Saves a figure to the directory if save flag active
    Meant to be used to switch on/off output figs from main BLUEPRINT run,
    typically flagged in reactor.py
    """
    if save is True:
        if folder is None:
            folder = get_bluemira_path("plots", subfolder="data/BLUEPRINT")
        name = os.sep.join([folder, name]) + "." + formatt
        if os.path.isfile(name):
            os.remove(name)  # f.savefig will otherwise not overwrite
        f.savefig(name, dpi=dpi, bbox_inches="tight", format=formatt, **kwargs)
    else:
        pass


def weather_front(d2, n=10, scale=True, ends=True, **kwargs):
    """
    Plots a "weather front" on a design space plot
    """
    if d2 is None:
        return
    fig = kwargs.get("fig", plt.gcf())
    ax = kwargs.get("ax", plt.gca())
    c = kwargs.get("color", "r")
    x, y = d2.T
    f = interp1d(x, y)
    xn = np.linspace(x[0], x[-1], 50)
    yn = f(xn)
    # xn, yn = x, y
    i = int(len(yn) / n) - 1

    # xn, yn = xn[::i], yn[::i]
    dx = np.gradient(xn)
    dy = np.gradient(yn)
    if scale:
        a_r = fig.get_figheight() / fig.get_figwidth()
        s = a_r / ax.get_data_ratio()
    else:  # TODO: Check / clean
        s = 1
    asp = 1
    dy *= s
    dx = dx / asp
    mag = np.sqrt(dx**2 + dy**2)
    dx /= mag
    dy /= mag
    x = xn[i::i]
    y = yn[i::i]
    dx = dx[i::i]
    dy = dy[i::i]
    if ends:
        x = np.append(x, [d2.T[0][0], d2.T[0][-1]])
        y = np.append(y, [d2.T[1][0], d2.T[1][-1]])
        dx = np.append(dx, [-1, 0])
        dy = np.append(dy, [0, 1])

    ax.plot(*d2.T, "s", marker=None, ls="-", color=c)
    last = ax.quiver(x, y, dy, -dx, color=c)
    last.set_zorder(20)  # force quiver on top


def ring_coding(ob):
    """
    The codes will be all "LINETO" commands, except for "MOVETO"s at the
    beginning of each subpath
    """
    n = len(ob.coords)
    codes = np.ones(n, dtype=Path.code_type) * Path.LINETO
    codes[0] = Path.MOVETO
    return codes


def pathify(polygon):
    """
    Convert coordinates to path vertices. Objects produced by Shapely's
    analytic methods have the proper coordinate order, no need to sort.
    """
    vertices = np.concatenate(
        [polygon.exterior.coords] + [r.coords for r in polygon.interiors]
    )
    vertices = vertices[:, 0:2]
    codes = np.concatenate(
        [ring_coding(polygon.exterior)] + [ring_coding(r) for r in polygon.interiors]
    )
    return Path(vertices, codes)


class Plot3D(Axes3D):
    """
    Cheap and cheerful
    """

    def __init__(self):
        fig = plt.figure(figsize=[14, 14])
        super().__init__(fig)
        # \n to offset labels from axes
        self.set_xlabel("\n\nx [m]")
        self.set_ylabel("\n\ny [m]")
        self.set_zlabel("\n\nz [m]")


class BPPathPatch3D(PathPatch3D):
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
        Patch.__init__(self, **kwargs)

        if translation is None:
            translation = [0, 0, 0]

        self._patch2d = PathPatch(path, color=color, **kwargs)
        self._path2d = path
        self._code3d = path.codes
        self._facecolor3d = self._patch2d.get_facecolor

        r_matrix = self.rotation_matrix(normal, (0, 0, 1))
        t_matrix = np.array(translation)

        points = path.vertices

        new_points = np.array([np.dot(r_matrix, np.array([x, y, 0])) for x, y in points])

        self._segment3d = new_points + t_matrix

    def __getattr__(self, key):
        """
        Transfer the key getattr to underlying PathPatch object.
        """
        if key != "_patch2d":
            return getattr(self._patch2d, key)
        else:
            return self.__dict__.get("_patch2d", None)

    @staticmethod
    def rotation_matrix(v1, v2):
        """
        Get a rotation matrix based off two vectors.
        """
        return rotation_matrix_v1v2(v1, v2)
