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
A collection of plotting tools.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import Patch, PathPatch
from mpl_toolkits.mplot3d.art3d import PathPatch3D
from mpl_toolkits.mplot3d import Axes3D
from bluemira.geometry._deprecated_tools import rotation_matrix_v1v2, check_ccw


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
        Patch.__init__(self, **kwargs)

        if translation is None:
            translation = [0, 0, 0]

        self._patch2d = PathPatch(path, color=color, **kwargs)
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
