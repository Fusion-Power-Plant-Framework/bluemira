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
Plotting functionality for the bluemira geometry module.
WARNING: This module is only drafted. It must be updated.
"""

# import graphical lib
import matplotlib.pyplot as plt

# import bluemira lib
from .wire import BluemiraWire
from .face import BluemiraFace

from bluemira.utilities.plot_tools import Plot3D

DEFAULT = {}
DEFAULT["poptions"] = {"s": 30, "facecolors": "blue", "edgecolors": "black"}
DEFAULT["woptions"] = {"color": "black", "linewidth": "2"}
DEFAULT["foptions"] = {"color": "red"}


def plot_wire(
    wires,
    axis=None,
    show: bool = False,
    ndiscr: int = 100,
    poptions: dict = DEFAULT["poptions"],
    woptions: dict = DEFAULT["woptions"],
    *args,
    **kwargs,
):
    """Plot a BluemiraWire or list(BluemiraWire)

    Parameters
    ----------
    wires : BluemiraWire, list(BluemiraWire)
        wires to be plotted.
    axis : matplot.axis
        matplotlib axis for plotting. (Default value = None)
    show: bool
         matplotlib option. (Default value = False)
    ndiscr: int
        number of discretization points (Default value = 100)
    poptions: dict
         (Default value = DEFAULT['poptions'])
    woptions: dict
         (Default value = DEFAULT['woptions'])
    *args :
    **kwargs :

    Returns
    -------
    axis : matplot.axis
        axis used for multiplots.

    Raises
    ------
    ValueError
        in case the obj is not a Bluemira.Wire
    """
    # Note: only BluemiraWire objects are allowed as input for this function.
    # However, any object that can be discretized by means of the function
    # "discretizeByEdges" would be suitable. In case the function can be
    # extended to other objects changing the "discretizeByEdges" function.

    # if no plot options are given, the wire is not plotted
    if not woptions and not poptions:
        return axis

    if axis is None:
        axis = Plot3D()

    if not hasattr(wires, "__len__"):
        wires = [wires]

    for w in wires:

        if not isinstance(w, BluemiraWire):
            raise ValueError("wire must be a BluemiraWire")

        pointsw = w.discretize(ndiscr=ndiscr, byedges=True)

        for p in pointsw:
            x = [p[0] for p in pointsw]
            y = [p[1] for p in pointsw]
            z = [p[2] for p in pointsw]

        axis.plot(x, y, z, **woptions)

    plt.gca().set_aspect("auto")
    if show:
        plt.show()

    return axis


def plot_face(
    faces,
    axis=None,
    show: bool = False,
    ndiscr: int = 100,
    poptions: dict = DEFAULT["poptions"],
    woptions: dict = DEFAULT["woptions"],
    foptions: dict = DEFAULT["foptions"],
    *args,
    **kwargs,
):
    """Plot a BluemiraFace or list(BluemiraFace)

    Parameters
    ----------
    faces :
        faces to be plotted
    axis :
         (Default value = None)
    show: bool
         (Default value = False)
    ndiscr: int
         (Default value = 100)
    plane: Union[str, Base.Placement]
         (Default value = 'xy')
    poptions: dict
         (Default value = DEFAULT['poptions'])
    woptions: dict
         (Default value = DEFAULT['woptions'])
    foptions: dict
         (Default value = DEFAULT['foptions'])
    *args :

    **kwargs :

    Returns
    -------
    axis:
    """
    if not foptions and not woptions and not poptions:
        return axis

    if axis is None:
        axis = Plot3D()

    if not hasattr(faces, "__len__"):
        faces = [faces]

    if not all(isinstance(f, BluemiraFace) for f in faces):
        raise ValueError("faces must be BluemiraFace objects")

    for f in faces:

        for boundary in f.boundary:
            axis = plot_wire(
                boundary, axis, show, ndiscr, poptions=poptions, woptions=woptions
            )

    plt.gca().set_aspect("auto")
    if show:
        plt.show()

    return axis
