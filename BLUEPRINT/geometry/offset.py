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
Offset tools (including ClipperLib functions and some homebrew ones)
"""
import numpy as np
from pyclipper import (
    PyclipperOffset,
    ET_CLOSEDPOLYGON,
    ET_OPENSQUARE,
    ET_OPENROUND,
    JT_ROUND,
    JT_MITER,
    JT_SQUARE,
)
from BLUEPRINT.base.error import GeometryError
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.geometry._deprecated_tools import innocent_smoothie
from BLUEPRINT.geometry.geomtools import order, normal
from BLUEPRINT.geometry.geomtools import normal_vector, side_vector, vector_intersect
from BLUEPRINT.geometry.boolean import PyclipperMixin, loop_to_pyclippath


class OffsetOperationManager(PyclipperMixin):
    """
    Abstract base class for offset operations

    Parameters
    ----------
    loop: BLUEPRINT Loop object
        The base loop upon which to perform the offset operation
    delta: float
        The value of the offset [m]. Positive for increasing size, negative for
        decreasing
    """

    method = NotImplemented
    closed_method = ET_CLOSEDPOLYGON
    open_method = NotImplementedError

    def __init__(self, loop, delta):
        self.dims = loop.plan_dims
        self.tool = PyclipperOffset()
        path = loop_to_pyclippath(loop)
        self._scale = path[0][0] / loop.d2[0][0]  # Store scale

        if loop.closed:
            co_method = self.closed_method
        else:
            co_method = self.open_method

        self.tool.AddPath(path, self.method, co_method)
        self._result = self.perform(delta)

    def perform(self, delta):
        """
        Perform the offset operation.
        """
        delta = int(round(delta * self._scale))  # approximation
        solution = self.tool.Execute(delta)
        return self.handle_solution(solution)


class RoundOffset(OffsetOperationManager):
    """
    Offset class for rounded offsets.
    """

    name = "Round Offset"
    method = JT_ROUND
    open_method = ET_OPENROUND


class SquareOffset(OffsetOperationManager):
    """
    Offset class for squared offsets.
    """

    name = "Square Offset"
    method = JT_SQUARE
    open_method = ET_OPENSQUARE


class MiterOffset(OffsetOperationManager):
    """
    Offset class for mitered offsets.
    """

    name = "Miter Offset"
    method = JT_MITER
    open_method = ET_OPENROUND

    def __init__(self, loop, delta, miter_limit=2.0):
        super().__init__(loop, delta)

        self.tool.MiterLimit = miter_limit


def offset_clipper(loop, delta, method="square", miter_limit=2.0):
    """
    Carries out an offset operation on the Loop using the ClipperLib library

    Parameters
    ----------
    loop: BLUEPRINT Loop object
        The base loop upon which to perform the offset operation
    delta: float
        The value of the offset [m]. Positive for increasing size, negative for
        decreasing
    method: str from ['square', 'round', 'miter'] (default = 'square')
        The type of offset to perform
    miter_limit: float (default = 2.0)
        The ratio of delta to used when mitering acute corners. Only used if
        method == 'miter'

    Returns
    -------
    loop: BLUEPRINT Loop
        The offsetted loop result
    """
    if method == "square":
        tool = SquareOffset(loop, delta)
    elif method == "round":
        bluemira_warn(
            "Je ne sais pas pourquoi, mais c'est tres lent.. vaut mieux se"
            " servir d'autre chose..."
        )
        tool = RoundOffset(loop, delta)
    elif method == "miter":
        tool = MiterOffset(loop, delta, miter_limit=miter_limit)
    else:
        raise GeometryError(
            "Please choose an offset method from:\n" " round \n square \n miter"
        )
    return tool.result[0]


def offset(x, z, offset_value):
    """
    A square-based offset function (no splines). N-sized output

    Parameters
    ----------
    x: np.array(N)
        X coordinate vector
    z: np.array(N)
        Z coordinate vector
    offset_value: float
        The offset value [m]

    Returns
    -------
    xo: np.array(N)
        The X offset coordinates
    zo: np.array(N)
        The Z offset coordinates
    """
    # check numpy arrays:
    x, z = np.array(x), np.array(z)
    # check closed:
    if (x[-2:] == x[:2]).all() and (z[-2:] == z[:2]).all():
        closed = True
    elif x[0] == x[-1] and z[0] == z[-1]:
        closed = True
        # Need to lock it for closed curves
        x = np.append(x, x[1])
        z = np.append(z, z[1])
    else:
        closed = False
    p = np.array([np.array(x), np.array(z)])
    # Normal vectors for each side
    v = normal_vector(side_vector(p))
    # Construct points offset
    off_p = np.column_stack(p + offset_value * v)
    off_p2 = np.column_stack(np.roll(p, 1) + offset_value * v)
    off_p = np.array([off_p[:, 0], off_p[:, 1]])
    off_p2 = np.array([off_p2[:, 0], off_p2[:, 1]])
    ox = np.empty((off_p2[0].size + off_p2[0].size,))
    oz = np.empty((off_p2[1].size + off_p2[1].size,))
    ox[0::2], ox[1::2] = off_p2[0], off_p[0]
    oz[0::2], oz[1::2] = off_p2[1], off_p[1]
    off_s = np.array([ox[2:], oz[2:]]).T
    pnts = []
    for i in range(len(off_s[:, 0]) - 2)[0::2]:
        pnts.append(vector_intersect(off_s[i], off_s[i + 1], off_s[i + 3], off_s[i + 2]))
    pnts.append(pnts[0])
    pnts = np.array(pnts)[:-1][::-1]  # sorted ccw nicely
    if closed:
        pnts = np.concatenate((pnts, [pnts[0]]))  # Closed
    else:  # Add end points
        pnts = np.concatenate((pnts, [off_s[0]]))
        pnts = np.concatenate(([off_s[-1]], pnts))
    # sorted ccw nicely - i know looks weird but.. leave us kids alone
    # drop nan values
    return pnts[~np.isnan(pnts).any(axis=1)][::-1].T


def offset_smc(x, z, dx, min_steps=5, close_loop=False, s=0):
    """
    Simon's spliny offset algorithm.
    """
    x, z = order(x, z)  # enforce anti-clockwise
    dr_max = np.mean(dx) / min_steps  # maximum step size
    if np.mean(dx) != 0:
        dx_i, nr = max_steps(dx, dr_max)
        for i in range(nr):
            x_normal, z_normal = normal(x, z)
            x = x + dx_i * x_normal
            z = z + dx_i * z_normal
            x, z = innocent_smoothie(x, z, n=len(x), s=s / nr)
            if close_loop:
                x[0], z[0] = np.mean([x[0], x[-1]]), np.mean([z[0], z[-1]])
                x[-1], z[-1] = x[0], z[0]
    return x, z


def max_steps(dx, dr_max):
    """
    Get offset steps.
    """
    d_rbar = np.mean(dx)
    nr = int(np.ceil(d_rbar / dr_max))
    if nr < 2:
        nr = 2
    dx = dx / nr
    return dx, nr


class VariedAngularOffset:
    """
    Tool for carrying out varied offsets on an angular basis. Starting angle
    is at the outboard midplane (3 o'clock).

    Parameters
    ----------
    loop: Loop
        The Loop to make an offset from
    centre: tuple or None
        The centre about which to index the angles (default None=Loop.centroid)
    """

    def __init__(self, loop, centre=None):
        self.x, self.z = loop.d2
        if not centre:
            self.centre = loop.centroid

    def perform(
        self,
        tk_inner,
        tk_outer,
        outer_angle=np.pi / 4,
        blend_angle=np.pi / 4,
        f_smoothing=0,
        dt_max=0.1,
    ):
        """
        Perform the varied offsetting operation.

        Parameters
        ----------
        tk_inner: float
            The thickness on the inner edge of the shape (west)
        tk_outer: float
            The thickness on the outer edge of the shape (east)
        outer_angle: float
            The range of angle about 3 o'clock to have the full outer thickness
        blend_angle: float
            The range of angle over which to blend from inner to outer thickness
        f_smoothing: float
            Spline smoothing parameter
        dt_max: float
            Maximum thickness variation
        """
        thicknesses = self._blend(
            [tk_inner, tk_outer], outer_angle=outer_angle, blend_angle=blend_angle
        )
        dt, nt = max_steps(thicknesses, dt_max)
        for i in range(nt):
            self._part_fill(
                dt=dt,
                f_smoothing=f_smoothing,
            )
        return self.x, self.z

    def _part_fill(
        self,
        dt=0,
        f_smoothing=0,
    ):
        x, z = innocent_smoothie(self.x, self.z, n=len(self.x), s=f_smoothing)
        x_out, z_out = offset_smc(x, z, dt)
        self.x, self.z = x_out, z_out  # update

    def _blend(self, dt, outer_angle=np.pi / 4, blend_angle=np.pi / 4):
        """
        Blend the two thicknesses over an angle range.
        """
        theta = np.arctan2(self.z - self.centre[1], self.x - self.centre[0])
        theta[theta > np.pi] = theta[theta > np.pi] - 2 * np.pi
        tblend = dt[0] * np.ones(len(theta))  # inner
        tblend[(theta > -outer_angle) & (theta < outer_angle)] = dt[1]  # outer
        if blend_angle > 0:
            for updown in [-1, 1]:
                blend_index = (updown * theta >= outer_angle) & (
                    updown * theta < outer_angle + blend_angle
                )
                tblend[blend_index] = dt[1] + (dt[0] - dt[1]) / blend_angle * (
                    updown * theta[blend_index] - outer_angle
                )
        return tblend


def varied_angular_offset(
    loop,
    tk_inner,
    tk_outer,
    outer_angle=np.pi / 4,
    blend_angle=np.pi / 4,
    f_smoothing=0,
    dt_max=0.1,
    centre=None,
):
    """
    Perform a varied offset on a shape, by angle about the centroid. The reference
    0 angle is at 3 o'clock.

    Parameters
    ----------
    loop: Loop
        The Loop to make an offset from
    tk_inner: float
        The thickness on the inner edge of the shape (west)
    tk_outer: float
        The thickness on the outer edge of the shape (east)
    outer_angle: float
        The range of angle about 3 o'clock to have the full outer thickness
    blend_angle: float
        The range of angle over which to blend from inner to outer thickness
    f_smoothing: float
        Spline smoothing parameter
    dt_max: float
        Maximum thickness variation
    centre: tuple or None
        The centre about which to index the angles (default None=Loop.centroid)

    Returns
    -------
    offset_loop: Loop
        The offset Loop
    """
    from BLUEPRINT.geometry.loop import Loop

    dims = loop.plan_dims
    tool = VariedAngularOffset(loop, centre=centre)
    x, z = tool.perform(
        tk_inner, tk_outer, outer_angle, blend_angle, f_smoothing, dt_max
    )
    loop_dict = {dims[0]: x, dims[1]: z}
    return Loop.from_dict(loop_dict)


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
