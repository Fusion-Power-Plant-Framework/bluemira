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
Shape parameterisations and optimisation variable classes
"""
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp
import numpy as np
from scipy.special import binom
from scipy.special import iv as bessel
from collections import OrderedDict
from pandas import DataFrame
from BLUEPRINT.base.error import GeometryError
from bluemira.geometry._deprecated_tools import innocent_smoothie
from BLUEPRINT.geometry.geomtools import clock, qrotate, xz_interp
from BLUEPRINT.geometry.geomtools import circle_seg


class OptVariables(OrderedDict):
    """
    Optimisation variables ordered dictionary class
    """

    @staticmethod
    def normalise_variable(val, lb, ub):
        """
        Normalises a value relative to a lower and upper bound

        Parameters
        ----------
        val: float
            The value to normalise
        lb: float
            The lower bound against which to normalise
        ub: float
            The upper bound against which to normalise

        Returns
        -------
        n_val: float
            The normalised value between the lower and upper bound
        """
        return (val - lb) / (ub - lb)

    def normalise_key(self, key):
        """
        Normalises a value against a key in the OptVariables object

        Parameters
        ----------
        key: str
            The key of the value to normalise

        Returns
        -------
        n_val: float
            The normalised value against the key's bounds
        """
        return self.normalise_variable(
            self[key]["value"], self[key]["lb"], self[key]["ub"]
        )

    def set_limit(self, key):
        """
        Enforces bound limits on a key in the OptVariables object

        Parameters
        ----------
        key: str
            The key of the value to enforce normalisation
        """
        self[key]["value"] = np.clip(
            self[key]["value"], self[key]["lb"], self[key]["ub"]
        )

    def check_var(self, var):
        """
        Check a variable in the OptimisationVariables. Handles Polyspline tension
        recognition.
        """
        if var == "l":
            var = "l0s"
        if var not in self.keys():  # match sub-string
            var = next((v for v in self.keys() if var in v))
        return var

    def set_oppvar(self, oppvar):
        """
        Set the optimisation variables, normalise them, and set up bounds.
        """
        nopp = len(oppvar)
        xnorm, bnorm = np.zeros(nopp), np.zeros((nopp, 2))
        for i, var in enumerate(oppvar):
            var = self.check_var(var)
            xnorm[i] = self.normalise_key(var)
            bnorm[i, :] = [0, 1]
        return xnorm, bnorm

    def get_oppvar(self, oppvar, xnorm):
        """
        Get the optimisation variables.
        """
        x = np.copy(xnorm)
        for i, var in enumerate(oppvar):
            var = self.check_var(var)
            x[i] = x[i] * (self[var]["ub"] - self[var]["lb"]) + self[var]["lb"]
        return x

    def get_value(self):
        """
        Returns the current values of the optimisation variables

        Returns
        -------
        values: np.array(len(self))
            The ordered array of the (un-normalised) optimisation variable
            values
        """
        x = np.zeros(len(self))
        for i, name in enumerate(self.keys()):
            x[i] = self[name]["value"]
        return x

    def get_xnorm(self):
        """
        Returns the current normalised values of the optimisation variables

        Returns
        -------
        xnorm: np.array(len(self))
            The ordered array of the normalised optimisation variable
            values
        """
        xn = np.zeros(len(self))
        for i, k in enumerate(self.keys()):
            xn[i] = self.normalise_key(k)
        return xn

    def plot_oppvar(self, oppvar, ax=None, eps=1e-2, fmt="1.2f", scale=1, postfix=""):
        """
        Plot the optimisation variables and their position w.r.t. their constraints.
        """
        if ax is None:
            _, ax = plt.subplots()
        _, _ = self.set_oppvar(oppvar)
        for var in self.keys():
            self[var]["xnorm"] = self.normalise_key(var)

        data = DataFrame(self).T
        data.reset_index(level=None, inplace=True)
        sns.set_color_codes("muted")
        sns.barplot(x="xnorm", y="index", data=data, color="b")
        sns.despine(bottom=True)
        plt.ylabel("")
        ax.get_xaxis().set_visible(False)
        patch = ax.patches
        keys = list(self.keys())
        values = [self[var]["value"] for var in keys]
        xnorms = [self[var]["xnorm"] for var in keys]
        for p, value, xnorm, var in zip(patch, values, xnorms, keys):
            x = p.get_width()
            if xnorm < 0:
                x = 0
            y = p.get_y() + p.get_height() / 2
            size = "small"
            if xnorm < eps or xnorm > 1 - eps:
                size = "large"
            text = " {:{fmt}}".format(scale * value, fmt=fmt)
            text += postfix + " "
            if var not in oppvar:
                text += "*"
            if xnorm < 0.1:
                ha = "left"
                color = 0.25 * np.ones(3)
            else:
                ha = "right"
                color = 0.75 * np.ones(3)
            ax.text(x, y, text, ha=ha, va="center", size=size, color=color)
        ax.plot(
            0.5 * np.ones(2),
            np.sort(ax.get_ylim()),
            "--",
            color=0.5 * np.ones(3),
            zorder=0,
            lw=1,
        )
        ax.plot(
            np.ones(2),
            np.sort(ax.get_ylim()),
            "-",
            color=0.2 * np.ones(3),
            zorder=0,
            lw=1.5,
        )
        xlim = ax.get_xlim()
        xmin, xmax = np.min([0, xlim[0]]), np.max([1, xlim[1]])
        ax.set_xlim([xmin, xmax])


class Parameterisation:
    """
    Base shape parameterisation object

    Parameters
    ----------
    npoints: int
        The number of points to use when drawing the resulting shape

    Attributes
    ----------
    xo: OptVariables
    oppvar: list

    Notes
    -----
    kwargs are overloaded to enable different parameterisations to have the
    same signatures
    """

    name: str = NotImplemented

    def __init__(self, npoints=200, **kwargs):
        self.xo = OptVariables()
        self.npoints = npoints
        self.oppvar = NotImplemented

    def adjust_xo(self, name, **kwargs):
        """
        Adjust the optimisation variables of the parameterisation.

        Parameters
        ----------
        name: str
            The variable name

        Other Parameters
        ----------------
        value: float
            The value to set
        lb: float
            The lower bound to set
        ub: float
            The upper bound to set

        Notes
        -----
        If the new value violates the bounds in the optimisation variables, the
        bounds will be re-adjusted accordingly.
        """
        if name not in self.xo.keys():
            if name == "l":
                pass
            else:
                raise KeyError(
                    f'Parameter "{name}" not in {self.name} ' "parameterisation."
                )
        for var in kwargs:
            if name == "l":
                for lkey in self.lkeyo:
                    self.xo[lkey][var] = kwargs[var]
            else:
                self.xo[name][var] = kwargs[var]
                if "value" in kwargs.keys():
                    if kwargs["value"] <= kwargs.get("lb", self.xo[name]["lb"]):
                        self.xo[name]["lb"] = 0.5 * kwargs["value"]
                    if kwargs["value"] > kwargs.get("ub", self.xo[name]["ub"]):
                        self.xo[name]["ub"] = 1.5 * kwargs["value"]

    def set_oppvar(self):
        """
        Set the optimisation variables, normalise them, and set up bounds.
        """
        return self.xo.set_oppvar(self.oppvar)

    def get_oppvar(self, xnorm):
        """
        Get the active optimisation variables.
        """
        return self.xo.get_oppvar(self.oppvar, xnorm)

    def remove_oppvar(self, key):
        """
        Removes a variable from the optimisation variables

        Parameters
        ----------
        key: str
            The key of the optimisation variable to remove
        """
        try:
            self.oppvar.remove(key)
        except ValueError:
            pass

    def get_xo(self):
        """
        Get the all the values of the Parameterisation OptimisationVariables, regardless
        of whether they have been removed from the optimisation problem.

        Returns
        -------
        values: list
            The values of all the Parameterisation variables
        """
        values = []
        for n in self.oppvar:
            values.append(self.xo[n]["value"])
        return values

    def set_input(self, **kwargs):
        """
        Set the active optimisation variables in the Parameterisation.
        """
        inputs = self.get_input(oppvar=self.oppvar, **kwargs)
        for key in inputs:
            if key in self.xo:
                try:  # dict
                    for k in inputs[key]:
                        self.xo[key][k] = inputs[key][k]
                except TypeError:  # single value - object is not iterable
                    self.xo[key]["value"] = inputs[key]
                self.xo.set_limit(key)

    @staticmethod
    def get_input(oppvar=[], **kwargs):
        """
        Get the active optimisation variable inputs
        """
        if "x" in kwargs:
            inputs = {}
            x_input = kwargs.get("x")
            try:
                for var, x in zip(oppvar, x_input):
                    inputs[var] = x
            except ValueError:
                raise ValueError("\nRequire self.variables")
        elif "inputs" in kwargs:
            inputs = kwargs["inputs"]
        else:
            inputs = {}
        return inputs

    def close_loop(self, p):
        """
        Close a Parameterisation shape. Resamples the loop for even point spacing.

        Parameters
        ----------
        p: dict
            The coordinate dictionary

        Returns
        -------
        p: dict
            The output closed coordinate dictionary
        """
        for var in ["x", "z"]:
            p[var] = np.append(p[var], p[var][0])
        p["x"], p["z"] = innocent_smoothie(p["x"], p["z"], n=self.npoints)
        return p

    def draw(self, *args, **kwargs):
        """
        Override with the method to generate the coordinate dictionary.
        """
        raise NotImplementedError

    def plot(self, ax=None, inputs={}):
        """
        Plot the Parameterisation.
        """
        if ax is None:
            _, ax = plt.subplots()

        x = self.draw(inputs=inputs)
        ax.plot(x["x"], x["z"], "-", ms=8, color=0.4 * np.ones(3))
        ax.set_aspect("equal")

    def plot_oppvar(self, ax=None, eps=1e-2, fmt="1.2f", scale=1, postfix=""):
        """
        Plots optimisation variables of the final shape and the position
        relative to the bounds
        """
        if ax is None:
            _, ax = plt.subplots()
        self.xo.plot_oppvar(
            self.oppvar, ax=ax, eps=eps, fmt=fmt, scale=scale, postfix=postfix
        )


def picture_frame(x1, x2, z1, z2, ri=0.8, ro=0.8, npoints=200):
    """
    Defines PictureFrame shape parameterisation

    Parameters
    ----------
    x1: float
        x-coordinate (from machine axis) of innermost leg [m]
    x2: float
        x-coordinate (from machine axis) of outer leg [m]
    z1: float
        z-height (from mid-plane) of coil [m]
    ri: float
        radius of inner rounded corners [m]
    ro: float
        radius of outer rounded corners [m]
    npoints: int (default = 200)
        The size of the x, z coordinate sets to return

    Returns
    -------
    x: np.array(npoints)
        The x coordinates of the tapered PictureFrame shape
    z: np.array(npoints)
        The z coordinates of the tapered PictureFrame shape

    Note
    ----
    Returns an open set of coordinates
    """
    if x2 < x1:
        raise GeometryError(
            "picture_frame parameterisation requires an x2 "
            f"value greater than or equal to x1: {x2} < {x1}"
        )

    if z1 < 0:
        raise GeometryError(
            "picture_frame parameterisation requires an z1 value"
            f"that is not negative: {0} >= {z1}"
        )

    if z2 > 0:
        raise GeometryError(
            "picture_frame parameterisation requires an z1 value"
            f"that is not positive: {0} <= {z2}"
        )

    if ro >= x2 - x1:
        raise GeometryError(
            "picture_frame parameterisation requires an ro value "
            f"less than x2-x1: {ro} >= {x2-x1}"
        )
    if ri >= x2 - x1:
        raise GeometryError(
            "picture_frame parameterisation requires an ri value "
            f"less than x2-x1: {ri} >= {x2-x1}"
        )
    if ro >= z1:
        raise GeometryError(
            "picture_frame parameterisation requires an ro value"
            f"less than z1: {ro} >= {z1}"
        )
    if ri >= z1:
        raise GeometryError(
            "picture_frame parameterisation requires an ri value"
            f"less than z1: {ri} >= {z1}"
        )
    if x1 <= 0:
        raise GeometryError(
            "picture_frame parameterisation requires an x1 value"
            f"greater than 0: {0} >= {x1}"
        )

    if ro < 0:
        raise GeometryError(
            "picture_frame parameterisation requires an ro value"
            f"that is positive or 0: {0} > {ro}"
        )
    if ri < 0:
        print("Given negative ri value, resetting ri to = 0")
        ri = 0

    if npoints < 10:
        raise ValueError(
            "At least 10 point must be used to define the shape" f"= {npoints}"
        )

    x, z = [], []
    # Inner leg, positive z
    npts = 2
    ri = np.round(ri, decimals=1)
    x = np.append(x, x1 * np.ones(npts))
    z = np.append(z, np.linspace(0, z1 - ri, npts))

    # Top inner corner
    if ri > 0:
        npts = int((npoints) * 0.25)
        x_c, z_c = circle_seg(ri, (x1 + ri, z1 - ri), angle=-90, npoints=npts, start=180)
        x = np.append(x, x_c)
        z = np.append(z, z_c)
    else:
        x = np.append(x, x1)
        z = np.append(z, z1)

    # Top leg
    npts = 2
    x = np.append(x, np.linspace(x1 + ri, x2 - ro, npts))
    z = np.append(z, z1 * np.ones(npts))

    # Top outer corner
    npts = int((npoints) * 0.25)
    x_c, z_c = circle_seg(ro, (x2 - ro, z1 - ro), angle=-90, npoints=npts, start=90)
    x = np.append(x, x_c)
    z = np.append(z, z_c)

    # Outer Leg
    npts = 2
    x = np.append(x, x2 * np.ones(npts))
    z = np.append(z, np.linspace(z1 - ro, z2 + ro, npts))

    # Bottom Outer Corner
    npts = int((npoints) * 0.25)
    x_c, z_c = circle_seg(ro, (x2 - ro, z2 + ro), angle=-90, npoints=npts, start=0)
    x = np.append(x, x_c)
    z = np.append(z, z_c)

    # Bottom leg
    npts = 2
    x = np.append(x, np.linspace(x2 - ro, x1 + ri, npts))
    z = np.append(z, z2 * np.ones(npts))

    # Bottom inner corner
    if ri > 0:
        npts = int((npoints) * 0.25)
        x_c, z_c = circle_seg(ri, (x1 + ri, z2 + ri), angle=-90, npoints=npts, start=-90)
        x = np.append(x, x_c)
        z = np.append(z, z_c)

    # Inner leg, negative z
    npts = 2
    x = np.append(x, x1 * np.ones(npts))
    z = np.append(z, np.linspace(z2 + ri, 0, npts))

    return x, z


class PictureFrame(Parameterisation):
    """
    Picture-frame parameterisation
    """

    name = "PictureFrame"

    def __init__(self, npoints=200, **kwargs):
        super().__init__(npoints=npoints)

        self.xo["x1"] = {"value": 4.486, "lb": 4, "ub": 5}  # inner leg
        self.xo["x2"] = {"value": 16, "lb": 14, "ub": 20}  # outer leg
        self.xo["z1"] = {"value": 8, "lb": 5, "ub": 15}  # Vertical height
        self.xo["z2"] = {"value": -6, "lb": -15, "ub": -3}  # vertical
        self.xo["ri"] = {"value": 0.0, "lb": 0.0, "ub": 0.2}  # Inboard corner radius
        self.xo["ro"] = {"value": 2, "lb": 1.999, "ub": 2.001}  # outboard corner radius
        self.oppvar = list(self.xo.keys())

        self.segments = None

    def draw(self, **kwargs):
        """
        Calculate the PictureFrame parameterisation shape.

        Returns
        -------
        p: dict
            The coordinate dictionary of points
        """
        self.set_input(**kwargs)

        x1, x2, z1, z2, ri, ro = self.xo.get_value()

        x, z = picture_frame(x1, x2, z1, z2, ri, ro)
        self.segments = {"x": x, "z": z}
        p = self.segments
        p = self.close_loop(p)
        p["x"], p["z"] = clock(p["x"], p["z"])
        return p


def tapered_picture_frame(x1, x2, x3, z1_frac, z2, z3, r, npoints=200):
    """
    Defines Tapered PictureFrame shape parameterisation

    Parameters
    ----------
    x1: float
        x-coordinate (from machine axis) of innermost leg [m]
    x2: float
        x-coordinate (from machine axis) of middle leg [m]
    x3: float
        x-coordinate (from machine axis) of outermost leg [m]
    z1_frac: float
        z-height on straight part of tapered section as fraction of z2
    z2: float
        z-height (from mid-plane) of tapered section top [m]
    z3: float
        total z-height (from mid-plane) of coil shape [m]
    r: float
        radius of outer rounded corners [m]
    npoints: int (default = 200)
        The size of the x, z coordinate sets to return

    Returns
    -------
    x: np.array(npoints)
        The x coordinates of the tapered PictureFrame shape
    z: np.array(npoints)
        The z coordinates of the tapered PictureFrame shape

    Note
    ----
    Returns an open set of coordinates


    """
    if x2 < x1:
        raise GeometryError(
            "tapered_picture_frame parameterisation requires an x2 "
            f"value greater than or equal to x1: {x2} < {x1}"
        )
    if x3 <= x2:
        raise GeometryError(
            "tapered_picture_frame parameterisation requires an x3 value"
            f"greater than x2: {x2} >= {x3}"
        )
    if z1_frac > 1.00 or z1_frac < 0:
        raise GeometryError(
            "tapered_picture_frame parameterisation requires 0 <= z1_frac < 1:"
            f"z1_frac = {z1_frac}"
        )
    if z2 < 0:
        raise GeometryError(
            "tapered_picture_frame parameterisation requires an z2 value"
            f"that is not negative: {0} >= {z2}"
        )
    if z3 <= z2:
        raise GeometryError(
            "tapered_picture_frame parameterisation requires an z3 value"
            f"greater than z2: {z2} >= {z3}"
        )

    z1 = z1_frac * (z2)

    if r >= x3 - x2:
        raise GeometryError(
            "tapered_picture_frame parameterisation requires an r value "
            f"less than x3-x2: {r} >= {x3-x2}"
        )
    if r >= z3:
        raise GeometryError(
            "tapered_picture_frame parameterisation requires an r value"
            f"less than z3: {r} >= {z3}"
        )
    if x1 <= 0:
        raise GeometryError(
            "tapered_picture_frame parameterisation requires an x1 value"
            f"greater than 0: {0} >= {x1}"
        )
    if z1 > z2:
        raise GeometryError(
            "tapered_picture_frame parameterisation requires an z1 value"
            f"that is less than or equal to z2: {z1} > {z2}"
        )
    if r < 0:
        raise GeometryError(
            "tapered_picture_frame parameterisation requires an r value"
            f"that is positive or 0: {0} > {r}"
        )
    if npoints < 10:
        raise ValueError(
            "At least 10 point must be used to define the shape" f"= {npoints}"
        )

    # Define segments of shape, and assign number of points
    # inner most vertical bit (-z1 to z1)
    n_pts_inner_vertical = (
        1  # minimum number 1 , can be changed to fraction of npoints later
    )

    # diagonals (z1 to z2_frac, -z1 to -z2_frac)
    n_pts_diagonals = 1  # minimum 1, can play w npoints later

    # z2_frac to z3 verticals
    n_pts_mid_vertical = 1

    n_pts_long_horizontal = 1

    n_pts_outer_vertical = 1

    n_pts_corner = int(
        2
        + np.ceil(
            0.5 * npoints
            - (
                n_pts_inner_vertical
                + n_pts_diagonals
                + n_pts_mid_vertical
                + n_pts_long_horizontal
                + n_pts_outer_vertical
            )
        )
    )

    # Make sections

    z_inner_pos = np.linspace(0, (z1), n_pts_inner_vertical, endpoint=False)
    x_inner_pos = x1 * np.ones(n_pts_inner_vertical)

    z_diag_pos = np.linspace((z1), z2, n_pts_diagonals, endpoint=False)
    x_diag_pos = np.linspace(x1, x2, n_pts_diagonals, endpoint=False)

    z_mid_pos = np.linspace(z2, z3, n_pts_mid_vertical, endpoint=False)
    x_mid_pos = x2 * np.ones(n_pts_mid_vertical)

    x_hor_pos = np.linspace(x2, x3 - r, n_pts_long_horizontal, endpoint=False)
    z_hor_pos = z3 * np.ones(n_pts_long_horizontal)

    (
        x_corner_pos,
        z_corner_pos,
    ) = circle_seg(r, (x3 - r, z3 - r), angle=-90, npoints=n_pts_corner, start=90)

    z_outer = np.linspace(z3 - r, -z3 + r, n_pts_outer_vertical * 2 - 1, endpoint=False)
    x_outer = x3 * np.ones(
        n_pts_outer_vertical * 2 - 1,
    )

    x_corner_neg = np.flip(x_corner_pos)
    z_corner_neg = -np.flip(z_corner_pos)

    # now remove extra point from top outboard corner (repeated curve end point):
    x_corner_pos = x_corner_pos[:-1]
    z_corner_pos = z_corner_pos[:-1]

    x_hor_neg = np.flip(x_hor_pos)  # np.linspace(x3 - r, x2, n_pts_long_horizontal)
    z_hor_neg = -z_hor_pos

    z_mid_neg = -np.flip(z_mid_pos)
    x_mid_neg = x_mid_pos

    z_diag_neg = -np.flip(z_diag_pos)
    x_diag_neg = np.flip(x_diag_pos)

    z_inner_neg = -np.flip(z_inner_pos)
    x_inner_neg = x_inner_pos

    xx = np.concatenate(
        [
            x_inner_pos,
            x_diag_pos,
            x_mid_pos,
            x_hor_pos,
            x_corner_pos,
            x_outer,
            x_corner_neg,
            x_hor_neg,
            x_mid_neg,
            x_diag_neg,
            x_inner_neg,
        ]
    )
    zz = np.concatenate(
        [
            z_inner_pos,
            z_diag_pos,
            z_mid_pos,
            z_hor_pos,
            z_corner_pos,
            z_outer,
            z_corner_neg,
            z_hor_neg,
            z_mid_neg,
            z_diag_neg,
            z_inner_neg,
        ]
    )

    return xx, zz


class TaperedPictureFrame(Parameterisation):
    """
    Tapered Picture-frame parameterisation
    """

    name = "TaperedPictureFrame"

    def __init__(self, npoints=200, **kwargs):
        super().__init__(npoints=npoints)
        # Do special stuff, like defining optimization variables

        self.xo["x1"] = {"value": 0.4, "lb": 0.3, "ub": 0.5}  # inner leg
        self.xo["x2"] = {"value": 1.1, "lb": 1, "ub": 1.3}  # middle leg
        self.xo["x3"] = {"value": 6.5, "lb": 6, "ub": 9}  # outer leg
        self.xo["z1_frac"] = {"value": 0.5, "lb": 0.45, "ub": 0.8}  # Vertical height
        self.xo["z2"] = {"value": 6.5, "lb": 6, "ub": 8}  # vertical
        self.xo["z3"] = {"value": 7, "lb": 6.825, "ub": 9}  # vertical

        self.xo["r"] = {"value": 0.5, "lb": 0.00, "ub": 1.0}  # Corner radius
        self.oppvar = list(self.xo.keys())

        self.segments = None

    def draw(self, **kwargs):
        # Draw the x, z points of the shape parameterisation
        """
        Calculate the tapered PictureFrame parameterisation shape.

        Returns
        -------
        p: dict
            The coordinate dictionary of points
        """
        self.set_input(**kwargs)

        x1, x2, x3, z1, z2_frac, z3, r = self.xo.get_value()
        x, z = tapered_picture_frame(x1, x2, x3, z1, z2_frac, z3, r, npoints=200)
        self.segments = {"x": x, "z": z}
        p = self.segments
        p = self.close_loop(p)
        p["x"], p["z"] = clock(p["x"], p["z"])
        return p


def curved_picture_frame(
    x_in,
    x_mid,
    x_curve_start,
    x_out,
    z_in,
    z_mid_up,
    z_mid_down,
    z_top,
    z_bottom,
    r_c,
    npoints=200,
):
    """
    Curved PictureFrame shape parameterisation

    Parameters
    ----------
    x_in: float
        x-coordinate (from machine axis) of innermost leg [m]
    x_mid: float
        x-coordinate (from machine axis) of the taper top [m]
    x_j: float
        x-coordinate (from machine axis) of start of top curve [m]
    x_out: float
        x-coordinate (from machine axis) of outermost leg [m]
    z_in: float
        z-height (from mid-plane) of coil tapered section [m]
    z_mid_up: float
        z-height of straight part of coil top leg [m]
    z_mid_down: float
        z_height of straight part of coil bottom leg
    z_top: float
        max height of top legs [m]
    z_bottom: float
        max height of bottom legs [m]
    r_c: float
        radius of corner/transitioning curves [m]
    npoints: int (default = 200)
        The size of the x, z coordinate sets to return

    Returns
    -------
    x: np.array(npoints)
        The x coordinates of the tapered PictureFrame shape
    z: np.array(npoints)
        The z coordinates of the tapered PictureFrame shape

    Note
    ----
    Returns an open set of coordinates
    """
    if x_in < 0:
        raise GeometryError(
            "Curved picture_frame parameterisation requires an x_in value"
            f"that is not negative: {0} >= {x_in}"
        )
    # SC coil
    if x_mid == 0:
        x_mid = x_in

    if x_mid < x_in:
        raise GeometryError(
            "Curved picture_frame parameterisation requires an x_mid "
            f"value greater than or equal to x_in: {x_mid} < {x_in}"
        )
    if x_curve_start < x_mid:
        raise GeometryError(
            "Curved picture_frame parameterisation requires an x_curve_start "
            f"value greater than or equal to x_mid: {x_curve_start} < {x_mid}"
        )
    if x_out < x_curve_start:
        raise GeometryError(
            "Curved picture_frame parameterisation requires an x_out "
            f"value greater than or equal to x_curve_start: {x_curve_start} < {x_mid}"
        )
    if z_in < 0:
        raise GeometryError(
            "Curved picture_frame parameterisation requires an z_in value"
            f"that is not negative: {0} >= {z_in}"
        )

    if npoints < 10:
        raise ValueError("N. of Points must be > 10, npoints inputted " f"= {npoints}")

    x, z = [], []

    # If no taper, define a straight line
    npts = int(npoints * 0.1)
    if abs(x_mid - x_in) <= 1e-3 or z_in == 0:
        # If there is no tapering
        x = np.append(x, [x_in, x_in])
        z = np.append(z, [z_mid_down, z_mid_up])

    # Inboard Curved Taper, positive z
    else:
        # Curved taper radius
        x_t = x_mid - x_in
        alpha = np.arctan(z_in / (x_t))
        theta_t = np.pi - 2 * alpha
        r_taper = z_in / np.sin(theta_t)

        # Curved taper angle
        angle = np.rad2deg(np.arcsin(z_in / r_taper))

        x_c_t, z_c_t = circle_seg(
            r_taper,
            h=(x_in + r_taper, 0),
            angle=-2 * angle,
            start=180 + angle,
            npoints=npts,
        )

        x = np.append(x, x_mid)
        z = np.append(z, z_mid_down)
        x = np.append(x, x_c_t)
        z = np.append(z, z_c_t)
        x = np.append(x, x_mid)
        z = np.append(z, z_mid_up)

    if z_top > (z_mid_up + 0.01):
        # If top leg is domed
        x, z = CurvedPictureFrame.domed_leg(
            x, x_out, x_curve_start, x_mid, z, z_top, z_mid_up, npoints, flip=False
        )
    else:
        # If top leg is flat
        r_c = min(x_curve_start - x_mid, 0.8)
        x = np.append(x, x_out - r_c)
        z = np.append(z, z_mid_up)
        npts = int(npoints * 0.1)
        x_c, z_c = circle_seg(
            r_c, h=(x_out - r_c, z_mid_up - r_c), angle=-90, npoints=npts, start=90
        )
        x = np.append(x, x_c)
        z = np.append(z, z_c)
    # Outer leg
    npts = 2
    x = np.append(x, x_out)
    z = np.append(z, z_mid_down)
    if z_bottom < (z_mid_down - 0.01):
        # Domed bottom leg
        x, z = CurvedPictureFrame.domed_leg(
            x, x_out, x_curve_start, x_mid, z, z_bottom, z_mid_down, npoints, flip=True
        )
    else:
        # flat bottom leg
        x = np.append(x, x_out - r_c)
        z = np.append(z, z_mid_down)
        npts = int(npoints * 0.1)
        x_c, z_c = circle_seg(
            r_c, h=(x_out - r_c, z_mid_down + r_c), angle=-90, npoints=npts, start=0
        )
        x = np.append(x, x_c)
        z = np.append(z, z_c)

    x = np.append(x, x_mid)
    z = np.append(z, z_mid_down)

    return x, z


class CurvedPictureFrame(Parameterisation):
    """
    Curved Picture-frame parameterisation
    """

    name = "CurvedPictureFrame"

    def __init__(self, npoints=200, **kwargs):
        super().__init__(npoints=npoints)
        # Do special stuff, like defining optimization variables

        self.xo["x_in"] = {"value": 0.4, "lb": 0.3, "ub": 0.5}  # inner leg
        self.xo["x_mid"] = {"value": 1.55, "lb": 1.5, "ub": 1.6}  # middle leg
        self.xo["x_curve_start"] = {"value": 2.5, "lb": 2.4, "ub": 2.6}  # middle leg
        self.xo["x_out"] = {"value": 9.5, "lb": 9.4, "ub": 9.8}  # outer leg
        self.xo["z_in"] = {"value": 0.5, "lb": 0.45, "ub": 0.8}  # Vertical height
        self.xo["z_mid_up"] = {"value": 7.5, "lb": 6, "ub": 8}  # vertical
        self.xo["z_mid_down"] = {"value": -7.5, "lb": -8, "ub": -6}  # vertical
        self.xo["z_top"] = {"value": 14.5, "lb": 14.0, "ub": 15}  # vertical
        self.xo["z_bottom"] = {"value": -14.5, "lb": -15.0, "ub": -14}  # vertical
        self.xo["r_c"] = {"value": 0.3, "lb": 0.00, "ub": 0.8}  # Corner radius

        self.oppvar = list(self.xo.keys())

        self.segments = None

    @staticmethod
    def domed_leg(
        x, x_out, x_curve_start, x_mid, z, z_top, z_mid, npoints, flip=False, *, r_c=0
    ):
        """
        Makes smooth dome for CP coils
        """
        # If top leg is domed
        # Define basic Top Curve (with no joint or corner transitions)
        r_j = min(x_curve_start - x_mid, 0.8)
        alpha = np.arctan(0.5 * (x_out - x_curve_start) / abs(z_top - z_mid))
        theta_leg_basic = 2 * (np.pi - 2 * alpha)
        r_leg = 0.5 * (x_out - x_curve_start) / np.sin(theta_leg_basic / 2)
        z_top_r_leg = z_top + r_leg if flip else z_top - r_leg
        leg_centre = (x_out - 0.5 * (x_out - x_curve_start), z_top_r_leg)
        # Transitioning Curve
        sin_a = np.sin(theta_leg_basic / 2)
        cos_a = np.cos(theta_leg_basic / 2)
        alpha_leg = (
            np.arcsin(np.abs(r_leg * sin_a - r_c) / (r_leg - r_c)) + theta_leg_basic / 2
        )
        # Joint Curve
        theta_j = np.arccos((r_leg * cos_a + r_j) / (r_leg + r_j))
        z_mid_r_j = z_mid - r_j if flip else z_mid + r_j
        joint_curve_centre = (
            leg_centre[0] - (r_leg + r_j) * np.sin(theta_j),
            z_mid_r_j,
        )
        theta_leg_final = alpha_leg - (theta_leg_basic / 2 - theta_j)
        x_c_j, z_c_j = circle_seg(
            r_j,
            h=joint_curve_centre,
            angle=-np.rad2deg(theta_j) if flip else np.rad2deg(theta_j),
            start=90 if flip else -90,
            npoints=int(npoints * 0.1),
        )
        angle2 = np.rad2deg(theta_leg_final)
        start2 = 90 + np.rad2deg(theta_j)
        x_c_l, z_c_l = circle_seg(
            r_leg,
            h=leg_centre,
            angle=angle2 if flip else -angle2,
            start=-start2 if flip else start2,
            npoints=int(npoints * 0.2),
        )
        if flip:
            x = np.append(x, np.flip(x_c_l))
            z = np.append(z, np.flip(z_c_l))
            x = np.append(x, np.flip(x_c_j))
            z = np.append(z, np.flip(z_c_j))
        else:
            x = np.append(x, x_c_j)
            z = np.append(z, z_c_j)
            x = np.append(x, x_c_l)
            z = np.append(z, z_c_l)

        return x, z

    def draw(self, **kwargs):
        # Draw the x, z points of the shape parameterisation
        """
        Calculate the curved PictureFrame parameterisation shape.

        Returns
        -------
        p: dict
            The coordinate dictionary of points
        """
        self.set_input(**kwargs)

        (
            x_in,
            x_mid,
            x_curve_start,
            x_out,
            z_in,
            z_mid_up,
            z_mid_down,
            z_top,
            z_bottom,
            r_c,
        ) = self.xo.get_value()
        x, z = curved_picture_frame(
            x_in,
            x_mid,
            x_curve_start,
            x_out,
            z_in,
            z_mid_up,
            z_mid_down,
            z_top,
            z_bottom,
            r_c,
            npoints=400,
        )
        self.segments = {"x": x, "z": z}
        p = self.segments
        p = self.close_loop(p)
        p["x"], p["z"] = clock(p["x"], p["z"])
        return p


class TripleArc(Parameterisation):
    """
    Triple arc shape parameterisation
    """

    # fmin_slsqp does really badly on this one... lots of minima?
    name = "TripleArc"

    def __init__(self, npoints=200, **kwargs):
        super().__init__(npoints=npoints)

        self.xo["xo"] = {"value": 4.486, "lb": 4.4, "ub": 4.5}  # x origin
        self.xo["zo"] = {"value": 0, "lb": -1, "ub": 1}  # z origin
        self.xo["sl"] = {"value": 6.428, "lb": 6, "ub": 12}  # straight length
        self.xo["f1"] = {"value": 3, "lb": 3, "ub": 12}  # rs == f1*z small
        self.xo["f2"] = {"value": 4, "lb": 3, "ub": 12}  # rm == f2*rs mid
        self.xo["a1"] = {"value": 20, "lb": 5, "ub": 120}  # small arc angle, deg
        self.xo["a2"] = {"value": 40, "lb": 10, "ub": 120}  # middle arc angle, deg
        # TODO: EDGE CASE: a1+a1 > 90
        self.oppvar = list(self.xo.keys())

        self.segments = None

    def draw(self, **kwargs):
        """
        Calculate the TripleArc parameterisation shape.

        Returns
        -------
        p: dict
            The coordinate dictionary of points
        """
        self.npoints = kwargs.get("npoints", self.npoints)
        self.set_input(**kwargs)
        self.segments = {"x": [], "z": []}
        xo, zo, sl, f1, f2, a1, a2 = self.xo.get_value()
        a1 *= np.pi / 180  # convert to radians
        a2 *= np.pi / 180
        asum = a1 + a2
        # straight section
        x = xo * np.ones(2)
        z = np.array([zo, zo + sl])
        self.segments["x"].append(x)
        self.segments["z"].append(z)

        # small arc
        theta = np.linspace(0, a1, int(round(0.5 * self.npoints * a1 / np.pi)))
        rx, zx = x[-1], z[-1]
        x = np.append(x, x[-1] + f1 * (1 - np.cos(theta)))
        z = np.append(z, z[-1] + f1 * np.sin(theta))
        self.segments["x"].append(rx + f1 * (1 - np.cos(theta)))
        self.segments["z"].append(zx + f1 * np.sin(theta))

        # mid arc
        theta = np.linspace(theta[-1], asum, int(round(0.5 * self.npoints * a2 / np.pi)))
        rx, zx = x[-1], z[-1]
        x = np.append(x, x[-1] + f2 * (np.cos(a1) - np.cos(theta)))
        z = np.append(z, z[-1] + f2 * (np.sin(theta) - np.sin(a1)))
        self.segments["x"].append(rx + f2 * (np.cos(a1) - np.cos(theta)))
        self.segments["z"].append(zx + f2 * (np.sin(theta) - np.sin(a1)))

        # large arc
        rl = (z[-1] - zo) / np.sin(np.pi - asum)
        theta = np.linspace(theta[-1], np.pi, 60)
        rx, zx = x[-1], z[-1]
        x = np.append(x, x[-1] + rl * (np.cos(np.pi - theta) - np.cos(np.pi - asum)))
        z = np.append(z, z[-1] - rl * (np.sin(asum) - np.sin(np.pi - theta)))
        self.segments["x"].append(
            rx + rl * (np.cos(np.pi - theta) - np.cos(np.pi - asum))
        )
        self.segments["z"].append(zx - rl * (np.sin(asum) - np.sin(np.pi - theta)))
        x = np.append(x, x[::-1])[::-1]
        z = np.append(z, -z[::-1] + 2 * zo)[::-1]
        x, z = innocent_smoothie(x, z, n=self.npoints)  # distribute points
        x = {"x": x[::-1], "z": z[::-1]}
        x = self.close_loop(x)
        return x


class PrincetonD(Parameterisation):
    """
    Princeton-D "bending-free" shape parameterisation
    """

    name = "PrincetonD"

    def __init__(self, npoints=200, **kwargs):
        super().__init__(npoints=npoints)

        self.xo["x1"] = {"value": 4.486, "lb": 3, "ub": 5}  # inner radius
        self.xo["x2"] = {"value": 15.708, "lb": 10, "ub": 20}  # outer radius
        self.xo["dz"] = {"value": 0, "lb": -1, "ub": 1}  # vertical offset
        self.oppvar = list(self.xo.keys())

        self.segments = None

    def draw(self, **kwargs):
        """
        Calculate the PrincetonD parameterisation shape.

        Returns
        -------
        loop: dict
            The coordinate dictionary of points
        """
        self.npoints = kwargs.get("npoints", self.npoints)
        self.set_input(**kwargs)
        self.segments = {"x": [], "z": []}
        x, z = princetonD(*self.xo.get_value(), self.npoints)
        self.segments["x"].append([x[-1], x[0]])
        self.segments["z"].append([z[-1], z[0]])
        self.segments["x"].append(x)
        self.segments["z"].append(z)

        loop = {"x": x, "z": z}
        loop = self.close_loop(loop)
        loop["x"], loop["z"] = clock(loop["x"], loop["z"])
        return loop


def princetonD(x1, x2, dz, npoints=200):
    """
    Princeton D shape parameterisation (e.g. Gralnick and Tenney, 1976, or
    File, Mills, and Sheffield, 1971)

    Parameters
    ----------
    x1: float
        The inboard centreline radius of the Princeton D
    x2: float
        The outboard centrleine radius of the Princeton D
    dz: float
        The vertical offset (from z=0)
    npoints: int (default = 500)
        The size of the x, z coordinate sets to return

    Returns
    -------
    x: np.array(npoints)
        The x coordinates of the Princeton D shape
    z: np.array(npoints)
        The z coordinates of the Princeton D shape

    Note
    ----
    Returns an open set of coordinates

    :math:`x = X_{0}e^{ksin(\\theta)}`
    :math:`z = X_{0}k\\Bigg[\\theta I_{1}(k)+\\sum_{n=1}^{\\infty}{\\frac{i}{n}
    e^{\\frac{in\\pi}{2}}\\bigg(e^{-in\\theta}-1\\bigg)\\bigg(1+e^{in(\\theta+\\pi)}
    \\bigg)\\frac{I_{n-1}(k)+I_{n+1}(k)}{2}}\\Bigg]`

    Where:
        :math:`X_{0} = \\sqrt{x_{1}x_{2}}`
        :math:`k = \\frac{ln(x_{2}/x_{1})}{2}`

    Where:
        :math:`I_{n}` is the n-th order modified Bessel function
        :math:`x_{1}` is the inner radial position of the shape
        :math:`x_{2}` is the outer radial position of the shape
    """  # noqa (W505)
    if x2 <= x1:
        raise GeometryError(
            "Princeton D parameterisation requires an x2 value"
            f"greater than x1: {x1} >= {x2}"
        )

    xo = np.sqrt(x1 * x2)
    k = 0.5 * np.log(x2 / x1)
    theta = np.linspace(-0.5 * np.pi, 1.5 * np.pi, 2 * npoints)
    s = np.zeros(2 * npoints, dtype="complex128")
    n = 0
    while True:  # sum convergent series
        n += 1

        ds = 1j / n * (np.exp(-1j * n * theta) - 1)
        ds *= 1 + np.exp(1j * n * (theta + np.pi))
        ds *= np.exp(1j * n * np.pi / 2)
        ds *= (bessel(n - 1, k) + bessel(n + 1, k)) / 2
        s += ds
        if np.max(abs(ds)) < 1e-14:
            break

    z = abs(xo * k * (bessel(1, k) * theta + s))
    x = xo * np.exp(k * np.sin(theta))
    z -= np.mean(z)
    x, z = xz_interp(x, z, npoints=npoints)
    z += dz  # vertical shift
    return x, z


def _flatten(x, z):
    """
    Flattens a shape by dragging the lowest and highest point to the minimum
    radius point.
    """
    amin, amax = np.argmin(z), np.argmax(z)
    xmin = np.min(x)
    zmin, zmax = np.min(z), np.max(z)
    xx = np.array(xmin)
    xx = np.append(xx, x[amin:amax])
    xx = np.append(xx, xmin)
    zz = np.array(zmin)
    zz = np.append(zz, z[amin:amax])
    zz = np.append(zz, zmax)
    return xx, zz


def flatD(x1, x2, dz, npoints=200):
    """
    Flattened D shape based on a Princeton D, where the minimum and maximum z
    values are kept constant until the minimum radial position

    Parameters
    ----------
    x1: float
        The inboard centreline radius of the Princeton D
    x2: float
        The outboard centreline radius of the Princeton D
    dz: float
        The vertical offset (from z=0)
    npoints: int (default = 200)
        The size of the x, z coordinate sets to return

    Returns
    -------
    x: np.array(npoints)
        The x coordinates of the Princeton D shape
    z: np.array(npoints)
        The z coordinates of the Princeton D shape

    Note
    ----
    Returns an open set of coordinates
    """
    return _flatten(*princetonD(x1, x2, dz, npoints))


def negativeD(x1, x2, dz, npoints=200):
    """
    Negative D shape based on a Princeton D, for negative kappas

    Parameters
    ----------
    x1: float
        The inboard centreline radius of the Princeton D
    x2: float
        The outboard centreline radius of the Princeton D
    dz: float
        The vertical offset (from z=0)
    npoints: int (default = 200)
        The size of the x, z coordinate sets to return

    Returns
    -------
    x: np.array(npoints)
        The x coordinates of the Princeton D shape
    z: np.array(npoints)
        The z coordinates of the Princeton D shape

    Notes
    -----
    This is not a constant tension shape..!

    Returns a closed set of coordinates
    """
    x, z = princetonD(x1, x2, dz, npoints)
    x = np.append(x, x[0])
    z = np.append(z, z[0])

    xmax = np.max(x)
    y = np.zeros_like(x)
    r = qrotate(np.array([x, y, z]).T, theta=np.pi, p1=[0, 0, 0], p2=[0, 1, 0]).T
    x, z = r[0], r[2]
    xmax2 = np.max(x)
    x += xmax - xmax2
    arg = np.argmin(x)
    x = np.roll(np.roll(x[:-1], arg), 0)
    z = np.roll(np.roll(z[:-1], arg), 0)
    x = np.append(x, x[0])
    z = np.append(z, z[0])
    return x, z


def negativeflatD(x1, x2, dz, npoints=200):
    """
    Negative flattened D shape based on a Princeton D, for negative kappas.
    More appropriate for rectangular TFs.

    Parameters
    ----------
    x1: float
        The inboard centreline radius of the Princeton D
    x2: float
        The outboard centreline radius of the Princeton D
    dz: float
        The vertical offset (from z=0)
    npoints: int (default = 200)
        The size of the x, z coordinate sets to return

    Returns
    -------
    x: np.array(npoints)
        The x coordinates of the Princeton D shape
    z: np.array(npoints)
        The z coordinates of the Princeton D shape

    Notes
    -----
    Returns an open set of coordinates
    """
    return _flatten(*negativeD(x1, x2, dz, npoints=npoints))


class PolySpline(Parameterisation):  # polybezier
    """
    Poly Bezier spline Nova Loop object
    """

    name = "PolySpline"

    def __init__(self, npoints=200, symmetric=False, tension="full", **kwargs):
        super().__init__(npoints=npoints)
        self.symmetric = symmetric
        self.tension = tension

        self.lkey = None
        self.lkeyo = None
        self.length = None
        self.po = None
        self.initialise()

    def initialise(self):
        """
        Initialise the PolySpline shape with some default values and bounds.
        """
        self.xo["x1"] = {"value": 4.3, "lb": 4.2, "ub": 4.5}
        self.xo["x2"] = {"value": 16.56, "lb": 5, "ub": 25}  # outer radius
        self.xo["z2"] = {
            "value": 0.03,
            "lb": -1.9,
            "ub": 1.9,
        }  # outer node vertical shift
        self.xo["height"] = {"value": 15.5, "lb": 10, "ub": 50}  # full loop height
        self.xo["top"] = {"value": 0.52, "lb": 0.2, "ub": 1}  # horizontal shift
        self.xo["upper"] = {"value": 0.67, "lb": 0.1, "ub": 1}  # vertical shift
        self.xo["dz"] = {"value": -0.6, "lb": -5, "ub": 5}  # vertical offset
        # fraction outboard straight
        self.xo["flat"] = {"value": 0, "lb": 0, "ub": 0.8}
        # outboard angle [deg]
        self.xo["tilt"] = {"value": 4, "lb": -45, "ub": 45}
        self.set_lower()  # lower loop parameters (bottom,lower)
        self.xo["bottom"] = {"value": 0.4, "lb": 0.1, "ub": 1}
        self.oppvar = list(self.xo.keys())
        self.lkeyo = ["l0s", "l0e", "l1s", "l1e", "l2s", "l2e", "l3s", "l3e"]
        self._set_l({"value": 0.8, "lb": 0.1, "ub": 1.9})  # 1/tesion

        self.set_symmetric()
        self.set_tension()

    def reset_oppvar(self, symmetric):
        """
        Reset the optimisation variables for the Parameterisation.
        """
        self.initialise()
        self.oppvar = list(self.xo.keys())
        self.symmetric = symmetric
        self.set_symmetric()
        self.set_tension()

    def adjust_xo(self, name, **kwargs):
        """
        Adjust the optimisation variables of the parameterisation.

        Parameters
        ----------
        name: str
            The variable name

        Other Parameters
        ----------------
        value: float
            The value to set
        lb: float
            The lower bound to set
        ub: float
            The upper bound to set
        """
        super().adjust_xo(name, **kwargs)
        # self.set_lower()
        self.set_symmetric()
        self.set_tension()

    def set_lower(self):
        """
        Set up-down values based on upper values.
        """
        for upp, low in zip(["top", "upper"], ["bottom", "lower"]):
            self.xo[low] = {}
            for key in self.xo[upp]:
                self.xo[low][key] = self.xo[upp][key]

    def enforce_symmetric(self):
        """
        Enforce up-down symmetry in the PolySpline parameterisation.
        """
        self.symmetric = True
        self.set_symmetric()
        self.set_tension()

    def set_symmetric(self):
        """
        Set the PolySpline symmetry constraint, is symmetric.
        """
        if self.symmetric:  # set lower equal to upper
            self.xo["tilt"]["value"] = 0
            self.xo["z2"]["value"] = self.xo["dz"]["value"]
            if "z2" in self.oppvar:
                self.oppvar.remove("z2")
            for upp, low in zip(["top", "upper"], ["bottom", "lower"]):
                self.xo[low] = self.xo[upp]
                if low in self.oppvar:  # remove lower from oppvar
                    self.oppvar.remove(low)
            if "tilt" in self.oppvar:
                self.oppvar.remove("tilt")

    def set_tension(self):
        """
        Set the tension in the PolySpline.
        """
        tension = self.tension.lower()
        if self.symmetric:
            if tension == "full":
                self.tension = "half"
            elif tension == "half":
                self.tension = "dual"
            else:
                self.tension = tension
        else:
            self.tension = tension

        if self.tension == "single":
            self.lkey = ["l", "l", "l", "l", "l", "l", "l", "l"]
        elif self.tension == "dual":
            self.lkey = ["l0", "l0", "l1", "l1", "l1", "l1", "l0", "l0"]
        elif self.tension == "half":
            if self.symmetric:
                self.lkey = ["l0s", "l0e", "l1s", "l1e", "l1e", "l1s", "l0e", "l0s"]
            else:
                self.lkey = ["l0", "l0", "l1", "l1", "l2", "l2", "l3", "l3"]
        elif self.tension == "full":
            self.lkey = ["l0s", "l0e", "l1s", "l1e", "l2s", "l2e", "l3s", "l3e"]
        if not self.tension == "full":
            # NOTE: This section of code is responsible for mutating "l" keys..
            # This destroys read/write replication (as the variables change)
            oppvar = np.copy(self.oppvar)  # remove all length keys
            for var in oppvar:
                if var[0] == "l" and var != "lower":
                    self.oppvar.remove(var)
            for oppkey in np.unique(self.lkey):  # re-populate
                self.oppvar.append(oppkey)
            for i, (lkey, lkeyo) in enumerate(zip(self.lkey, self.lkeyo)):
                if len(lkey) == 1:
                    lkey += "0s"
                elif len(lkey) == 2:
                    lkey += "s"
                self.xo[lkeyo] = self.xo[lkey].copy()

    def get_xo(self):
        """
        Get the all the values of the PolySpline OptimisationVariables, regardless
        of whether they have been removed from the optimisation problem.

        Returns
        -------
        values: list
            The values of all the Parameterisation variables
        """
        values = []
        for var in [
            "x1",
            "x2",
            "z2",
            "height",
            "top",
            "bottom",
            "upper",
            "lower",
            "dz",
            "flat",
            "tilt",
        ]:
            if var not in self.xo:
                if var == "bottom":
                    var = "top"
                if var == "lower":
                    var = "upper"
            values.append(self.xo[var]["value"])
        return values

    def _set_l(self, l_key):
        for lkey in self.lkeyo:
            self.xo[lkey] = l_key.copy()

    def _get_l(self):
        ls, le = [], []  # start, end
        for i in range(4):
            ls.append(self.xo["l{:1.0f}s".format(i)]["value"])
            le.append(self.xo["l{:1.0f}e".format(i)]["value"])
        return ls, le

    @staticmethod
    def basis(t, v):
        """
        Calculate the basis function of a spline section.
        """
        n = 3  # spline order
        return binom(n, v) * t ** v * (1 - t) ** (n - v)

    @staticmethod
    def midpoints(p):
        """
        Convert polar coordinates to cartesian.
        """
        x = p["x"] + p["l"] * np.cos(p["t"])
        z = p["z"] + p["l"] * np.sin(p["t"])
        return x, z

    def control(self, p0, p3):
        """
        Add control points (length and theta or midpoint).
        """
        p1, p2 = {}, {}
        xm, ym = np.mean([p0["x"], p3["x"]]), np.mean([p0["z"], p3["z"]])
        dl = sp.linalg.norm([p3["x"] - p0["x"], p3["z"] - p0["z"]])
        for p, pm in zip([p0, p3], [p1, p2]):
            if "l" not in p:  # add midpoint length
                p["l"] = dl / 2
            else:
                p["l"] *= dl / 2
            if "t" not in p:  # add midpoint angle
                p["t"] = np.arctan2(ym - p["y"], xm - p["x"])
            pm["x"], pm["z"] = self.midpoints(p)
        return p1, p2, dl

    def _append_keys(self, key):
        if key[0] == "l" and key != "lower":
            length = len(key)
            if length == 1:
                key = self.lkeyo  # control all nodes
            elif length == 2:
                key = [key + "s", key + "e"]
            else:
                key = [key]
        else:
            key = [key]
        return key

    def set_input(self, **kwargs):
        """
        Set the active optimisation variables in the PolySpline.
        """
        inputs = self.get_input(oppvar=self.oppvar, **kwargs)
        for key in inputs:
            keyo = self._append_keys(key)
            for keyo_ in keyo:
                if keyo_ in self.xo.keys():
                    try:  # dict
                        for k in inputs[key]:
                            self.xo[keyo_][k] = inputs[key][k]
                            print("y")
                    except TypeError:  # single value - object is not iterable
                        self.xo[keyo_]["value"] = inputs[key]
                    self.xo.set_limit(keyo_)
        self.set_symmetric()
        self.set_tension()

    def verticies(self):
        """
        Calculate the PolySpine Bezier vertices.
        """
        (
            x1,
            x2,
            z2,
            height,
            top,
            bottom,
            upper,
            lower,
            dz,
            ds,
            alpha_s,
        ) = self.get_xo()
        x, z, theta = np.zeros(6), np.zeros(6), np.zeros(6)
        alpha_s *= np.pi / 180
        ds_z = ds * height / 2 * np.cos(alpha_s)
        ds_r = ds * height / 2 * np.sin(alpha_s)
        x[0], z[0], theta[0] = x1, upper * height / 2, np.pi / 2  # upper sholder
        x[1], z[1], theta[1] = x1 + top * (x2 - x1), height / 2, 0  # top
        #  outer, upper
        x[2], z[2], theta[2] = (
            x2 + ds_r,
            z2 * height / 2 + ds_z,
            -np.pi / 2 - alpha_s,
        )
        # outer, lower
        x[3], z[3], theta[3] = (
            x2 - ds_r,
            z2 * height / 2 - ds_z,
            -np.pi / 2 - alpha_s,
        )
        x[4], z[4], theta[4] = x1 + bottom * (x2 - x1), -height / 2, -np.pi  # bottom
        x[5], z[5], theta[5] = x1, -lower * height / 2, np.pi / 2  # lower sholder
        z += dz  # vertical loop offset
        return x, z, theta

    def linear_loop_length(self, x, z):
        """
        Get the length of the PolySpline shape.
        """
        self.length = 0
        for i in range(len(x) - 1):
            self.length += sp.linalg.norm([x[i + 1] - x[i], z[i + 1] - z[i]])

    def segment(self, p, dl):
        """
        Calculate a Bezier spline segment.
        """
        n = int(np.ceil(self.npoints * dl / self.length))  # segment point number
        t = np.linspace(0, 1, n)
        curve = {"x": np.zeros(n), "z": np.zeros(n)}
        for i, pi in enumerate(p):
            for var in ["x", "z"]:
                curve[var] += self.basis(t, i) * pi[var]
        return curve

    def polybezier(self, x, z, theta):
        """
        Calculate the underlying connected Bezier splines.
        """
        p = {"x": np.array([]), "z": np.array([])}
        self.po = []
        self.linear_loop_length(x, z)
        ls, le = self._get_l()
        for i, j, k in zip(range(len(x) - 1), [0, 1, 3, 4], [1, 2, 4, 5]):
            p0 = {"x": x[j], "z": z[j], "t": theta[j], "l": ls[i]}
            p3 = {"x": x[k], "z": z[k], "t": theta[k] - np.pi, "l": le[i]}
            p1, p2, dl = self.control(p0, p3)
            curve = self.segment([p0, p1, p2, p3], dl)
            self.po.append({"p0": p0, "p1": p1, "p2": p2, "p3": p3})
            for var in ["x", "z"]:
                p[var] = np.append(p[var], curve[var][:-1])
        for var in ["x", "z"]:
            p[var] = np.append(p[var], curve[var][-1])
            p[var] = p[var][::-1]
        return p

    def draw(self, **kwargs):
        """
        Calculate the PolySpline parameterisation shape.

        Returns
        -------
        p: dict
            The coordinate dictionary of points
        """
        self.npoints = kwargs.get("npoints", self.npoints)
        self.set_input(**kwargs)
        x, z, theta = self.verticies()
        p = self.polybezier(x, z, theta)
        p = self.close_loop(p)
        p["x"], p["z"] = clock(p["x"], p["z"])
        return p

    def plot(self, inputs={}, ax=None, ms=8):
        """
        Plot the PolySpline parameterisation.
        """
        if ax is None:
            _, ax = plt.subplots()
        p = self.draw(inputs=inputs)
        x, z, theta = self.verticies()
        c1, c2 = 0.75 * np.ones(3), 0.4 * np.ones(3)
        ax.plot(x, z, "s", color=c1, ms=2 * ms, zorder=10)
        ax.plot(x, z, "s", color=c2, ms=ms, zorder=10)
        ax.plot(p["x"], p["z"], "-", color=c2, ms=ms)
        self._plot_po(ax)
        ax.set_aspect("equal")

    def _plot_po(self, ax, points=None, ms=8):
        c1, c2 = 0.75 * np.ones(3), 0.4 * np.ones(3)
        if points is None:
            points = self.po
        for po in points:
            ax.plot(
                [po["p0"]["x"], po["p1"]["x"]],
                [po["p0"]["z"], po["p1"]["z"]],
                color=c1,
                ms=ms,
                zorder=5,
            )
            ax.plot(po["p1"]["x"], po["p1"]["z"], "o", color=c1, ms=2 * ms, zorder=6)
            ax.plot(po["p1"]["x"], po["p1"]["z"], "o", color=c2, ms=ms, zorder=7)
            ax.plot(
                [po["p3"]["x"], po["p2"]["x"]],
                [po["p3"]["z"], po["p2"]["z"]],
                color=c1,
                ms=ms,
                zorder=5,
            )
            ax.plot(po["p2"]["x"], po["p2"]["z"], "o", color=c1, ms=2 * ms, zorder=6)
            ax.plot(po["p2"]["x"], po["p2"]["z"], "o", color=c2, ms=ms, zorder=7)
        ax.set_aspect("equal")


class BackwardPolySpline(PolySpline):  # polybezier
    """
    Backwards poly Bezier spline Nova Loop object
    """

    name = "BackwardPolySpline"

    def __init__(self, npoints=200, symmetric=False, tension="full", **kwargs):
        super().__init__(npoints=npoints, symmetric=symmetric, tension=tension)

    def draw(self, **kwargs):
        """
        Calculate the BackwardPolySpline parameterisation shape.

        Returns
        -------
        p: dict
            The coordinate dictionary of points
        """
        p = super().draw(**kwargs)
        x = p["x"]
        delta = np.max(x) + np.min(x)
        x = -x + delta
        # Need to reverse the points in order for geometric_objective to work
        # the same way
        p["x"] = x[::-1]
        p["z"] = p["z"][::-1]
        return p

    def plot(self, inputs={}, ax=None, ms=8):
        """
        Plot the BackwardPolySpline parameterisation.
        """
        if ax is None:
            _, ax = plt.subplots()
        p = self.draw(inputs=inputs)
        x, z, theta = self.verticies()

        delta = np.max(x) + np.min(x)

        def mirror(xp):
            return -xp + delta

        c1, c2 = 0.75 * np.ones(3), 0.4 * np.ones(3)
        ax.plot(mirror(x), z, "s", color=c1, ms=2 * ms, zorder=10)
        ax.plot(mirror(x), z, "s", color=c2, ms=ms, zorder=10)
        ax.plot(p["x"], p["z"], "-", color=c2, ms=ms)

        points = self.po.copy()
        for po in points:
            for k in po:
                if k in ["p0", "p1", "p2", "p3"]:
                    po[k]["x"] = mirror(po[k]["x"])

        self._plot_po(ax, points)
        ax.set_aspect("equal")


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
