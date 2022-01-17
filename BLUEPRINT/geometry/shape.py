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
Base shape object for optimisations with parameterised shapes
"""
import os

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.file import get_bluemira_path, make_bluemira_path
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.geometry.error import GeometryError
from BLUEPRINT.geometry.geombase import JSONReaderWriter
from BLUEPRINT.geometry.geomtools import length, loop_volume, normal
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.optimiser import ShapeOptimiser
from BLUEPRINT.geometry.parameterisations import (
    BackwardPolySpline,
    CurvedPictureFrame,
    PictureFrame,
    PolySpline,
    PrincetonD,
    TaperedPictureFrame,
    TripleArc,
)

# fmt: off
PARAMETERISATION_MAP = {
    "S": PolySpline,           # Poly-Bezier (8-16 parameter)
    "A": TripleArc,            # Triple arc (5-7 parameter)
    "P": PictureFrame,         # Picture frame (4-5 parameter)
    "D": PrincetonD,           # Princeton D (3 parameter)
    "BS": BackwardPolySpline,  # Reversed Poly-Bezier spline (8-16 parameter)
    "TP": TaperedPictureFrame,  # Tapered Picture frame (7-8 parameter)
    "CP": CurvedPictureFrame,  # Picture frame coil with a curved top/bottom, top/down symmetric
}
# fmt: on


class Shape(JSONReaderWriter):
    """
    Shape object for use in parameterisations and shape optimisations

    Parameters
    ----------
    name: str
        The name of the shape
    family: str from ['S', 'A', 'P', 'D', 'BS', 'L', 'TP', 'CP']
        The type of shape parameterisation to use
    objective: str from ['L', 'V']
        The optimisation objective to use (can be overidden)
    npoints: int (default = 200)
        The number of points to use when drawing the shape
    read_write: bool (default = False)
        Whether or not to load/write to files
    read_directory: str (default = None)
        The full path directory name to read data from
    write_directory: str (default = None)
        The full path directory name to write data to

    Other Parameters
    ----------------
    nTF: int
        The number of TF coils
    symmetric: bool
        Symmetry flag (only valid for some parameterisations)
    """

    def __init__(
        self,
        name,
        family,
        objective="L",
        npoints=200,
        symmetric=False,
        read_write=False,
        read_directory=None,
        write_directory=None,
        **kwargs,
    ):
        self.read_write = read_write
        self.n_TF = kwargs.get("n_TF", "unset")

        # Constructors
        self.family = None
        self.parameterisation = None
        self.optimiser = None
        self.symmetric = None
        self.npoints = npoints
        self.xo = None
        self._f_obj_norm = None

        if objective not in ["L", "V", "max_L", "E"]:
            raise GeometryError(f'Shape objective "{objective}" not defined.')
        self.objective = objective

        if isinstance(self.n_TF, str):
            n_TF = self.n_TF
        else:
            n_TF = str(int(self.n_TF))

        name = "_".join([name, family, objective, n_TF])

        if read_write:
            if read_directory is None:
                read_directory = get_bluemira_path(
                    "geometry_data", subfolder="data/BLUEPRINT"
                )
            if write_directory is None:
                make_bluemira_path("generated_data/BLUEPRINT", subfolder="")
                write_directory = make_bluemira_path(
                    "geometry_data", subfolder="generated_data/BLUEPRINT"
                )
            self.read_filename = os.sep.join([read_directory, name + ".json"])
            self.write_filename = os.sep.join([write_directory, name + ".json"])

        self.update(npoints=npoints, symmetric=symmetric, family=family)
        self.bound = {}  # initialise bounds
        self.bindex = {"internal": [0], "interior": [0], "external": [0]}
        for side in ["internal", "interior", "external"]:
            self.bound[side] = {"x": [], "z": []}
            if side in kwargs:
                self.add_bound(kwargs[side], side)

        # Define optimisation defaults (to be over-ridden if desired)
        self.f_objective = self.geometric_objective
        self.f_ieq_constraints = self.geometric_constraints
        self.f_eq_constraints = None
        self.args = ()

    def update(self, **kwargs):
        """
        Assigns some class attributes and initialises parameterisation
        """
        for key in ["npoints", "symmetric", "family", "n_TF"]:
            if key in kwargs:
                setattr(self, key, kwargs[key])
        self.initialise_parameterisation()

    def initialise_parameterisation(self):
        """
        Assigns a parameterisation to the shape object
        """
        try:
            para_class = PARAMETERISATION_MAP[self.family]
        except KeyError:
            errtxt = f"\nparameterisation type '{self.family}' not recognised\n"
            errtxt += f"select from: {list(PARAMETERISATION_MAP.keys())}\n"
            raise GeometryError(errtxt)

        kwargs = {"npoints": self.npoints, "symmetric": self.symmetric}

        self.parameterisation = para_class(**kwargs)

        # Get the initial value of the geometry objective to normalise
        # (this helps the optimisation algorithms)
        geom = self.parameterisation.draw()

        if self.objective == "L":
            self._f_obj_norm = length(geom["x"], geom["z"])[-1]
        elif self.objective == "V":
            self._f_obj_norm = loop_volume(geom["x"], geom["z"])
        elif self.objective == "max_L":
            self._f_obj_norm = -length(geom["x"], geom["z"])[-1]

    def adjust_xo(self, name, **kwargs):
        """
        Adjusts a shape parameterisation variable

        Parameters
        ----------
        name: str
            The key of the optimisation variable to adjust
        kwargs: ['lb, 'value', 'ub']
            lb: float
                Lower bound of the optimisation variable
            value: float
                The default value upon initialisation of the optimisation
            'ub': float
                Upper bound of the optimisation variables
        """
        self.parameterisation.adjust_xo(name, **kwargs)

    def remove_oppvar(self, key):
        """
        Removes a variable from the optimisation variables

        Parameters
        ----------
        key: str
            The key of the optimisation variable to remove
        """
        self.parameterisation.remove_oppvar(key)

    def add_bound(self, p, side):
        """
        Add a bound to the Shape. For use in constrained optimisation.
        """
        for var in ["x", "z"]:
            self.bound[side][var] = np.append(self.bound[side][var], p[var])
        self.bindex[side].append(len(self.bound[side]["x"]))

    def clear_bound(self):
        """
        Clear the bounds in the Shape.
        """
        for side in self.bound:
            for var in ["x", "z"]:
                self.bound[side][var] = np.array([])

    def plot_bounds(self, ax=None):
        """
        Plots the geometric bounds of the shape optimisation problem
        """
        if self.bound == {}:
            bluemira_warn("Geometry::Shape No bounds to plot...")
            return

        if ax is None:
            _, ax = plt.subplots()

        for side, marker in zip(["internal", "interior", "external"], [".-", "d", "s"]):
            index = self.bindex[side]
            for i in range(len(index) - 1):
                ax.plot(
                    self.bound[side]["x"][index[i] : index[i + 1]],
                    self.bound[side]["z"][index[i] : index[i + 1]],
                    marker,
                    markersize=6,
                )

    def geometric_objective(self, xnorm, *args):
        """
        Default geometric objective function for Shape optimisation.

        Parameters
        ----------
        xnorm: np.array
            The normalised vector of shape parameterisation variables
        args: tuple
            The additional arguments that may be passed from the optimiser

        Returns
        -------
        value: float
            The value of the selected objective function
        """
        xo = self.parameterisation.get_oppvar(xnorm)  # de-normalize
        if self.xo is not None:
            self.xo = np.vstack([self.xo, xo])
        else:
            self.xo = xo
        x = self.parameterisation.draw(x=xo)

        if self.objective == "L":  # loop length
            value = length(x["x"], x["z"])[-1]
        elif self.objective == "V":  # loop volume (torus)
            value = loop_volume(x["x"], x["z"])
        elif self.objective == "max_L":
            value = -length(x["x"], x["z"])[-1]
        else:
            raise GeometryError(
                f"Shape objective {self.objective} not defined"
                " within geometric_objective function"
            )

        # Normalise the value of the objective function so that it is close to 1.
        return value / self._f_obj_norm

    def dot_difference(self, p, side):
        """
        Utility function for geometric constraints.
        """
        xloop, zloop = p["x"], p["z"]  # inside coil loop
        switch = 1 if side == "internal" else -1
        n_xloop, n_zloop = normal(xloop, zloop)
        x_bound, z_bound = self.bound[side]["x"], self.bound[side]["z"]
        dot = np.zeros(len(x_bound))
        for j, (x, z) in enumerate(zip(x_bound, z_bound)):
            i = np.argmin((x - xloop) ** 2 + (z - zloop) ** 2)
            dl = [xloop[i] - x, zloop[i] - z]
            dn = [n_xloop[i], n_zloop[i]]
            dot[j] = switch * np.dot(dl, dn)
        return dot

    def geometric_constraints(self, xnorm, *args):
        """
        Default geometric constraints function

        Parameters
        ----------
        xnorm: np.array
            The normalised vector of shape parameterisation variables
        args: tuple
            The additional arguments that may be passed from the optimiser

        Returns
        -------
        constraint: np.array
            The array of constraints to be passed to the optimiser
        """
        xo = self.parameterisation.get_oppvar(xnorm)  # de-normalize
        p = self.parameterisation.draw(x=xo)
        constraint = np.array([])
        for side in ["internal", "interior", "external"]:
            constraint = np.append(constraint, self.dot_difference(p, side))

        return constraint

    @property
    def n_geom_constraints(self):
        """
        The number of geometric constraints

        Returns
        -------
        n: int
            The number of geometric constraints applied to the Shape
        """
        n = 0
        for v in self.bound.values():
            n += len(v["x"])
        return n

    def f_update(self, xnorm, *args):
        """
        Empty function - overloaded externally
        """
        return 0

    def optimise(self, verbose=False, algorithm=None, **opt_kwargs):
        """
        Optimise the shape based on the specified objectives and constraints

        Parameters
        ----------
        verbose: bool
            The verbosity of the underlying optimisation algorithm
        algorithm: str
            The algorithm to use to optimise the Shape.

        See Also
        --------
        geometry/optimisation.py for details on default algorithms and opt_kwargs
        """
        self.optimiser = ShapeOptimiser(
            self.parameterisation,
            self.f_objective,
            f_ieq_constraints=self.f_ieq_constraints,
            f_eq_constraints=self.f_eq_constraints,
            args=self.args,
            algorithm=algorithm,
            **opt_kwargs,
        )

        x_norm = self.optimiser(verbose=verbose)

        # Sanity...
        self.parameterisation = self.optimiser.parameterisation

        # De-normalize variable vector
        x_star = self.parameterisation.get_oppvar(x_norm)
        # Update parameterisation with result
        self.parameterisation.set_input(x=x_star)
        self.f_update(x_norm)

    def get_loop(self, plan_dims=None):
        """
        Get a Loop for the Shape in its current state.

        Parameters
        ----------
        plan_dims: Union[None, List[str]]
            The planar dimensions of the Loop. Defaults to ["x", "z"]

        Returns
        -------
        loop: Loop
            The Loop for the Shape in its current state
        """
        if plan_dims is None:
            plan_dims = ["x", "z"]

        p = self.parameterisation.draw()
        d = {k: v for (k, v) in zip(plan_dims, p.values())}
        return Loop(**d)

    def load(self):
        """
        Attempts to load a shape from a JSON file
        """
        if self.read_write:
            if os.path.isfile(self.read_filename):
                self.loop_dict = super().load()
            else:
                bluemira_warn(
                    f"Geometry::Shape file {self.read_filename} not found\n"
                    "initializing new loop_dict"
                )
                self.loop_dict = {}
        else:
            self.loop_dict = {}
        self.update(**self.loop_dict)

        from BLUEPRINT.geometry.parameterisations import OptVariables

        try:
            xo = OptVariables(self.loop_dict["xo"])
        except KeyError:
            raise GeometryError("No relevant shape save file to load.")
        self.parameterisation.xo = xo.copy()
        for k, v in self.loop_dict.items():
            if k in ["oppvar", "symmetric", "tension", "limits"]:
                if hasattr(self.parameterisation, k):
                    setattr(self.parameterisation, k, v)

    def write(self):
        """
        Writes the Shape to a JSON file
        """
        cdict = {}
        for key in ["xo", "oppvar", "symmetric", "tension", "limits"]:
            if hasattr(self.parameterisation, key):
                cdict[key] = getattr(self.parameterisation, key)
        for key in ["family", "objective", "n_TF"]:
            cdict[key] = getattr(self, key)
        if self.read_write:  # write loop to file
            super().write(cdict)


def fit_shape_to_loop(shape_type, loop, n_points=100):
    """
    Fit a Shape to an existing Loop.

    Parameters
    ----------
    shape_type: str
        The type of shape parameterisation to fit to the Loop
    loop: Loop
        The geometry to fit the Shape to
    n_points: int
        The number of points in the Loop and Shape

    Returns
    -------
    shape: Shape
        The fitted shape
    """

    def least_squares(x_norm, *args):
        x_star = shape.parameterisation.get_oppvar(x_norm)
        # Update parameterisation with result
        shape.parameterisation.set_input(x=x_star)
        x1, z1 = shape.parameterisation.draw().values()
        x2, z2 = args
        diff = np.sqrt((x2 - x1) ** 2 + (z2 - z1) ** 2)
        return np.linalg.norm(diff, 2, axis=0)

    def length_diff(x_norm, *args):
        x_star = shape.parameterisation.get_oppvar(x_norm)
        # Update parameterisation with result
        shape.parameterisation.set_input(x=x_star)
        x1, z1 = shape.parameterisation.draw().values()
        _, _ = args
        return 0.05 * loop.length - abs(length(x1, z1) - loop.length) / loop.length

    loop = loop.copy()
    shape = Shape("fitting", shape_type, npoints=n_points)

    # Interpolate the loop for even spacing
    loop.interpolate(n_points)

    x, z = shape.parameterisation.draw().values()

    # Roll the loop so the 0 of the shape corresponds to the 0 of the loop
    arg = loop.argmin([x[0], z[0]])
    loop.reorder(0, arg)

    # Store the old objective function and arguments
    f_objective = shape.f_objective
    f_constraints = shape.f_ieq_constraints
    o_args = shape.args

    # This seriously helps reduce the problem, and is the only variable that is
    # shared in all the parameterisations
    shape.adjust_xo("x1", value=np.min(loop.x))

    # Set a new objective and run the fit
    shape.f_objective = least_squares
    shape.f_ieq_constraints = None
    shape.f_eq_constraints = None
    shape.args = loop.x, loop.z
    shape.optimise()

    # Keep fitting if the bounds have been hit (pseudo-unbounded)
    repeat = True

    while repeat:
        repeat = False

        for k, var in shape.parameterisation.xo.items():
            # If bounds hit, relax the bound
            width = var["ub"] - var["lb"]
            if np.isclose(var["value"], var["lb"]):
                shape.adjust_xo(k, lb=var["lb"] - width / 2)
                repeat = True

            elif np.isclose(var["value"], var["ub"]):
                shape.adjust_xo(k, ub=var["ub"] + width / 2)
                repeat = True

        # if shape.optimiser.result["fun"] > 0.1:
        #     repeat = True

        if repeat:
            shape.optimise(algorithm="SLSQP", ftol=1e-4)

    # Restore the old objective function
    shape.f_objective = f_objective
    shape.f_ieq_constraints = f_constraints
    shape.args = o_args
    return shape
