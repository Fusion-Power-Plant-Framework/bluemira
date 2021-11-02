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
Generic miscellaneous tools, including some amigo port-overs
"""
import numpy as np
from scipy.spatial.distance import cdist
import re
from collections import OrderedDict
from collections.abc import Mapping, Iterable
from typing import List, Union

from bluemira.base.constants import ABS_ZERO_C, ABS_ZERO_K
from bluemira.base.look_and_feel import bluemira_print, bluemira_warn
from bluemira.utilities.tools import is_num


CROSS_P_TOL = 1e-14  # Cross product tolerance


class PowerLawScaling:
    """
    Simple power law scaling object, of the form:

    \t:math:`c~\\pm~cerr \\times {a_{1}}^{n1\\pm err1}{a_{2}}^{n2\\pm err2}...`

    if cerr is specified, or of the form:

    \t:math:`ce^{\\pm cexperr} \\times {a_{1}}^{n1\\pm err1}{a_{2}}^{n2\\pm err2}...`

    Parameters
    ----------
    c: float
        The constant of the equation
    cerr: float
        The error on the constant
    cexperr: Union[float, None]
        The exponent error on the constant (cannot be specified with cerr)
    exponents: Union[np.array, List, None]
        The ordered list of exponents
    err: Union[np.array, List, None]
        The ordered list of errors of the exponents
    """  # noqa (W505)

    def __init__(self, c=1, cerr=0, cexperr=None, exponents=None, err=None):
        self._len = len(exponents)
        self.c = c
        self.cerr = cerr
        self.cexperr = cexperr
        self.exponents = np.array(exponents)
        self.errors = np.array(err)

    def __call__(self, *args):
        """
        Call the PowerLawScaling object for a set of arguments.
        """
        if len(args) != len(self):
            raise ValueError(
                "Number of arguments should be the same as the "
                f"power law length. {len(args)} != {len(self)}"
            )
        return self.calculate(*args)

    def calculate(self, *args, exponents=None):
        """
        Call the PowerLawScaling object for a set of arguments.
        """
        if exponents is None:
            exponents = self.exponents
        return self.c * np.prod(np.power(args, exponents))

    def error(self, *args):
        """
        Calculate the error of the PowerLawScaling for a set of arguments.
        """
        if self.cexperr is None:
            c = [(self.c + self.cerr) / self.c, (self.c - self.cerr) / self.c]
        else:
            if self.cerr != 0:
                bluemira_warn("PowerLawScaling object overspecified, ignoring cerr.")
            c = [np.exp(self.cexperr), np.exp(-self.cexperr)]
        up = max(c) * self.calculate(*args, exponents=self.exponents + self.errors)
        down = min(c) * self.calculate(*args, exponents=self.exponents - self.errors)
        return self.calculate(*args), min(down, up), max(down, up)

    def __len__(self):
        """
        Get the length of the PowerLawScaling object.
        """
        return self._len


def latin_hypercube_sampling(dimensions: int, samples: int):
    """
    Classic Latin Hypercube sampling function

    Parameters
    ----------
    dimensions: int
        The number of design dimensions
    samples: int
        The number of samples points within the dimensions

    Returns
    -------
    lhs: np.array(samples, dimensions)
        The array of 0-1 normed design points

    Notes
    -----
    Simon's rule of thumb for a good number of samples was that
    samples >= 3**dimensions
    """
    intervals = np.linspace(0, 1, samples + 1)

    r = np.random.rand(samples, dimensions)
    a = intervals[:samples]
    b = intervals[1 : samples + 1]

    points = np.zeros((samples, dimensions))

    for j in range(dimensions):
        points[:, j] = r[:, j] * (b - a) + a

    lhs = np.zeros((samples, dimensions))

    for j in range(dimensions):
        order = np.random.permutation(range(samples))
        lhs[:, j] = points[order, j]

    return lhs


def expand_nested_list(*lists):
    """
    Expands a nested iterable structure, flattening it into one iterable

    Parameters
    ----------
    lists: set of Iterables
        The object(s) to de-nest

    Returns
    -------
    expanded: list
        The fully flattened list of iterables
    """
    expanded = []
    for obj in lists:
        if isinstance(obj, Iterable):
            for o in obj:
                expanded.extend(expand_nested_list(o))
        else:
            expanded.append(obj)
    return expanded


def map_nested_dict(obj, function):
    """
    Wendet eine Funktion auf ein verschachteltes Wörterbuch an

    Parameters
    ----------
    obj: dict
        Nested dictionary object to apply function to
    function: callable
        Function to apply to all non-dictionary objects in dictionary.

    Note
    ----
    In place modification of the dict
    """
    for k, v in obj.items():
        if isinstance(v, Mapping):
            map_nested_dict(v, function)
        else:
            obj[k] = function(v)


def get_max_PF(coil_dict):  # noqa (N802)
    """
    Returns maximum external radius of the largest PF coil
    takes a nova ordered dict of PFcoils
    """
    x = []
    for _, coil in coil_dict.items():
        try:  # New with equilibria
            x.append(coil.x + coil.dx)
        except AttributeError:  # Old nova
            x.append(coil["x"] + coil["dx"] / 2)
    return max(x)


def furthest_perp_point(p1, p2, point_array):
    """
    Returns arg of furthest point from vector p2-p1 in point_array
    """
    v = p2 - p1
    d = v / np.sqrt(v.dot(v))
    dot = np.dot((point_array - p2), d) * d[:, None]
    xyz = p2 + dot.T
    perp_vec = xyz - point_array
    perp_mags = np.linalg.norm(perp_vec, axis=1)
    n = np.argmax(perp_mags)
    return n, perp_mags[n]


def furthest_point_arg(point, loop, coords=None, closest=False):
    """
    Pode ser que esta funcionando mas tenho que confirmar.
    """
    if coords is None:
        coords = ["x", "z"]
    pnts = np.array([loop[coords[0]], loop[coords[1]]]).T
    pnt = np.array([point[0], point[1]]).reshape(2, 1).T
    distances = cdist(pnts, pnt, "euclidean")
    if closest:
        return np.argmin(distances)

    return np.argmax(distances)


def ellipse(a, b, n=100):
    """
    Calculates an ellipse shape

    Parameters
    ----------
    a: float
        Ellipse major radius
    b: float
        Ellipse minor radius
    n: int
        The number of points in the ellipse
    \t:math: `y=\\pm\\sqrt{\\frac{a^2-x^2}{\\kappa^2}}`
    """
    k = a / b
    x = np.linspace(-a, a, n)
    y = ((a ** 2 - x ** 2) / k ** 2) ** 0.5
    x = np.append(x, x[:-1][::-1])
    y = np.append(y, -y[:-1][::-1])
    return x, y


def delta(v2, v1ref):
    """
    Calculates the absolute relative difference between a new value and an old
    reference value.

    Parameters
    ----------
    v2: float
        The new value to compare to the old
    v1ref: float
        The old reference value

    Returns
    -------
    delta: float
        The absolute relative difference between v2 and v1ref
    """
    return abs((v2 - v1ref) / v1ref)


def perc_change(v2, v1ref, verbose=False):
    """
    Calculates the percentage difference between a new value and an old
    reference value

    Parameters
    ----------
    v2: float
        The new value to compare to the old
    v1ref: float
        The old reference value
    verbose: bool
        Whether or not to print information

    Returns
    -------
    perc_change: float
        The percentage difference between v2 and v1ref
    """
    perc = 100 * (v2 - v1ref) / abs(v1ref)
    if verbose:
        if perc < 0:
            change = "decrease"
        else:
            change = "increase"
        bluemira_print("This is a {0} % ".format(round(perc, 3)) + change)
        factor = v2 / v1ref
        if factor < 1:
            fchange = "lower"
        else:
            fchange = "higher"
        bluemira_print("This is " + fchange + f" by a factor of {factor:.2f}")
    return perc


# materials functions
def tokelvin(temp_in_celsius):
    """
    Convert a temperature in Celsius to Kelvin.

    Parameters
    ----------
    temp_in_celsius: Union[float, np.array, List[float]]
        The temperature to convert [°C]

    Returns
    -------
    temp_in_kelvin: Union[float, np.array]
        The temperature [K]
    """
    if (is_num(temp_in_celsius) and temp_in_celsius < ABS_ZERO_C) or np.any(
        np.less(temp_in_celsius, ABS_ZERO_C)
    ):
        raise ValueError("Negative temperature in K specified.")
    return array_or_num(list_array(temp_in_celsius) - ABS_ZERO_C)


def tocelsius(temp_in_kelvin):
    """
    Convert a temperature in Celsius to Kelvin.

    Parameters
    ----------
    temp_in_kelvin: Union[float, np.array, List[float]]
        The temperature to convert [K]

    Returns
    -------
    temp_in_celsius: Union[float, np.array]
        The temperature [°C]
    """
    if (is_num(temp_in_kelvin) and temp_in_kelvin < ABS_ZERO_K) or np.any(
        np.less(temp_in_kelvin, ABS_ZERO_K)
    ):
        raise ValueError("Negative temperature in K specified.")
    return array_or_num(list_array(temp_in_kelvin) + ABS_ZERO_C)


def kgm3togcm3(density):
    """
    Convert a density in kg/m3 to g/cm3

    Parameters
    ----------
    density : Union[float, np.array, List[float]]
        The density [kg/m3]

    Returns
    -------
    density_gcm3 : Union[float, np.array]
        The density [g/cm3]
    """
    if density is not None:
        return array_or_num(list_array(density) / 1000.0)


def gcm3tokgm3(density):
    """
    Convert a density in g/cm3 to kg/m3

    Parameters
    ----------
    density : Union[float, np.array, List[float]]
        The density [g/cm3]

    Returns
    -------
    density_kgm3 : Union[float, np.array]
        The density [kg/m3]
    """
    if density is not None:
        return array_or_num(list_array(density) * 1000.0)


##########################


def print_format_table():
    """
    Prints table of formatted text format options.
    """
    for style in range(0, 10):
        for fg in range(26, 38):
            s1 = ""
            for bg in range(38, 48):
                formatt = ";".join([str(style), str(fg), str(bg)])
                s1 += "\x1b[%sm %s \x1b[0m" % (formatt, formatt)
            print(s1)
        print("\n")


def _apply_rule(a_r, op, b_r):
    return {
        ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b,
        "<": lambda a, b: a < b,
        "<=": lambda a, b: a <= b,
        "=": lambda a, b: a == b,
    }[op](float(a_r), float(b_r))


def _apply_rules(rule):
    if len(rule) == 3:
        return _apply_rule(*rule)
    if len(rule) == 5:
        return _apply_rule(*rule[:3]) and _apply_rule(*rule[2:])


def _split_rule(rule):
    return re.split("([<>=]+)", rule)


def nested_dict_search(odict, rules: Union[str, List[str]]):
    """
    Returns sub-set of nested dictionary which meet str rules for keys in each
    sub-dict.
    Use-case: R.PF.coil searching
    """
    r = []
    if isinstance(rules, str):
        rules = [rules]  # Handles single input if no list
    for rule in rules:
        r.append(_split_rule(rule))
    sub = OrderedDict()
    for n, c in odict.items():
        rules = [[c[i] if i in ["x", "z"] else i for i in j] for j in r]
        if all(_apply_rules(rule) for rule in rules):
            sub[n] = c
    if len(sub) == 0:
        return None
    return sub


def maximum(val, val_min):
    """
    Gets the maximum of val and val_min.
    This function wraps the numpy core umath maximum function
    in order to avoid the standard numpy clip function, as described in:
    https://github.com/numpy/numpy/issues/14281

    Handles scalars using built-ins.

    Parameters
    ----------
    val: scalar or array
        The value to be floored.
    val_min: scalar or array
        The minimum value.

    Returns
    -------
    maximum_val: scalar or array
        The maximum values.
    """
    if isinstance(val, np.ndarray):
        np.core.umath.maximum(val, val_min, out=val)
    else:
        val = val_min if val < val_min else val
    return val


def list_array(list_):
    """
    Always returns a numpy array
    Can handle int, float, list, np.ndarray

    Parameters
    ----------
    list_ : Any
        The value to convert into a numpy array.

    Returns
    -------
    result : np.ndarray
        The value as a numpy array.

    Raises
    ------
    TypeError
        If the value cannot be converted to a numpy array.
    """
    if isinstance(list_, list):
        return np.array(list_)
    elif isinstance(list_, np.ndarray):
        try:  # This catches the odd np.array(8) instead of np.array([8])
            len(list_)
            return list_
        except TypeError:
            return np.array([list_])
    elif is_num(list_):
        return np.array([list_])
    else:
        raise TypeError("Could not convert input type to list_array to a np.array.")


def array_or_num(array):
    """
    Always returns a numpy array or a float

    Parameters
    ----------
    array : Any
        The value to convert into a numpy array or number.

    Returns
    -------
    result : Union[np.ndarray, float]
        The value as a numpy array or number.

    Raises
    ------
    TypeError
        If the value cannot be converted to a numpy or number.
    """
    if is_num(array):
        return float(array)
    elif isinstance(array, np.ndarray):
        return array
    else:
        raise TypeError


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
