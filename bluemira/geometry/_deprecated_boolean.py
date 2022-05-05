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
A collection of Boolean operations, wrapping ClipperLib
"""
import numpy as np
from pyclipper import (
    CT_DIFFERENCE,
    CT_INTERSECTION,
    CT_UNION,
    PFT_EVENODD,
    PT_CLIP,
    PT_SUBJECT,
    CleanPolygon,
    PolyTreeToPaths,
    Pyclipper,
    PyPolyNode,
    SimplifyPolygon,
    scale_from_clipper,
    scale_to_clipper,
)
from scipy.spatial import ConvexHull

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.geometry._deprecated_loop import Loop
from bluemira.geometry.coordinates import get_area_2d
from bluemira.geometry.error import GeometryError

__all__ = [
    "boolean_2d_difference",
    "boolean_2d_union",
    "boolean_2d_xor",
    "boolean_2d_common",
    "convex_hull",
]


def loop_to_pyclippath(loop):
    """
    Transforms a BLUEPRINT Loop object into a Path for use in pyclipper

    Parameters
    ----------
    loop: Loop
        The Loop to be used in pyclipper

    Returns
    -------
    path: [(x1, z1), (x2, z2), ...]
        The vertex polygon path formatting required by pyclipper
    """
    return scale_to_clipper(loop.d2.T)


def shell_to_pyclippath(shell):
    """
    Transforms a BLUEPRINT Shell object into a Paths for use in pyclipper

    Parameters
    ----------
    shell: Shell
        The Shell to be used in pyclipper

    Returns
    -------
    paths: [[(x1, z1), (x2, z2), ...], [(x1, z1), (x2, z2), ...]]
        The vertex polygon paths formatting required by pyclipper
    """
    coordinates = np.append(shell.outer.d2, shell.inner.d2, axis=1)
    dims = shell.plan_dims
    d = {k: v for k, v in zip(dims, coordinates)}
    temp = Loop(**d)
    cut = len(shell.outer)
    path = loop_to_pyclippath(temp)
    return path[: cut - 1], path[cut - 1 :]


def pyclippath_to_loop(path, dims=None):
    """
    Transforms a pyclipper path into a Loop

    Parameters
    ----------
    path: [(x1, z1), (x2, z2), ...]
        The vertex polygon path formatting used in pyclipper
    dims: [str, str] (default = ['x', 'z'])
        The planar dimensions of the Loop. Will default to X, Z

    Returns
    -------
    loop: Loop
        The Loop from the path object
    """
    if dims is None:
        dims = ["x", "z"]
    p2 = scale_from_clipper(np.array(path).T)
    dict_ = {d: p for d, p in zip(dims, p2)}
    return Loop(**dict_)


def pyclippolytree_to_loops(polytree, dims=None):
    """
    Converts a ClipperLib PolyTree into a list of Loops

    Parameters
    ----------
    polytree: ClipperLib::PolyTree
        The polytree to convert to loops
    dims: None or iterable(str, str)
        The dimensions of the Loops
    """
    if dims is None:
        dims = ["x", "z"]
    paths = PolyTreeToPaths(polytree)
    return [pyclippath_to_loop(path, dims) for path in paths]


class PyclipperMixin:
    """
    Mixin class for typical pyclipper operations and processing
    """

    name = NotImplemented

    def perform(self):
        """
        Perform the pyclipper operation
        """
        raise NotImplementedError

    def raise_warning(self):
        """
        Raise a warning if None is to be returned.
        """
        bluemira_warn(
            f"{self.name} operation on 2-D polygons returning None.\n"
            "Nothing to perform."
        )

    def handle_solution(self, solution):
        """
        Handles the output of the Pyclipper.Execute(*) algorithms, turning them
        into Loop objects. NOTE: These are closed by default.

        Parameters
        ----------
        solution: Paths
            The tuple of tuple of tuple of path vertices

        Returns
        -------
        loops: List(Loop)
            The list of Loop objects produced by the Boolean operations
        """
        if not solution:
            self.raise_warning()
            return None
        else:
            loops = []
            if isinstance(solution, PyPolyNode):
                loops = pyclippolytree_to_loops(solution, dims=self.dims)
            else:
                for path in solution:
                    loop = pyclippath_to_loop(path, dims=self.dims)
                    loop.close()
                    loops.append(loop)

        if loops[0].closed:
            return sorted(loops, key=lambda x: -get_area_2d(*x.d2))
        else:
            # Sort open loops by length
            return sorted(loops, key=lambda x: -x.length)

    @property
    def result(self):
        """
        The result of the Pyclipper operation.
        """
        return self._result


class BooleanOperationManager(PyclipperMixin):
    """
    Abstract base class for Boolean operations. Wraps the Pyclipper object

    Parameters
    ----------
    subject: Loop
        The "subject" Loop .base.loop to be preserved)
    clip: Loop
        The "clip" Loop (secondary loop used as a cutter)
    """

    name = NotImplemented
    operation = NotImplemented

    def __init__(self, subject, clip):
        self.flag_open = False  # Defaults to closed loop handling
        self.dims = None
        self.get_dims(subject, clip)
        self.C = Pyclipper()
        self.add_paths(subject, typ=PT_SUBJECT)
        self.add_paths(clip, typ=PT_CLIP)
        self._result = self.perform()

    def get_dims(self, subject, clip):
        """
        Get the dimensions of the subject and clip Loops.

        Parameters
        ----------
        subject: List[Loop]
            The subject Loops
        clip: List[Loop]
            The clip Loops

        Returns
        -------
        dims: List[str]
            The dimensions of the operation

        Raises
        ------
        ValueError
            If the Loops are not on the same plane
        """
        if isinstance(subject, list):
            subject = subject[0]
        if isinstance(clip, list):
            clip = clip[0]
        if subject.plan_dims != clip.plan_dims:
            raise ValueError(
                f"Cannot perform {self.name} operation on polygons that are "
                " not co-planar."
            )

        self.dims = subject.plan_dims

    def add_paths(self, path_list, typ=None):
        """
        Adds paths to the BooleanOperationManager

        Parameters
        ----------
        path_list: List[Path]
            The paths to add
        typ: Union[PT_SUBJECT, PT_CLIP]
            The type of path to add: a subject or clipping path
        """
        from BLUEPRINT.geometry.shell import Shell

        if typ is None:
            typ = PT_SUBJECT

        if not isinstance(path_list, list):
            path_list = [path_list]
        for a in path_list:
            if isinstance(a, Shell):
                paths = shell_to_pyclippath(a)
                self.C.AddPaths(paths, typ, True)
            else:  # Loop
                if not a.closed:
                    self.flag_open = True
                path = loop_to_pyclippath(a)
                self.C.AddPath(path, typ, a.closed)

    def perform(self):
        """
        Perform the pyclipper operation.
        """
        if self.flag_open:
            solution = self.C.Execute2(self.operation, PFT_EVENODD, PFT_EVENODD)
        else:
            solution = self.C.Execute(self.operation, PFT_EVENODD, PFT_EVENODD)
        return self.handle_solution(solution)


class BooleanUnion(BooleanOperationManager):
    """
    Boolean union operation class.
    """

    name = "Boolean Union"
    operation = CT_UNION


class BooleanCommon(BooleanOperationManager):
    """
    Boolean common operation class.
    """

    name = "Boolean Common"
    operation = CT_INTERSECTION


class BooleanDifference(BooleanOperationManager):
    """
    Boolean difference operation class.
    """

    name = "Boolean Difference"
    operation = CT_DIFFERENCE


def boolean_2d_union(loop1, loop2):
    """
    Boolean union operation on 2-D Loop objects

    Parameters
    ----------
    loop1: Loop
        The "subject" Loop .base.loop to be preserved)
    loop2: Loop
        The "clip" Loop (secondary loop used as a cutter)

    Returns
    -------
    loop: Loop
        A single loop of the Boolean union
    """
    return BooleanUnion(loop1, loop2).result


def boolean_2d_common(loop1, loop2):
    """
    Boolean common operation on 2-D Loop objects

    Parameters
    ----------
    loop1: Loop
        The "subject" Loop .base.loop to be preserved)
    loop2: Loop
        The "clip" Loop (secondary loop used as a cutter)

    Returns
    -------
    loop: List(Loop)
        A list of Loops resulting from the Common operation
    """
    return BooleanCommon(loop1, loop2).result


def boolean_2d_difference(loop1, loop2):
    """
    Boolean differece operation on 2-D Loop objects

    Parameters
    ----------
    loop1: Loop
        The "subject" Loop .base.loop to be preserved)
    loop2: Loop
        The "clip" Loop (secondary loop used as a cutter)

    Returns
    -------
    loop: List(Loop)
        A list of Loops resulting from the Difference operation
    """
    return BooleanDifference(loop1, loop2).result


def boolean_2d_xor(loop1, loop2):
    """
    Boolean exclusive-or operation on 2-D Loop objects

    Parameters
    ----------
    loop1: Loop
        The "subject" Loop .base.loop to be preserved)
    loop2: BLUEPRINT Loop
        The "clip" Loop (secondary loop used as a cutter)

    Returns
    -------
    loop: List(Loop)
        A list of Loops resulting from the XOR operation
    """
    return boolean_2d_difference(loop1, loop2) + boolean_2d_difference(loop2, loop1)


def convex_hull(loops):
    """
    Make a convex hull from some co-planar Loops.

    Parameters
    ----------
    loops: List[Loop]
        The list of Loops from which to make a convex hull

    Returns
    -------
    hull_loop: Loop
        The convex hull Loop of the geometries
    """
    plan_dims = loops[0].plan_dims
    coords = loops[0].d2.T

    for loop in loops[1:]:
        if loop.plan_dims != plan_dims:
            raise GeometryError(
                "Geometries must be on the same plane for 2-D convex hull operation."
            )
        coords = np.concatenate([coords, loop.d2.T])

    hull = ConvexHull(coords)
    x = coords[hull.vertices, 0]
    y = coords[hull.vertices, 1]
    coord_dict = {k: v for k, v in zip(plan_dims, [x, y])}
    hull_loop = Loop.from_dict(coord_dict)
    hull_loop.close()
    return hull_loop


def split_loop(loop):
    """
    Detects and splits self-intersections, returning a list of loops,
    ordered from largest to smallest. Splits self-intersecting loops into
    individual loops. Removes some redundant points.

    Parameters
    ----------
    loop: Loop
        The Loop object to be simplified

    Returns
    -------
    loops: List(Loop)
        The size-sorted list of Loop objects
    """
    polys = SimplifyPolygon(loop_to_pyclippath(loop))
    loops = [pyclippath_to_loop(p, dims=loop.plan_dims) for p in polys]
    for loop in loops:
        loop.close()

    return sorted(loops, key=lambda x: -x.area)


def simplify_loop(loop):
    """
    Simplifies a loop, removing hanging chads and some redundant points.

    Parameters
    ----------
    loop: Loop
        The Loop object to be simplified

    Returns
    -------
    loops: Loop
        The size-sorted list of Loop objects
    """
    return split_loop(loop)[0]


def clean_loop(loop, verbose=False):
    """
    Cleans a loop, removing redundant points.

    Parameters
    ----------
    loop: Loop
        The Loop object to be simplified
    verbose: bool
        Whether or not to print warning if clean_loop did nothing (default = False)

    Returns
    -------
    new_loop: Loop
        The cleaned Loop object
    """
    poly = loop_to_pyclippath(loop)
    scale = poly[0][0] / loop.d2[0][0]
    min_length = loop.get_min_length()
    poly = CleanPolygon(poly, min_length * scale / 100)
    # Not sure why but the min size is often too aggressive.
    # This doesn't quite match up with the documentation...
    new_loop = pyclippath_to_loop(poly)
    if loop.closed:
        new_loop.close()
    if verbose and (len(loop) <= len(new_loop)):
        bluemira_warn("Geometry::clean_loop: Loop simplification was not effective.")
    return new_loop


def entagram(r, p=8, q=3, c=[0, 0]):
    """
    Make an entagram star shape.
    """
    # Isso aqui foi uma merda total mas voce precisava avancar ne?
    # Lakshmi {8|3}
    theta = np.linspace(np.pi / 2, 5 * np.pi / 2, p, endpoint=False)
    r_value = np.sin((p - 2 * q) / (2 * p) * np.pi) / np.sin((2 * q / p) * np.pi)
    inter_angle = (p - 2) * np.pi / p
    outer_angle = np.pi - inter_angle
    phi = np.pi - outer_angle - np.pi / 2
    r_value *= r * np.cos(phi) / np.sqrt(2)
    theta2 = np.linspace(
        np.pi / 2 - np.pi / p, 5 * np.pi / 2 - np.pi / p, p, endpoint=False
    )
    x = r * np.cos(theta)
    x2 = r_value * np.cos(theta2)
    z = r * np.sin(theta)
    z2 = r_value * np.sin(theta2)
    xx, zz = [], []
    for i in range(p):
        xx.extend([x2[i], x[i]])
        zz.extend([z2[i], z[i]])
    xx.append(x2[0])
    zz.append(z2[0])
    return xx, zz
