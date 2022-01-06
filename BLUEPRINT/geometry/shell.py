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
A shell object (essentially just a polygon with a single hole)
"""
# Plotting imports
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import PathPatch
from sectionproperties.analysis.cross_section import CrossSection
from sectionproperties.pre.sections import CustomSection
from shapely.geos import TopologicalError

from bluemira.geometry._deprecated_tools import get_intersect
from bluemira.geometry.error import GeometryError
from BLUEPRINT.geometry.boolean import boolean_2d_difference
from BLUEPRINT.geometry.geombase import GeomBase, _check_other, point_dict_to_array
from BLUEPRINT.geometry.geomtools import (
    bounding_box,
    clean_loop_points,
    distance_between_points,
    get_control_point,
    qrotate,
)
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.utilities.plottools import BPPathPatch3D, Plot3D, pathify


class Shell(GeomBase):
    """
    BLUEPRINT Shell object, defining a Loop with an internal (single) Loop
    hole

    Parameters
    ----------
    inner: BLUEPRINT::geometry::Loop object
        The inner hole coordinates of the shell
    outer: BLUEPRINT::geometry::Loop object
        The outer edge of the shell
    """

    def __init__(self, inner, outer):
        inner_tmp, outer_tmp = [self._type_checks(p) for p in [inner, outer]]

        # Check for intersections
        x_int, z_int = get_intersect(inner_tmp, outer_tmp)
        if len(x_int) > 0 or len(z_int) > 0:
            raise GeometryError("Cannot create a shell from two insersecting loops")

        self.inner = inner_tmp
        self.outer = outer_tmp

    @classmethod
    def from_dict(cls, xyz_dict):
        """
        Make a Shell from a dictionary.
        """
        inner, outer = Loop(**xyz_dict["inner"]), Loop(**xyz_dict["outer"])
        return cls(inner, outer)

    @classmethod
    def from_offset(cls, loop, offset):
        """
        Make a Shell from a Loop and an offset.
        """
        p_loop = cls._type_checks(cls, loop)
        cls.thickness = abs(offset)
        if offset < 0:
            outer = p_loop
            inner = p_loop.offset(offset)
        elif offset > 0:
            inner = p_loop
            outer = p_loop.offset(offset)
        return cls(inner, outer)

    def as_dict(self):
        """
        Cast the Shell as a dictionary.
        """
        return {"inner": self.inner.as_dict(), "outer": self.outer.as_dict()}

    def as_shpoly(self):
        """
        Cast the Shell as shapely polygon object.
        """
        try:
            p = self.outer.as_shpoly().difference(self.inner.as_shpoly())
        except TopologicalError:
            # This hopefully catches self-intersection errors from shapely.
            p1 = self.outer.as_shpoly().buffer(0)
            p2 = self.inner.as_shpoly().buffer(0)
            p = p1.difference(p2)
        return p

    def rotate(self, theta, update=True, **kwargs):  # noqa: (D102)
        __doc__ = qrotate.__doc__  # noqa

        ri = qrotate(self.inner.as_dict(), theta=np.radians(theta), **kwargs)
        ro = qrotate(self.outer.as_dict(), theta=np.radians(theta), **kwargs)
        if update:
            for k in ["x", "y", "z"]:
                self.inner.__setattr__(k, ri[k])
                self.outer.__setattr__(k, ro[k])
        else:
            return Shell(Loop(**ri), Loop(**ro))

    def rotate_dcm(self, dcm, update=True):
        """
        Rotates the shell based on a direction cosine matrix

        Parameters
        ----------
        dcm: np.array((3, 3))
            The direction cosine matrix array
        update: bool (default = True)
            if True: will update the Loop object
            if False: will return a new Loop object, and leave this one alone
        """
        xyz_inner = dcm @ self.inner.xyz
        xyz_outer = dcm @ self.outer.xyz
        if update:
            for i, k in enumerate(["x", "y", "z"]):
                self.inner.__setattr__(k, xyz_inner[i])
                self.outer.__setattr__(k, xyz_outer[i])
        else:
            return Shell(Loop(*xyz_inner), Loop(*xyz_outer))

    def translate(self, vector, update=True):
        """
        Translate the Shell.

        Parameters
        ----------
        vector: Iterable(3)
            The translation vector
        update: bool
            Whether or not to update the present instance.

        Returns
        -------
        shell: Union[None, Shell]
            Returns a translated Shell if update is False
        """
        if not isinstance(vector, np.ndarray):
            vector = np.array(vector)
        inner = self.inner.translate(vector, update=False)
        outer = self.outer.translate(vector, update=False)
        if update:
            self.inner = inner
            self.outer = outer
        else:
            return Shell(inner, outer)

    def connect_open_loops(self):
        """
        Connect open Loops in the Shell, if any.
        """
        if self.inner.closed or self.outer.closed:
            raise GeometryError("Cannot connect closed loops in Shell.")
        c = np.concatenate((self.outer.xyz, self.inner[::-1]), axis=1)
        loop = Loop(*c)
        loop.close()
        return loop

    def _get_other_3rd_dim(self, other, segments):
        plan_dim_segs, segs_3d = [], []
        for seg in segments:
            p3d, p2d = [], []
            for p in seg:
                p3, p2 = self._get_23d_points(p, other)
                p3d.append(p3)
                p2d.append(p2)
            plan_dim_segs.append(p2d)
            segs_3d.append(p3d)
        return np.array(plan_dim_segs), np.array(segs_3d)

    def _clock_sort(self, segments, concat=True):
        return self.outer._clock_sort(segments, concat)

    def section(self, plane, plot=False):
        """
        Take a section of the Shell with a Plane.

        Parameters
        ----------
        plane: Plane
            The plane along which to get intersections
        plot: bool
            Whether or not to plot the result
        """
        i = self.inner.section(plane, join=False)
        o = self.outer.section(plane, join=False)
        i.extend(o)
        i = np.array(i)
        i = i[np.lexsort((i[:, 1], i[:, 0]))]
        segments = []
        for p in range(len(i) - 1):
            mp = [(i[p][0] + i[p + 1][0]) / 2, (i[p][1] + i[p + 1][1]) / 2]
            if self.point_inside(mp):
                segments.append(np.array([i[p], i[p + 1]]))
        if plot:
            self.plot()
            for p in i:
                plt.plot(p[0], p[1], "r", marker="o")
            for s in segments:
                plt.plot(*s.T, "r")
        return i[::-1], segments[::-1]

    def get_thickness(self):
        """
        Returns the (approximate) thickness
        """
        # NOTE: Probably only going to work for Offset Shells!!
        d = distance_between_points(self.inner, self.outer)
        i, v = np.histogram(d)
        return v[0]

    def point_inside(self, point):
        """
        Determine whether a point lies inside the Shell.
        """
        # Doesn't check boundary! But this is what you want
        if self.outer.point_inside(point):
            if not self.inner.point_inside(point):
                return True
        else:
            return False

    def _type_checks(self, p_loop):
        p_loop = _check_other(p_loop, "Loop")
        if not p_loop.ccw:
            p_loop.reverse()
        return p_loop

    def _get_23d_points(self, point, other):
        othernodim = list(set(self.plan_dims) - set(other.plan_dims))[0]
        point_dict = {}
        for i, c in zip(point, other.outer.plan_dims):
            point_dict[c] = i
        # Only works for flat Shells
        point_dict[othernodim] = other.outer[othernodim][0]
        p3 = point_dict_to_array(point_dict)
        p2 = self.outer._point_other_32d(other.outer, p3)
        return p3, p2

    def plot(self, ax=None, points=False, **kwargs):
        """
        Plot the Shell.

        Parameters
        ----------
        ax: Axes
            The matplotlib Axes on which to plot onto
        points: bool
            Whether or not to show and annotate the points in the Loops
        Other Parameters
        ----------------
        facecolor: str
            The color with which to plot the fill of the Shell
        edgecolor: str
            The color with which to plot the edge lines
        """
        c = kwargs.get("facecolor", "darkblue")
        if self.inner.ndim == 3:
            self._plot_3d(ax=ax, points=points, **kwargs)
            return

        if ax is None:
            ax = kwargs.get("ax", plt.gca())

        a, b = self.plan_dims

        if not hasattr(ax, "zaxis"):
            # 2-D plot
            poly = pathify(self.as_shpoly())
            p = PathPatch(poly, color=c, alpha=kwargs.get("alpha", 1))

            ax.add_patch(p)
            ax.set_aspect("equal")

        self.inner.plot(ax=ax, points=points, fill=False, **kwargs)
        self.outer.plot(ax=ax, points=points, fill=False, **kwargs)
        plt.xlabel(a + " [m]")
        plt.ylabel(b + " [m]")

        if points:
            for i, p in enumerate(self.inner.d2.T):
                ax.annotate(i, xy=(p[0], p[1]))
            for i, p in enumerate(self.outer.d2.T):
                ax.annotate(i, xy=(p[0], p[1]))

    def _plot_3d(self, ax=None, points=False, **kwargs):
        c = kwargs.get("facecolor", "darkblue")

        if ax is None:
            ax = Plot3D()
            # Maintenant on re-arrange un peu pour que matplotlib puisse nous
            # montrer qqchose d'un peu plus correct
            x_bb, y_bb, z_bb = bounding_box(*self.outer.xyz)
            for x, y, z in zip(x_bb, y_bb, z_bb):
                ax.plot([x], [y], [z], color="w")
        # 3-D plot
        # On transforme d'abord l'objet pour qu'il soit situe au plan en
        # 2-D
        dcm = self.rotation_matrix(-self.n_hat, np.array([0.0, 0.0, 1.0]))
        shell = self.rotate_dcm(dcm.T, update=False)

        centroid = np.array(shell.inner._point_23d(shell.inner.centroid))
        shell.translate(-centroid, update=True)

        # Pour en faire un objet que matplotlib puisse comprendre
        poly = pathify(shell.as_shpoly())

        centroid = np.array(self.inner._point_23d(self.inner.centroid))
        # En suite en re-transforme l'objet matploblib en 3-d!
        p = BPPathPatch3D(
            poly, -self.n_hat, centroid, color=c, alpha=kwargs.get("alpha", 1)
        )
        ax.add_patch(p)
        # plot the inner and outer loops too
        self.inner.plot(ax=ax, points=points, fill=False, **kwargs)
        self.outer.plot(ax=ax, points=points, fill=False, **kwargs)

    def get_points(self):
        """
        Get the [x, z] points corresponding to this shell

        Returns
        -------
        points : List[float, float]
            The [x, z] points corresponding to this shell.
        """
        return self.inner.get_points() + self.outer.get_points()

    def get_closed_facets(self, start=0):
        """
        Get the closed facets corresponding to this shell

        The facets are closed by linking the last point back to the first.

        Parameters
        ----------
        start : int, optional
            The index to assign to the first point, by default 0

        Returns
        -------
        facets : List[int, int]
            The closed facets corresponding to this shell.
        """
        inner_facets = self.inner.get_closed_facets(start)
        outer_facets = self.outer.get_closed_facets(start + len(inner_facets))
        return inner_facets + outer_facets

    def create_closed_shell(self):
        """
        Creates a closed shell from this shell.

        Returns
        -------
        closed_shell : Shell
            The closed shell created from this shell.
        """
        closed_shell_inner = Loop(*self.inner.d2)
        closed_shell_outer = Loop(*self.outer.d2)
        closed_shell = Shell(closed_shell_inner, closed_shell_outer)
        closed_shell.inner.close()
        closed_shell.outer.close()
        return closed_shell

    def get_control_point(self):
        """
        Get the control point correponding to this shell.

        Returns
        -------
        control_point : (float, float)
            The control point corresponding to this shell.
        """
        return list(get_control_point(self.create_closed_shell()))

    def get_hole(self):
        """
        Get the hole corresponding to this shell

        This corresponds to the centroid of the inner loop.

        Returns
        -------
        hole : (float, float)
            The hole corresponding to this shell.
        """
        return list(self.create_closed_shell().inner.centroid)

    def generate_cross_section(
        self, mesh_sizes=None, min_length=None, min_angle=None, verbose=True
    ):
        """
        Generate the meshed `CrossSection` for this Shell

        This cleans the inner and outer Loops based on the `min_length` and `min_angle`
        using the :func:`~BLUEPRINT.geometry.geomtools.clean_loop_points` algorithm.

        The clean inner and outer loops are then fed into a sectionproperties
        `CustomSection` object, with corresponding facets and control point. In order
        to maintain the shell geometry, a hole is defined in the centroid of the shell.
        The geometry is cleaned, using the sectionproperties `clean_geometry` method,
        before creating a mesh and loading the mesh and geometry into a sectionproperties
        `CrossSection`.

        Also provides the cleaned `Shell` representing the geometry used to generate the
        `CrossSection`.

        Parameters
        ----------
        mesh_sizes : List[float], optional
            The mesh sizes to use for the sectionproperties meshing algorithm,
            by default None
        min_length : float, optional
            The minimum length [m] by which any two points should be separated,
            by default None.
        min_angle : float, optional
           The minimum angle [°] between any three points, beyond which points are not
           removed by cleaning even if they lie within min_length, by default None.
        verbose : bool, optional
            Determines if verbose mesh cleaning output should be provided,
            by default True.

        Returns
        -------
        cross_section : sectionproperties.analysis.cross_section.CrossSection
            The resulting `CrossSection` from meshing the cleaned shell.
        clean_shell : Shell
            The clean shell geometry used to generate the `CrossSection`.
        """
        if mesh_sizes is None:
            mesh_sizes = [
                min([self.inner.get_min_length(), self.outer.get_min_length()])
            ]

        clean_outer_points = clean_loop_points(
            self.outer, min_length=min_length, min_angle=min_angle
        )
        clean_outer_loop = Loop(*np.array(clean_outer_points).T)

        clean_inner_points = clean_loop_points(
            self.inner, min_length=min_length, min_angle=min_angle
        )
        clean_inner_loop = Loop(*np.array(clean_inner_points).T)

        clean_shell = Shell(inner=clean_inner_loop, outer=clean_outer_loop)

        points = clean_shell.get_points()
        facets = clean_shell.get_closed_facets()
        control_point = clean_shell.get_control_point()
        hole = clean_shell.get_hole()

        geometry = CustomSection(points, facets, [hole], [control_point])
        geometry.clean_geometry(verbose=verbose)
        mesh = geometry.create_mesh(mesh_sizes=mesh_sizes)
        cross_section = CrossSection(geometry, mesh)
        return cross_section, clean_shell

    def split_by_line(self, p1, p2):
        """
        Get a pair of loops obtained splitting the shell in two
        spearated loops using from a straight line. This is done
        using the Loop split_by_line() function

        Parameters
        ----------
        p1: np.array(2)
            The first point in the split line
        p2: np.array(2)
            The second point in the split line

        Returns
        -------
        out_loop_1: Loop
            First loop obtained by the cut.
        out_loop_2: Loop
            Second loop obtained by the cut.
        """
        # Getting the split coordinates
        split_out_1, split_out_2 = self.outer.split_by_line(p1=p1, p2=p2)

        # Getting 2 array of x insead of an array [x,z] points
        split_out_1 = split_out_1.T
        split_out_2 = split_out_2.T

        # Defining the cut loops
        cut_loop_1 = Loop(x=split_out_1[0], z=split_out_1[1])
        cut_loop_2 = Loop(x=split_out_2[0], z=split_out_2[1])

        # Close the loop
        cut_loop_1.close()
        cut_loop_2.close()

        # Make the shell cut to obtain get the output loops
        out_loop_1 = boolean_2d_difference(self, cut_loop_1)[0]
        out_loop_2 = boolean_2d_difference(self, cut_loop_2)[0]

        return out_loop_1, out_loop_2

    def __str__(self):
        """
        Return a string representation of the Shell.
        """
        a, b = self.plan_dims
        return "{}-{} Shell".format(a, b)

    @property
    def n_hat(self):
        """
        The normal vector of the Shell.
        """
        return self.inner.n_hat

    @property
    def plan_dims(self):
        """
        The planar dimensions of the Shell
        """
        return self.inner.plan_dims

    @property
    def centroid(self):
        """
        The centroid of the Shell.
        """
        y_l, z_l = self.outer.d2

        sy, sz = 0, 0

        for (y, y2, z, z2) in zip(y_l[:-1], y_l[1:], z_l[:-1], z_l[1:]):
            sy += (y + y2) * (y * z2 - y2 * z)
            sz += (z + z2) * (y * z2 - y2 * z)

        y_l, z_l = self.inner.d2
        for (y, y2, z, z2) in zip(y_l[:-1], y_l[1:], z_l[:-1], z_l[1:]):
            sy -= (y + y2) * (y * z2 - y2 * z)
            sz -= (z + z2) * (y * z2 - y2 * z)

        sy /= 6 * self.area
        sz /= 6 * self.area
        return sy, sz

    @property
    def area(self):
        """
        The area of the Shell cross-section.
        """
        return self.outer.area - self.inner.area

    @property
    def enclosed_area(self):
        """
        The enclosed area of the Shell (inner Loop).
        """
        return self.inner.area

    @property
    def length(self):
        """
        Perimeter
        """
        return self.inner.length + self.outer.length

    @property
    def plane(self):
        """
        The Shell Plane.
        """
        return self.inner.plane

    def __eq__(self, other):
        """
        Check equality of the Shell with another.
        """
        try:
            return np.all(
                [
                    np.all(np.allclose(self.inner.xyz, other.inner.xyz)),
                    np.all(np.allclose(self.outer.xyz, other.outer.xyz)),
                ]
            )
        except (ValueError, AttributeError):
            return False

    __hash__ = None

    def __instancecheck__(self, instance):
        """
        Check whether an instance is a Shell object.
        """
        # from BLUEPRINT.geometry.shell import Shell

        if isinstance(instance, Shell):
            return True
        else:
            return False


class MultiShell(GeomBase):
    """
    A collection of multiple Shell objects

    Parameters
    ----------
    shell_list: list(Shell, Shell, ..)
        The list of Shells from which to create a MultiShell
    stitch: bool
        Whether or not to stitch the shells together
    """

    def __init__(self, shell_list):
        for shell in shell_list:
            _check_other(shell, "Shell")
        self.shells = shell_list

    def plot(self, ax=None, points=False, **kwargs):
        """
        Plot the MultiShell.
        """
        if ax is None:
            ax = kwargs.get("ax", plt.gca())
        for shell in self.shells:
            shell.plot(ax, points, **kwargs)
