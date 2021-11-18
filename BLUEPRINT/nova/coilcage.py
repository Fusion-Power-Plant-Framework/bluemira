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
Coil cage object used for calculating 3-D toroidal field ripple
"""
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
import matplotlib
from bluemira.magnetostatics.biot_savart import BiotSavartFilament
from bluemira.base.constants import MU_0
from BLUEPRINT.base.error import NovaError
from bluemira.geometry._deprecated_tools import innocent_smoothie
from BLUEPRINT.utilities.plottools import Plot3D
from BLUEPRINT.geometry.geombase import Plane
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.geomtools import (
    clock,
    rotate_matrix,
    lengthnorm,
    bounding_box,
    loop_plane_intersect,
)


class HelmholtzCage:
    """
    Calculate 3-D ripple field in a toroidal field coil quasi-Helmholtz cage.
    """

    def __init__(
        self,
        n_TF,
        R_0,
        z_0,
        B_0,
        separatrix,
        winding_pack,
        rc=None,
        ny=3,
        nr=1,
        npts=100,
    ):
        self.n_TF = n_TF
        self.R_0 = R_0
        self.z_0 = z_0
        self.B_0 = B_0
        self.sep = separatrix
        self.ny = ny  # winding pack depth discretization
        self.nx = nr  # winding pack radial discretization
        self.npts = npts  # plasma and vv discretization

        # Constructors
        self.bm = None

        self._peak_ripple_points = []

        self.plasma_loop = None
        self.plasma_interp = None
        self.coil_loop = None
        self.bsl = None
        self.lf_point = None
        self.npoints = None
        self.coil_angles = None
        self.current = None  # single coil amp-turns
        self.ripple = None  # Used in ToroidalFieldCoils
        self.res = None

        self.process_separatrix()

        self.dy = winding_pack["depth"] / 2
        self.dx = winding_pack["width"] / 2

        if rc is None:
            # Default filament radius discretised per radial width
            rc = (self.dx + self.dy) / 2 / nr
        self.rc = rc

    def process_separatrix(self):
        """
        Process the plasma separatrix and prepare interpolation objects.
        """
        self.plasma_loop = np.zeros((self.npts, 3))  # initalise loop array
        lf_index = np.argmax(self.sep["x"])
        self.lf_point = np.array([self.sep["x"][lf_index], self.sep["z"][lf_index]])
        # Ensure counter-clockwise
        x, z = clock(self.sep["x"], self.sep["z"])
        (self.plasma_loop[:, 0], self.plasma_loop[:, 2]) = innocent_smoothie(
            x, z, n=self.npts
        )

        length_n = lengthnorm(x, z)
        xfun = interp1d(length_n, x)
        zfun = interp1d(length_n, z)
        self.plasma_interp = {"x": xfun, "z": zfun}

        # Store minima, mid, and maxima (ripple hotspots)
        argmax = np.argmax(self.plasma_loop[:, 2])
        argmin = np.argmin(self.plasma_loop[:, 2])
        argmid = np.argmax(self.plasma_loop[:, 0])

        top = self.plasma_loop[argmax]
        mid = self.plasma_loop[argmid]
        bot = self.plasma_loop[argmin]

        self._peak_ripple_points = [top, mid, bot]

    def amp_turns(self):
        """
        Calculate the single turn current of the TF coil cage.
        """
        # Set the coil current to 1 to get the Green field at R_0.
        self.current = 1
        field = self.get_field([self.R_0, 0, self.z_0])
        self.current = -self.B_0 / field[1]  # single coil amp-turns
        self.bsl.current = self.current

    def pattern(self):
        """
        Generate the TF coil winding pack patterning for the full cage.

        Returns
        -------
        wp_loops: List[Loop]
            The list of the discretised and patterned loops of the current
            centrelines
        """
        if self.nx > 1:
            dx_wp = np.linspace(
                self.dx * (1 / self.nx - 1), self.dx * (1 - 1 / self.nx), self.nx
            )
        else:
            dx_wp = [0]

        if self.ny > 1:
            dy_wp = np.linspace(
                self.dy * (1 / self.ny - 1), self.dy * (1 - 1 / self.ny), self.ny
            )
        else:
            dy_wp = [0]  # coil centreline

        # Assemble all the filament loops for a single TF winding pack (nx * ny)
        wp_loops = []
        for dx in dx_wp:
            loop = self.coil_loop.offset(dx)
            for dy in dy_wp:
                filament_loop = loop.translate([0, dy, 0], update=False)
                wp_loops.append(filament_loop)

        # Pattern the single TF winding pack lacks for all TF coils (n_TF)
        all_loops = []
        # By convention, y = 0 is the plane between two TF coils
        angles = 180 / self.n_TF + np.linspace(0, 360, int(self.n_TF), endpoint=False)
        for theta in angles:
            for loop in wp_loops:
                # We need to ignore CCW here, otherwise the vectors can change direction
                new_loop = loop.rotate(
                    theta, p1=[0, 0, 0], p2=[0, 0, 1], update=False, enforce_ccw=False
                )
                all_loops.append(new_loop)

        self.bsl = BiotSavartFilament(all_loops, self.rc)
        return all_loops

    def get_field(self, point):
        """
        Get the magnetic field at a point.

        Parameters
        ----------
        point: Iterable(3)
            The x, y, z point at which to calcuate the variable

        Returns
        -------
        field: np.array(3)
            The vector of the magnetic field at the point [T]
        """
        return (self.current / (self.nx * self.ny)) * self.bsl.field(*point)

    def get_ripple(self, point):
        """
        Get the toroidal field ripple at a point.

        Parameters
        ----------
        point: Iterable(3)
            The x, y, z point at which to calcuate the variable

        Returns
        -------
        ripple: float
            The value of the TF ripple at the point [%]
        """
        ripple_field = np.zeros(2)
        n = np.array([0, 1, 0])
        planes = [np.pi / self.n_TF, 0]  # rotate (inline, ingap)

        for i, theta in enumerate(planes):
            sr = np.dot(point, rotate_matrix(theta))
            nr = np.dot(n, rotate_matrix(theta))
            field = self.get_field(sr)
            ripple_field[i] = np.dot(nr, field)

        ripple = 1e2 * (ripple_field[0] - ripple_field[1]) / np.sum(ripple_field)
        return ripple

    def set_coil(self, coil_centreline):
        """
        Set the TF coil current centreline in the HelmholtzCage.

        Parameters
        ----------
        coil_centreline: Union[Loop, dict]
            The coil current centreline coordinates
        """
        self.coil_loop = Loop(x=coil_centreline["x"], z=coil_centreline["z"])
        self.npoints = len(self.coil_loop)
        self.pattern()
        self.amp_turns()

    def loop_ripple(self):
        """
        Update the ripple along the separatrix loop.
        """
        self.ripple = np.zeros(self.npts)
        for i, plasma_point in enumerate(self.plasma_loop):
            self.ripple[i] = self.get_ripple(plasma_point)

    def edge_ripple(self, npoints=10, min_l=0.2, max_l=0.8):
        """
        Function handle that is used in the optimiser to check the ripple at the
        separatrix edge. It has been optimised for speed below based on
        physical knowledge of the problem (i.e. where the peak ripple is likely
        to occur)

        Returns
        -------
        ripple: np.array
            The array of ripple values on the plasma edge

        Notes
        -----
        Relies on clock to set 0 index to 9 o'clock
        """
        ripple = np.zeros(npoints)

        l_points = np.linspace(min_l, max_l, npoints - 3)
        for i, l in enumerate(l_points):
            x, z = self.plasma_interp["x"](l), self.plasma_interp["z"](l)
            ripple[i] = self.get_ripple((x, 0, z))

        # Catch the top and bottom points
        for j, point in enumerate(self._peak_ripple_points):
            ripple[i + j + 1] = self.get_ripple(point)

        return ripple

    def get_max_ripple(self):
        """
        Get the maximum ripple along the interpolated plasma separatrix loop.

        Returns
        -------
        max_ripple: float
            The value of the maximum ripple
        """

        def ripple_opp(x):
            """
            Ripple optimiser objective function handle.

            Parameters
            ----------
            x: float
                The normalised position variable value

            Returns
            -------
            ripple: float
                The negative ripple value for the given normalised position
            """
            s = np.zeros(3)
            s[0], s[2] = self.plasma_interp["x"](x), self.plasma_interp["z"](x)
            ripple = self.get_ripple(s)
            return -ripple

        self.res = minimize_scalar(
            ripple_opp, method="bounded", bounds=[0, 1], options={"xatol": 0.1}
        )
        self.res["fun"] *= -1  # negate minimum (max ripple)
        return self.res["fun"]

    def energy(self):
        """
        Calculate the stored magnetic energy inside the coil cage using the
        inductance.

        Returns
        -------
        stored_energy: float
            The stored magnetic energy in the TF coil cage [J]

        Notes
        -----
        \t:math:`E_{stored}=\\dfrac{LI^{2}}{2}`

        When using discretisation, note that the inductance from the BiotSavartLoop
        is with respect to a single reference current filament. So what we're
        actually doing here is:

        \t:math:`E_{stored}=\\sum^{N}_{m,n} L_{m,n}I_{m}I_{n}`
        """
        total_inductance = self.bsl.inductance()  # total inductance
        # For one TF coil
        stored_energy = (
            0.5 * total_inductance * (self.current / (self.nx * self.ny)) ** 2
        )
        return self.n_TF * self.nx * self.ny * stored_energy

    def energy_via_potential(self):
        """
        Calculate the stored magnetic energy inside the coil cage using vector
        potential (super-duper fast version).

        Returns
        -------
        stored_energy: float
            The stored magnetic energy in the TF coil cage [J]
        """
        # TODO: Determine most accurate stored energy calculation approach
        vec_potential = np.zeros((len(self.bsl.ref_loop) - 1, 3))
        for i, x in enumerate(self.bsl.ref_loop.xyz.T[:-1]):
            vec_potential[i, :] += MU_0 * self.bsl.potential(x)
        vec_potential[:, 1] = 0  # zero out-of-plane values
        vec_potential /= self.nx * self.ny  # Normalise by number of filaments

        l_vector = 0
        for i in range(len(self.bsl.ref_loop) - 1):
            l_vector += np.dot(vec_potential[i], self.bsl.ref_d_l.T[i])
        stored_energy = 0.5 * self.current ** 2 * self.n_TF * l_vector
        return stored_energy

    def tf_forces(self, point, current, f_bx, f_bz, method="function"):
        """
        Calculates the magnetic forces on a current carrying element

        Parameters
        ----------
        point: np.array(3)
            The point at which to calculate the forces

        current: np.array(3)
            The current vector at the point
        f_bx: callable
            A method to calculate the radial poloidal field at a given point
        f_bz: callable
            A method to calculate the vertical poloidal field at a given point
        method: str ['function', 'Biot-Savart']
            The method to calculate the toroidal field at a given point:
                'function': uses 1/x to estimate B_t
                'Biot-Savart': uses Biot-Savart to calculate B_t

        Returns
        -------
        F: np.array(3)
            The J x B forces at the point
        """
        # Get toroidal field
        if method == "function":
            # calculate tf field as fitted 1/x function
            # (fast version / only good for TF cl)
            if self.bm is None:
                i = np.argmax(self.coil_loop["z"])
                xo, zo = self.coil_loop["x"][i], self.coil_loop["z"][i]
                # TF moment
                self.bm = xo * self.get_field((xo, 0, zo))[1]
            field = np.zeros(3)
            if point[0] == 0:
                raise NovaError("Cannot calculate toroidal field on machine axis.")
            field[1] = self.bm / point[0]

        elif method == "Biot-Savart":
            # Calculate TF field using Biot-Savart law (slow + correct)
            field = self.get_field(point)

        else:
            raise NovaError(f"Unrecognised method: {method}")

        # Get poloidal field
        theta = np.arctan2(point[1], point[0])
        # rotate to PF plane
        pf_point = np.dot(rotate_matrix(-theta, "z"), point)
        b_x = f_bx(pf_point[0], pf_point[2])
        b_z = f_bz(pf_point[0], pf_point[2])
        field += np.dot(rotate_matrix(theta, "z"), [b_x, 0, b_z])  # add PF
        body_force = np.cross(current, field)
        return body_force

    # Used in ToroidalFieldCoilPlotter

    def plot_loops(self, ax=None, **kwargs):
        """
        Plot the ripple along the separatrix loop.

        Parameters
        ----------
        ax: Axes, optional
            The optional Axes to plot onto, by default None.
            If None then the current Axes will be used.

        Returns
        -------
        sm: ScalarMappable
            The scalar mappable to set the colorbar in the ripple plot
        """
        if ax is None:
            ax = kwargs.get("ax", plt.gca())

        self.loop_ripple()
        rpl, zpl = self.plasma_loop[:, 0], self.plasma_loop[:, 2]
        dx, dz = self.ripple * np.gradient(rpl), self.ripple * np.gradient(zpl)
        norm = matplotlib.colors.Normalize()
        norm.autoscale(self.ripple)
        cm = matplotlib.cm.viridis
        sm = matplotlib.cm.ScalarMappable(cmap=cm, norm=norm)
        sm.set_array([])
        ax.quiver(
            rpl,
            zpl,
            dz,
            -dx,
            color=cm(norm(self.ripple)),
            headaxislength=0,
            headlength=0,
            width=0.02,
        )
        return sm

    def plot_contours_xz(self, ax=None, variable="ripple", theta=0, n=3e3, **kwargs):
        """
        Plot the toroidal field or ripple contours in the x-z plane.

        Parameters
        ----------
        ax: Axes, optional
            The optional Axes to plot onto, by default None.
            If None then the current Axes will be used.
        variable: str
            'ripple' or 'field' to plot the TF ripple or TF field
        theta: float
            Angle of the rotated x-z plane in radians
        n: int
            The reference resolution at which to plot the contours of the variable

        Other Parameters
        ----------------
        xmin: float
            The minimum x value for the grid
        xmax: float
            The maximum x value for the grid
        zmin: float
            The minimum z value for the grid
        zmax: float
            The maximum z value for the grid
        """
        if variable == "ripple":
            func = self.get_ripple
        elif variable == "field":
            func = self.get_field
        else:
            raise ValueError(f"Unknown variable '{variable}'.")

        if ax is None:
            ax = kwargs.pop("ax", plt.gca())

        # Default grid to enclose TF coil with an offset of 2.0 m
        loop = self.coil_loop.copy()
        offset_loop = loop.offset(2.0)
        xmin, xmax = max(0.0, min(offset_loop.x)), max(offset_loop.x)
        zmin, zmax = min(offset_loop.z), max(offset_loop.z)

        xmin = kwargs.pop("xmin", xmin)
        xmax = kwargs.pop("xmax", xmax)
        zmin = kwargs.pop("zmin", zmin)
        zmax = kwargs.pop("zmax", zmax)

        x, z, x_grid, z_grid = self._grid(xmin, xmax, zmin, zmax, n)

        clip_loop = self.coil_loop.offset(-self.dx)
        # Build the matrix of variable values inside the TF offset loop
        values = np.zeros((len(x), len(z)))
        rotation = rotate_matrix(theta, axis="z")
        for i, r_ in enumerate(x):
            for j, z_ in enumerate(z):
                ri, yi, zi = np.dot([r_, 0, z_], rotation)

                if variable == "ripple":
                    if clip_loop.point_inside([r_, z_]):
                        values[i, j] = func((ri, yi, zi))
                    else:
                        values[i, j] = np.NaN
                else:
                    values[i, j] = np.sqrt(np.sum(func([ri, yi, zi]) ** 2))

        if variable == "ripple":
            cm = self._plot_ripple_levels(ax, x_grid, z_grid, values)
        else:
            levels = np.linspace(np.amin(values), np.amax(values), 12)
            cm = ax.contourf(x_grid, z_grid, values, levels=levels)
        ax.set_aspect("equal")
        return cm

    def plot_contours_xy(self, ax=None, z=0, n=3e3, **kwargs):
        """
        Plot the toroidal field contours in the x-y plane.

        Parameters
        ----------
        ax: Axes, optional
            The optional Axes to plot onto, by default None.
            If None then the current Axes will be used.
        z: float
            Height of the x-z plane in meters
        n: int
            The reference resolution at which to plot the contours of the variable

        Other Parameters
        ----------------
        xmin: float
            The minimum x value for the grid
        xmax: float
            The maximum x value for the grid
        ymin: float
            The minimum y value for the grid
        ymax: float
            The maximum y value for the grid
        """
        if ax is None:
            ax = kwargs.pop("ax", plt.gca())

        # Default x-y grid centred around machine axis
        half_radius = 0.75 * self.R_0
        xmin = kwargs.pop("xmin", -half_radius)
        xmax = kwargs.pop("xmax", half_radius)
        ymin = kwargs.pop("ymin", -half_radius)
        ymax = kwargs.pop("ymax", half_radius)

        x, y, x_grid, y_grid = self._grid(xmin, xmax, ymin, ymax, n)

        # Build the matrix of variable values on the grid
        values = np.zeros((len(x), len(y)))
        for i, xi in enumerate(x):
            for j, yi in enumerate(y):
                values[i, j] = np.sqrt(np.sum(self.get_field([xi, yi, z]) ** 2))

        levels = np.linspace(np.amin(values), np.amax(values), 12)
        cm = ax.contourf(x_grid, y_grid, values, levels=levels)
        ax.set_aspect("equal")
        return cm

    def plot(self, ax=None):
        """
        Plot the HelmholtzCage in 3-D.

        Parameters
        ----------
        ax: Axes, optional
            The optional Axes to plot onto, by default None. Note that only 3-D
            Axes will be used.
        """
        if ax is None or not ax.name == "3d":
            ax = Plot3D()

        loops = self.pattern()
        x_bb, y_bb, z_bb = bounding_box(*self.bsl.mid_points)
        for loop in loops:
            loop.plot(ax, fill=False)

        for x, y, z in zip(x_bb, y_bb, z_bb):
            ax.plot([x], [y], [z], color="w")

        n = self.npoints // 10

        # Quiver plot every n-th TF current vector
        points = self.bsl.mid_points.T[::n].T
        vectors = self.bsl.d_l.T[::n].T
        ax.quiver3D(
            *points,
            *vectors,
            color="r",
            length=2.0,
            arrow_length_ratio=0.8,
            pivot="middle",
        )

        # Quiver plot the toroidal field
        point = [self.R_0, 0, 0]
        points = []
        half_angle = np.pi / self.n_TF
        for theta in np.linspace(
            half_angle, 2 * np.pi + half_angle, self.n_TF, endpoint=False
        ):
            r = rotate_matrix(theta, axis="z")
            points.append(np.dot(point, r))

        fields = []
        for point in points:
            fields.append(self.get_field(point))

        ax.quiver3D(
            *np.array(points).T, *np.array(fields).T, color="b", arrow_length_ratio=0.5
        )

    def plot_xy(self, ax=None):
        """
        Plot the HelmholtzCage in the x-y plane.

        Parameters
        ----------
        ax: Axes, optional
            The optional Axes to plot onto, by default None. Note that only 3-D
            Axes will be used.
        """
        if ax is None:
            _, ax = plt.subplots()

        plane = Plane([0, 0, 0], [1, 0, 0], [0, 1, 0])
        loops = self.pattern()
        for loop in loops:
            inters = loop_plane_intersect(loop, plane)
            ax.plot(inters.T[0], inters.T[1], "s", marker="X", color="r")
        ax.set_aspect("equal")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    @staticmethod
    def _grid(xmin, xmax, ymin, ymax, n):
        """
        Set up a grid in 2-D for a given number of points.

        Parameters
        ----------
        xmin: float
            The minimum x value for the grid
        xmax: float
            The maximum x value for the grid
        ymin: float
            The minimum y value for the grid
        ymax: float
            The maximum y value for the grid
        n: int
            The number of points to place on the grid (approximately)

        Returns
        -------
        x: np.array
            The 1-D x values
        y: np.array
            The 1-D y values
        x_grid: np.array
            The 2-D x values
        y_grid: np.array
            The 2-D y values
        """
        ar = (xmax - xmin) / (ymax - ymin)
        ny = int(np.sqrt(n / ar))
        nx = int(n / ny)

        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)

        x_grid, y_grid = np.meshgrid(x, y, indexing="ij")
        return x, y, x_grid, y_grid

    def _plot_ripple_levels(self, ax, x_grid, y_grid, rpl):
        levels = 10 ** (np.linspace(np.log10(0.01), np.log10(3.5), 5))
        levels = np.append(levels, 10 ** np.log(10))
        levels = np.round(levels, decimals=2)
        self.get_max_ripple()  # get max ripple on plasma contour
        rpl_max = self.res["fun"]
        iplasma = np.argmin(abs(np.log10(levels) - np.log10(rpl_max)))
        levels[iplasma] = rpl_max  # select edge contour
        cs = ax.contour(
            x_grid,
            y_grid,
            rpl,
            levels=levels,
            linewidths=2,
            norm=matplotlib.colors.PowerNorm(gamma=rpl_max),
            cmap="viridis",
        )
        zc = cs.collections[iplasma]
        plt.setp(zc, color="r")
        plt.clabel(cs, inline=1, fontsize="medium", colors="k", fmt="%1.2f")
        return cs


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
