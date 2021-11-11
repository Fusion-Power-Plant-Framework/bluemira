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
Equilibrium manipulation tool - reduced version of SF from nova
"""
import numpy as np
from copy import deepcopy
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from collections import OrderedDict
from matplotlib._contour import QuadContourGenerator
from bluemira.equilibria.file import EQDSKInterface
from bluemira.equilibria.find import find_OX_points
from bluemira.base.look_and_feel import bluemira_warn
from BLUEPRINT.base.error import NovaError
from bluemira.geometry._deprecated_tools import innocent_smoothie
from BLUEPRINT.geometry.geomtools import theta_sort, length, lengthnorm, clock
from BLUEPRINT.geometry.offset import offset_smc


class StreamFlow:
    """
    A reduced and simplfied version of S. McIntosh's SF object from the
    original Nova code
    """

    def __init__(self, filename, **kwargs):
        self.shape = {}
        self.filename = filename

        # Constructors (WIP: untangle spaghetti)
        self.o_point = None
        self.o_psi = None
        self.x_point = None
        self.x_psi = None
        self.xp_location = None
        self.lfp_x = None
        self.lfp_z = None
        self.hfp_x = None
        self.hfp_z = None

        self.cfield = None
        self.cfield_bndry = None
        self.n = None
        self.nx = None
        self.nz = None
        self.dx = None
        self.dz = None
        self.delta = None
        self.x2d = None
        self.z2d = None
        self.x = None
        self.z = None
        self.xbdry = None
        self.zbdry = None
        self.nbdry = None
        self.psi = None

        self.n_sol = None
        self.x_sol = None
        self.z_sol = None
        self.sol_psi = None

        self.ff_prime = None
        self.f_psi = None
        self.d_f_psi = None
        self.p_prime = None
        self.d_p_psi = None
        self.p_spline = None
        self.b_spline = None
        self.Bx = None
        self.Bz = None

        self.t_id = None
        self.tleg = None
        self.nleg = None
        self.legs = None

        self.flag_CREATE = False
        self.current_dir = 1
        self.norm = 1
        self.b_scale = 1

        for key in kwargs:
            setattr(self, key, kwargs[key])

        eqdsk = EQDSKInterface()
        self.eqdsk = eqdsk.read(self.filename)

        self.xcentre = self.eqdsk["xcentre"]
        self.bcentre = self.eqdsk["bcentre"]
        self.cplasma = self.eqdsk["cplasma"]

        self.normalise()  # unit normalisation
        self.set_plasma(self.eqdsk)
        self.set_boundary(self.eqdsk["xbdry"], self.eqdsk["zbdry"])
        self.set_flux_functions(self.eqdsk)  # calculate flux profiles

        self.get_mx_points()  # UNDER DEVLEOPMENT

        self.set_contour()  # set cfeild
        self.get_lowfield_points()
        x, z = self.get_boundary()
        self.set_boundary(x, z)

        self.xlim = self.eqdsk["xlim"]
        self.nlim = self.eqdsk["nlim"]
        try:
            self.ylim = self.eqdsk["ylim"]
        except KeyError:
            self.ylim = self.eqdsk["zlim"]

        self.rcirc = 0.3 * abs(self.o_point[1] - self.x_point[1])
        self.drcirc = 0.15 * self.rcirc  # leg search width
        self.shape = {}

    def get_mx_points(self):
        """
        Retrofitting better O/M, X search algorithms to SF object
        """
        o_points, x_points = find_OX_points(self.x2d, self.z2d, self.psi)
        self.o_point = o_points[0][:2]
        self.o_psi = o_points[0][-1]
        self.x_point = x_points[0][:2]
        self.x_psi = x_points[0][-1]

        if self.x_point[1] > self.o_point[1]:
            self.xp_location = "upper"
        elif self.x_point[1] < self.o_point[1]:
            self.xp_location = "lower"
        else:
            raise NovaError(
                "Hier stimmt was nicht, oder hast du endlich "
                "einen Feuerball Plasmaform parametrisiert?"
            )

    def normalise(self):
        """
        Normalise the poloidal magnetic flux depending on the origin of the
        equilibrium. Standardises to poloidal magnetic flux per radian.
        """
        if (
            "Fiesta" in self.eqdsk["name"]
            or "Nova" in self.eqdsk["name"]
            or "disr" in self.eqdsk["name"]
            or "equilibria" in self.eqdsk["name"]
            and "CREATE" not in self.eqdsk["name"]
        ):
            self.norm = 1
            self.flag_CREATE = False
        else:  # CREATE
            self.flag_CREATE = True
            self.eqdsk["cplasma"] *= -1
            self.norm = 2 * np.pi
            for key in ["psi", "psimag", "psibdry"]:
                self.eqdsk[key] /= self.norm  # Webber/loop to Webber/radian
            for key in ["ffprime", "pprime"]:
                # []/(Webber/loop) to []/(Webber/radian)
                self.eqdsk[key] *= self.norm
        self.b_scale = 1  # flux function scaling

    def set_plasma(self, eq):
        """
        Set the plasma x, z, and psi values.
        """
        for key in ["x", "z", "psi"]:
            if key in eq:
                setattr(self, key, eq[key])
        self.trim_x()
        self.space()
        self.set_field()

    def trim_x(self, rmin=1.5):
        """
        Drop 0 values in x - otherwise we can't calculate the fields.
        """
        if self.x[0] == 0:  # trim zero x-coordinate entries
            i = np.argmin(abs(self.x - rmin))
            self.x = self.x[i:]
            self.psi = self.psi[i:, :]

    def space(self):
        """
        Set the grid for the StreamFlow object.
        """
        self.nx = len(self.x)
        self.nz = len(self.z)
        self.n = self.nx * self.nz
        self.dx = (self.x[-1] - self.x[0]) / (self.nx - 1)
        self.dz = (self.z[-1] - self.z[0]) / (self.nz - 1)
        self.delta = np.sqrt(self.dx ** 2 + self.dz ** 2)
        self.x2d, self.z2d = np.meshgrid(self.x, self.z, indexing="ij")

    def set_field(self):
        """
        Set the radial and vertical field arrays.
        """
        psi_x, psi_z = np.gradient(self.psi, self.dx, self.dz)
        xm = self.x.reshape(-1, 1) * np.ones([1, self.nz])
        xm[xm == 0] = 1e-34
        self.Bx = -psi_z / xm
        self.Bz = psi_x / xm

    def set_boundary(self, x, z, n=5e2):
        """
        Set the plasma boundary in the StreamFlow.
        """
        self.nbdry = int(n)
        self.xbdry, self.zbdry = innocent_smoothie(x, z, n)

    def set_flux_functions(self, eqdsk):
        """
        Set the flux functions to use in the StreamFlow.
        """
        f_pol = eqdsk["fpol"]
        pressure = eqdsk["pressure"]
        n = len(f_pol)
        psi_ff = np.linspace(0, 1, n)
        f_pol = interp1d(psi_ff, f_pol)(psi_ff)
        pressure = interp1d(psi_ff, pressure)(psi_ff)
        d_f_pol = np.gradient(f_pol, 1 / (n - 1))
        d_pressure = np.gradient(pressure, 1 / (n - 1))
        self.f_psi = interp1d(psi_ff, f_pol)
        self.d_f_psi = interp1d(psi_ff, d_f_pol)
        self.d_p_psi = interp1d(psi_ff, d_pressure)
        ff_prime = UnivariateSpline(psi_ff, eqdsk["ffprime"], s=1e-5)(psi_ff)
        p_prime = UnivariateSpline(psi_ff, eqdsk["pprime"], s=1e2)(psi_ff)  # s=1e5
        self.ff_prime = interp1d(psi_ff, ff_prime, fill_value=0, bounds_error=False)
        self.p_prime = interp1d(psi_ff, p_prime, fill_value=0, bounds_error=False)

    def point_psi(self, point):
        """
        Calculate the psi at a point.
        """
        if self.p_spline is None:
            self.p_spline = RectBivariateSpline(self.x, self.z, self.psi)
        psi = self.p_spline.ev(point[0], point[1])
        return psi

    def get_x_psi(self, po=None, select="lower"):
        """
        Get the X-point poloidal flux value.
        """
        # Still used in crosssection TODO: abcise
        if po is None:
            if self.x_point is not None:
                po = self.x_point
            else:
                xo_arg = np.argmin(self.eqdsk["zbdry"])
                po = [self.eqdsk["xbdry"][xo_arg], self.eqdsk["zbdry"][xo_arg]]
        x_point = np.zeros((2, 2))
        x_psi = np.zeros(2)
        for i, flip in enumerate([1, -1]):
            po[1] *= flip
            x_point[:, i] = self._get_x(po=po)
            x_psi[i] = self.point_psi(x_point[:, i])
        index = np.argsort(x_point[1, :])
        x_point = x_point[:, index]
        x_psi = x_psi[index]
        if select == "lower":
            i = 0  # lower Xpoint
        elif select == "upper":
            i = 1  # upper Xpoint
        elif select == "primary":
            i = np.argmax(self.current_dir * x_psi)

        self.x_psi = x_psi[i]
        self.x_point = x_point[:, i]

        if i == 0:
            po[1] *= -1  # re-flip
        if self.x_point[1] < self.o_point[1]:
            self.xp_location = "lower"
        else:
            self.xp_location = "upper"
        return self.x_psi, self.x_point

    def _get_x(self, po=None):
        def field(p):
            b = self.point_field(p)
            return sum(b * b) ** 0.5

        res = minimize(
            field,
            np.array(po),
            method="nelder-mead",
            options={"xatol": 1e-7, "disp": False},
        )
        return res.x

    def point_field(self, point, check_bounds=False):
        """
        Returns the poloidal magnetic field at a point

        Parameters
        ----------
        point: np.array(2)
            The x, z coordinates of the point
        check_bounds: bool
            Whether or not to check interpolation bounds

        Returns
        -------
        field: np.array(2)
            The poloidal magnetic field x and z components
        """
        field = np.zeros(2)  # function re-name (was Bcoil)
        if self.b_spline is None:
            self.b_spline = [[], []]
            self.b_spline[0] = RectBivariateSpline(self.x, self.z, self.Bx)
            self.b_spline[1] = RectBivariateSpline(self.x, self.z, self.Bz)
        if check_bounds:
            inbound = (
                point[0] >= np.min(self.x)
                and point[0] <= np.max(self.x)
                and point[1] >= np.min(self.z)
                and point[1] <= np.max(self.z)
            )
            return inbound
        else:
            for i in range(2):
                field[i] = self.b_spline[i].ev(point[0], point[1])
            return field

    def __getstate__(self):
        """
        Prepare StreamFlow for pickling by dropping C objects.
        """
        d = dict(self.__dict__)
        # Cannot pickle these C objects
        d.pop("cfield", None)
        d.pop("cfield_bndry", None)
        return d

    def __setstate__(self, d):
        """
        Initialise StreamFlow following unpickling.
        """
        self.__dict__ = d
        self.set_contour()

    def set_contour(self):
        """
        Set the flux contour tracing utilies.
        """
        psi_boundary = 1.1 * (self.x_psi - self.o_psi) + self.o_psi
        psi_bndry = np.pad(
            self.psi[1:-1, 1:-1], (1,), mode="constant", constant_values=psi_boundary
        )

        self.cfield = QuadContourGenerator(self.x2d, self.z2d, self.psi, None, None, 0)
        self.cfield_bndry = QuadContourGenerator(
            self.x2d, self.z2d, psi_bndry, None, None, 0
        )

    def get_lowfield_points(self, alpha=0.999):
        """
        Get the low field points.
        """
        x, z = self.get_boundary(alpha=alpha)
        if self.x_point[1] < self.o_point[1]:
            index = z > self.x_point[1]
        else:  # alowance for upper Xpoint
            index = z < self.x_point[1]
        x_loop, z_loop = x[index], z[index]
        xc, zc = self.o_point
        radius = ((x_loop - xc) ** 2 + (z_loop - zc) ** 2) ** 0.5
        theta = np.arctan2(z_loop - zc, x_loop - xc)
        index = theta.argsort()
        radius, theta = radius[index], theta[index]
        theta = np.append(theta[-1] - 2 * np.pi, theta)
        radius = np.append(radius[-1], radius)
        x = xc + radius * np.cos(theta)
        z = zc + radius * np.sin(theta)
        f_lfs_x = interp1d(theta, x)
        f_lfs_z = interp1d(theta, z)
        self.lfp_x, self.lfp_z = f_lfs_x(0), f_lfs_z(0)
        self.lfp_x = self.get_midplane(self.lfp_x, self.lfp_z)
        self.hfp_x, self.hfp_z = f_lfs_x(-np.pi), f_lfs_z(-np.pi)
        self.hfp_x = self.get_midplane(self.hfp_x, self.hfp_z)
        self.shape["R"] = np.mean([self.hfp_x, self.lfp_x])
        self.shape["a"] = (self.lfp_x - self.hfp_x) / 2
        self.shape["AR"] = self.shape["R"] / self.shape["a"]
        return self.lfp_x, self.lfp_z, self.hfp_x, self.hfp_z

    def get_boundary(self, alpha=0.999, plot=False):
        """
        Get the plasma boundary.
        """
        psi_norm = alpha * (self.x_psi - self.o_psi) + self.o_psi

        psi_line = self.get_contour([psi_norm], boundary=True)[0]
        x_bndry, z_bndry = np.array([]), np.array([])
        for line in psi_line:
            x, z = line[:, 0], line[:, 1]
            if self.xp_location == "lower":  # lower Xpoint
                index = z >= self.x_point[1]
            elif self.xp_location == "upper":  # upper Xpoint
                index = z <= self.x_point[1]
            if sum(index) > 0:
                x, z = x[index], z[index]
                loop = (
                    np.sqrt((x[0] - x[-1]) ** 2 + (z[0] - z[-1]) ** 2) < 5 * self.delta
                )
                if (z > self.o_point[1]).any() and (z < self.o_point[1]).any() and loop:
                    x_bndry, z_bndry = np.append(x_bndry, x), np.append(z_bndry, z)
        x_bndry, z_bndry = clock(x_bndry, z_bndry)
        if plot:
            plt.plot(x_bndry, z_bndry)
        return x_bndry, z_bndry

    def get_contour(self, levels, boundary=False):
        """
        Get the contours at certain flux levels.
        """
        if boundary:
            contour_func = self.cfield_bndry.create_contour
        else:
            contour_func = self.cfield.create_contour

        lines = []
        for level in levels:
            psi_line = contour_func(level)
            lines.append(psi_line)
        return lines

    def get_midplane(self, x, z):
        """
        Get the flux at the midplane from a specific point.
        """

        def psi_err(x_opt, *args):
            z_opt = args[0]
            psi = self.point_psi((x_opt, z_opt))
            return abs(psi - self.x_psi)

        res = minimize(
            psi_err,
            np.array(x),
            method="nelder-mead",
            args=(z),
            options={"xatol": 1e-7, "disp": False},
        )
        return res.x[0]

    def plot(
        self,
        n_std=1.5,
        n_levels=31,
        x_norm=True,
        lw=1,
        plot_vac=True,
        boundary=True,
        **kwargs,
    ):
        """
        Plot the StreamFlow object.
        """
        alpha = np.array([1, 1], dtype=float)
        lw = lw * np.array([2.25, 1.75])
        if boundary:
            x, z = self.get_boundary(1 - 1e-3)
            plt.plot(x, z, linewidth=lw[0], color=0.75 * np.ones(3))
            self.set_boundary(x, z)
        if self.x_psi is None:
            self.get_x_psi()
        if self.o_psi is None:
            self.get_mx_points()
        if "levels" not in kwargs.keys():

            level, n = [-n_std * np.std(self.psi), n_std * np.std(self.psi)], n_levels
            if (
                n_std * np.std(self.psi) < self.o_psi - self.x_psi
                and self.z.max() > self.o_point[1]
            ):
                n_std = (self.o_psi - self.x_psi) / np.std(self.psi)
                level, n = (
                    [-n_std * np.std(self.psi), n_std * np.std(self.psi)],
                    n_levels,
                )
            levels = np.linspace(level[0], level[1], n)
            linetype = "-"
        else:
            levels = kwargs["levels"]
            linetype = "-"
        color = "k"
        if "linetype" in kwargs.keys():
            linetype = kwargs["linetype"]
        if color == "k":
            alpha *= 0.25
        if x_norm:
            levels = levels + self.x_psi

        for psi_line in levels:
            for line in psi_line:
                x, z = line[:, 0], line[:, 1]
                if self.in_plasma(x, z) and boundary:
                    pindex = 0
                else:
                    pindex = 1
                if (not plot_vac and pindex == 0) or plot_vac:
                    plt.plot(
                        x,
                        z,
                        linetype,
                        linewidth=lw[pindex],
                        color=color,
                        alpha=alpha[pindex],
                    )
        if boundary:
            plt.plot(
                self.xbdry,
                self.zbdry,
                linetype,
                linewidth=lw[pindex],
                color=color,
                alpha=alpha[pindex],
            )
        plt.axis("equal")
        plt.axis("off")
        return levels

    def in_plasma(self, x, z, delta=0):
        """
        Determine whether a point is inside the plasma.
        """
        return (
            x.min() >= self.xbdry.min() - delta
            and x.max() <= self.xbdry.max() + delta
            and z.min() >= self.zbdry.min() - delta
            and z.max() <= self.zbdry.max() + delta
        )

    def _minimum_field(self, radius, theta):
        x_field = radius * np.sin(theta) + self.x_point[0]
        z_field = radius * np.cos(theta) + self.x_point[1]
        values = np.zeros(len(x_field))
        for i, (x, z) in enumerate(zip(x_field, z_field)):
            field = self.point_field((x, z))
            values[i] = np.sqrt(field[0] ** 2 + field[1] ** 2)
        return np.argmin(values)

    def get_legs(self, debug=False):
        """
        Get the StreamFlow separatrix legs.
        """
        if debug:
            theta = np.linspace(-np.pi, np.pi, 100)
            x = (self.rcirc - self.drcirc / 2) * np.cos(theta)
            z = (self.rcirc - self.drcirc / 2) * np.sin(theta)
            plt.plot(x + self.x_point[0], z + self.x_point[1], "k--", alpha=0.5)
            x = (self.rcirc + self.drcirc / 2) * np.cos(theta)
            z = (self.rcirc + self.drcirc / 2) * np.sin(theta)
            plt.plot(x + self.x_point[0], z + self.x_point[1], "k--", alpha=0.5)
        self.tleg = np.array([])
        for i in range(len(self.x_sol)):
            x, t = self.topolar(self.x_sol[i], self.z_sol[i])
            index = (x > self.rcirc - self.drcirc / 2) & (
                x < self.rcirc + self.drcirc / 2
            )
            self.tleg = np.append(self.tleg, t[index])
        nbin = 50
        nhist, bins = np.histogram(self.tleg, bins=nbin)
        flag, self.nleg, self.tleg = 0, 0, np.array([])
        for i in range(len(nhist)):
            if nhist[i] > 0:
                if flag == 0:
                    tstart = bins[i]
                    tend = bins[i]
                    flag = 1
                if flag == 1:
                    tend = bins[i]
            elif flag == 1:
                self.tleg = np.append(self.tleg, (tstart + tend) / 2)
                self.nleg += 1
                flag = 0
            else:
                flag = 0
        if nhist[-1] > 0:
            tend = bins[-1]
            self.tleg = np.append(self.tleg, (tstart + tend) / 2)
            self.nleg += 1
        struct = {
            "X": [[] for i in range(self.n_sol)],
            "Z": [[] for i in range(self.n_sol)],
            "i": 0,
        }
        if self.nleg == 6:  # snow flake
            self.legs = {
                "inner1": deepcopy(struct),
                "inner2": deepcopy(struct),
                "outer1": deepcopy(struct),
                "outer2": deepcopy(struct),
                "core1": deepcopy(struct),
                "core2": deepcopy(struct),
            }
        else:
            self.legs = {
                "inner": deepcopy(struct),
                "outer": deepcopy(struct),
                "core1": deepcopy(struct),
                "core2": deepcopy(struct),
            }

        self.legs = OrderedDict(sorted(self.legs.items(), key=lambda _x: _x[0]))

        if self.nleg == 0:
            err_txt = "legs not found\n"
            raise ValueError(err_txt)
        self.t_id = np.arange(self.nleg)
        self.t_id = np.append(self.nleg - 1, self.t_id)
        self.t_id = np.append(self.t_id, 0)
        self.tleg = np.append(-np.pi - (np.pi - self.tleg[-1]), self.tleg)
        self.tleg = np.append(self.tleg, np.pi + (np.pi + self.tleg[1]))

        self._store_legs()

    def _store_legs(self):
        for i in range(len(self.x_sol)):
            ends, ro = [0, -1], np.zeros(2)
            for e in ends:
                ro[e] = np.sqrt(self.x_sol[i][e] ** 2 + self.z_sol[i][e] ** 2)
            x, t = self.topolar(self.x_sol[i], self.z_sol[i])
            post = False
            rpost, tpost = 0, 0
            if ro[0] == ro[-1]:  # cut loops
                if np.min(x * np.cos(t)) > self.drcirc - self.rcirc:
                    nmax = np.argmax(x * np.sin(t))  # LF
                else:
                    nmax = np.argmin(x * np.cos(t))  # minimum z
                x = np.append(x[nmax:], x[:nmax])
                t = np.append(t[nmax:], t[:nmax])
            while len(x) > 0:
                if x[0] > self.rcirc:
                    if np.min(x) < self.rcirc:
                        ncut = np.arange(len(x))[x < self.rcirc][0]
                        xloop, tloop = x[:ncut], t[:ncut]
                        loop = False
                    else:
                        ncut = -1
                        xloop, tloop = x, t
                        loop = True
                    if post:
                        xloop = np.append(rpost, xloop)
                        tloop = np.append(tpost, tloop)
                else:
                    ncut = np.arange(len(x))[x > self.rcirc][0]
                    xin, tin = x[:ncut], t[:ncut]
                    nx = self._minimum_field(xin, tin)  # minimum feild
                    rpre, tpre = xin[: nx + 1], tin[: nx + 1]
                    rpost, tpost = xin[nx:], tin[nx:]
                    loop = True
                    post = True
                    xloop, tloop = np.append(xloop, rpre), np.append(tloop, tpre)
                if loop:
                    if xloop[0] < self.rcirc and xloop[-1] < self.rcirc:
                        if np.min(xloop * np.cos(tloop)) > self.drcirc - self.rcirc:
                            nmax = np.argmax(xloop * np.sin(tloop))  # LF
                        else:
                            nmax = np.argmax(xloop)
                        self.store_leg(xloop[:nmax], tloop[:nmax])
                        self.store_leg(xloop[nmax:], tloop[nmax:])
                    else:
                        self.store_leg(xloop, tloop)
                if ncut == -1:
                    x, t = [], []
                else:
                    x, t = x[ncut:], t[ncut:]

    def topolar(self, x, z):
        """
        Convert cartesian to polar coordinates.
        """
        r = np.sqrt((x - self.x_point[0]) ** 2 + (z - self.x_point[1]) ** 2)
        if self.xp_location == "lower":
            t = np.arctan2(x - self.x_point[0], z - self.x_point[1])
        elif self.xp_location == "upper":
            t = np.arctan2(x - self.x_point[0], self.x_point[1] - z)
        else:
            raise ValueError("Xloc not set (get_Xpsi)")
        return r, t

    def upsample_sol(self, nmult=10):
        """
        Upsample the scrape-off layer.
        """
        k = 1  # smoothing factor
        for i, (x, z) in enumerate(zip(self.x_sol, self.z_sol)):
            length_norm = lengthnorm(x, z)
            l_vector = np.linspace(0, 1, nmult * len(length_norm))
            self.x_sol[i] = InterpolatedUnivariateSpline(length_norm, x, k=k)(l_vector)
            self.z_sol[i] = InterpolatedUnivariateSpline(length_norm, z, k=k)(l_vector)

    def sol(self, dx=3e-3, n_sol=5, debug=False):
        """
        Scrape-off layer update function.
        """
        self.get_sol_psi(d_sol=dx, n_sol=n_sol)

        contours = self.get_contour(self.sol_psi)
        self.x_sol, self.z_sol = self.pick_contour(
            contours, xpoint=True, midplane=False, plasma=False
        )
        self.upsample_sol(nmult=10)  # upsample
        self.get_legs(debug=debug)

    def get_sol_psi(self, d_sol, n_sol):
        """
        Get scrape-off layer flux value.
        """
        self.n_sol = n_sol

        self.get_lowfield_points()
        d_sol = np.linspace(0, d_sol, self.n_sol)
        x = self.lfp_x + d_sol
        z = self.lfp_z * np.ones(len(x))
        self.sol_psi = np.zeros(len(x))
        for i, (rp, zp) in enumerate(zip(x, z)):
            self.sol_psi[i] = self.point_psi([rp, zp])

    def midplane_loop(self, x, z):
        """
        Get a loop from the midplane at a point.
        """
        index = np.argmin((x - self.lfp_x) ** 2 + (z - self.lfp_z) ** 2)
        if z[index] <= self.lfp_z:
            index -= 1
        x = np.append(x[: index + 1][::-1], x[index:][::-1])
        z = np.append(z[: index + 1][::-1], z[index:][::-1])
        length_ = lengthnorm(x, z)
        index = np.append(np.diff(length_) != 0, True)
        x, z = x[index], z[index]  # remove duplicates
        return x, z

    def first_wall_psi(self, trim=True, single_contour=False, **kwargs):
        """
        Get firstwall psi loop.
        """
        if "point" in kwargs:
            xeq, zeq = kwargs.get("point")
            psi = self.point_psi([xeq, zeq])
        else:
            xeq, zeq = self.lfp_x, self.lfp_z
            if "psi_n" in kwargs:  # normalized psi
                psi_n = kwargs.get("psi_n")
                psi = psi_n * (self.x_psi - self.o_psi) + self.o_psi
            elif "psi" in kwargs:
                psi = kwargs.get("psi")
            else:
                raise ValueError("set point=(x,z) or psi in kwargs")
        contours = self.get_contour([psi])
        x_contour, z_contour = self.pick_contour(contours, xpoint=False)
        if single_contour:
            min_contour = np.empty(len(x_contour))
            for i in range(len(x_contour)):
                min_contour[i] = np.min(
                    (x_contour[i] - xeq) ** 2 + (z_contour[i] - zeq) ** 2
                )
            imin = np.argmin(min_contour)
            x, z = x_contour[imin], z_contour[imin]
        else:
            x, z = np.array([]), np.array([])
            for i in range(len(x_contour)):
                x = np.append(x, x_contour[i])
                z = np.append(z, z_contour[i])
        if trim:
            if self.xp_location == "lower":
                x, z = x[z <= zeq], z[z <= zeq]
            elif self.xp_location == "upper":
                x, z = x[z >= zeq], z[z >= zeq]
            else:
                raise ValueError("Xloc not set (get_Xpsi)")
            if xeq > self.x_point[0]:
                x, z = x[x > self.x_point[0]], z[x > self.x_point[0]]
            else:
                x, z = x[x < self.x_point[0]], z[x < self.x_point[0]]
            istart = np.argmin((x - xeq) ** 2 + (z - zeq) ** 2)
            x = np.append(x[istart + 1 :], x[:istart])
            z = np.append(z[istart + 1 :], z[:istart])
        istart = np.argmin((x - xeq) ** 2 + (z - zeq) ** 2)
        if istart > 0:
            x, z = x[::-1], z[::-1]
        return x, z, psi

    def firstwall_loop(self, **kwargs):
        """
        Get first wall loop based on a normalised psi value, or geometric offset
        value from the plasma separatrix (psi_norm = 1)
        """
        if self.lfp_x is None:
            self.get_lowfield_points()
        if "psi_n" in kwargs:
            x, z, psi = self.first_wall_psi(psi_n=kwargs["psi_n"], trim=False)
            psi_lfs = psi_hfs = psi
        elif "dx" in kwargs:  # geometric offset
            dx = kwargs.get("dx")
            lf_fwx, lf_fwz = self.lfp_x + dx, self.lfp_z
            hf_fwx, hf_fwz = self.hfp_x - dx, self.hfp_z
            x_lfs, z_lfs, psi_lfs = self.first_wall_psi(point=(lf_fwx, lf_fwz))
            x_hfs, z_hfs, psi_hfs = self.first_wall_psi(point=(hf_fwx, hf_fwz))
            x_top, z_top = self.get_offset(dx)
            if self.xp_location == "lower":
                x_top, z_top = theta_sort(
                    x_top,
                    z_top,
                    xo=self.x_point,  # NOTE: changed from "po" to "xo" kwargs to enable theta_sort kwargs. Needs checking
                    origin="top",
                )
                index = z_top >= self.lfp_z
            else:
                x_top, z_top = theta_sort(x_top, z_top, xo=self.x_point, origin="bottom")
                index = z_top <= self.lfp_z
            x_top, z_top = x_top[index], z_top[index]
            istart = np.argmin((x_top - hf_fwx) ** 2 + (z_top - hf_fwz) ** 2)
            if istart > 0:
                x_top, z_top = x_top[::-1], z_top[::-1]
            x = np.append(x_hfs[::-1], x_top)
            x = np.append(x, x_lfs)
            z = np.append(z_hfs[::-1], z_top)
            z = np.append(z, z_lfs)
        else:
            raise NovaError("requre 'psi_n' or 'dx' in kwargs")

        return x[::-1], z[::-1], (psi_lfs, psi_hfs)

    def get_offset(self, dx, n_sub=0):
        """
        Get an offset loop from the plasma separatrix.
        """
        rpl, zpl = self.get_boundary()  # boundary points
        rpl, zpl = offset_smc(rpl, zpl, dx)  # offset from sep
        if n_sub > 0:  # sub-sample
            rpl, zpl = innocent_smoothie(rpl, zpl, n_sub)
        return rpl, zpl

    def expansion(self, x_sol, z_sol):
        """
        Flux explansion calculation.
        """
        x_expansion = np.array([])
        b_m = np.abs(self.bcentre * self.xcentre)
        for x, z in zip(x_sol, z_sol):
            field = self.point_field([x, z])
            b_p = np.sqrt(field[0] ** 2 + field[1] ** 2)  # polodial feild
            b_tor = b_m / x  # toroidal field
            x_expansion = np.append(x_expansion, b_tor / b_p)  # feild expansion
        return x_expansion

    @staticmethod
    def orientate(x, z):
        """
        Counter-clock-wise orientation
        """
        if x[-1] > x[0]:
            x = x[::-1]
            z = z[::-1]
        return x, z

    def pick_contour(self, contours, xpoint=False, midplane=True, plasma=False):
        """
        Pick a contour
        """
        x_s, z_s = [], []

        is_xpoint, is_midplane, is_plasma = True, True, True
        for psi_line in contours:
            for line in psi_line:
                x, z = line[:, 0], line[:, 1]
                if xpoint:  # check Xpoint proximity
                    r_x = np.sqrt(
                        (x - self.x_point[0]) ** 2 + (z - self.x_point[1]) ** 2
                    )
                    if min(r_x) < self.rcirc:
                        is_xpoint = True
                    else:
                        is_xpoint = False
                if midplane:  # check lf midplane crossing
                    if (np.max(z) > self.lfp_z) and (np.min(z) < self.lfp_z):
                        is_midplane = True
                    else:
                        is_midplane = False
                if plasma:
                    if (
                        (np.max(x) < np.max(self.xbdry))
                        and (np.min(x) > np.min(self.xbdry))
                        and (np.max(z) < np.max(self.zbdry))
                        and (np.min(z) > np.min(self.zbdry))
                    ):
                        is_plasma = True
                    else:
                        is_plasma = False
                if is_xpoint and is_midplane and is_plasma:
                    x, z = self.orientate(x, z)
                    x_s.append(x)
                    z_s.append(z)
        return x_s, z_s

    def store_leg(self, xloop, tloop):
        """
        Store a leg in the StreamFlow object.
        """
        if np.argmin(xloop) > len(xloop) / 2:  # point legs out
            xloop, tloop = xloop[::-1], tloop[::-1]
        ncirc = np.argmin(abs(xloop - self.rcirc))
        t_id = np.argmin(abs(tloop[ncirc] - self.tleg))
        leg_id = self.t_id[t_id]
        if self.nleg == 6:
            if leg_id <= 1:
                label = "inner" + str(leg_id + 1)
            elif leg_id >= 4:
                label = "outer" + str(leg_id - 3)
            elif leg_id == 2:
                label = "core1"
            elif leg_id == 3:
                label = "core2"
            else:
                label = ""
        else:
            if leg_id == 0:
                label = "inner"
            elif leg_id == 3:
                label = "outer"
            elif leg_id == 1:
                label = "core1"
            elif leg_id == 2:
                label = "core2"
            else:
                label = ""
        if label:
            i = self.legs[label]["i"]
            x = xloop * np.sin(tloop) + self.x_point[0]
            if self.xp_location == "lower":
                z = xloop * np.cos(tloop) + self.x_point[1]
            elif self.xp_location == "upper":
                z = -xloop * np.cos(tloop) + self.x_point[1]
            else:
                raise NovaError("Xloc not set (get_Xpsi)")
            if i > 0:
                if x[0] ** 2 + z[0] ** 2 == (
                    self.legs[label]["X"][i - 1][0] ** 2
                    + self.legs[label]["Z"][i - 1][0] ** 2
                ):
                    i -= 1
            if "core" in label:
                x, z = x[::-1], z[::-1]
            self.legs[label]["X"][i] = x
            self.legs[label]["Z"][i] = z
            self.legs[label]["i"] = i + 1

    def snip(self, leg, layer_index=0, l2d=0):
        """
        Snips a separatrix leg to a certain 2-D length.
        """
        if self.x_sol is None:
            self.sol()
        x_sol = self.legs[leg]["X"][layer_index]
        z_sol = self.legs[leg]["Z"][layer_index]
        l_sol = length(x_sol, z_sol)
        if l2d == 0:
            l2d = l_sol[-1]
        if layer_index != 0:
            x_sol_o = self.legs[leg]["X"][0]
            z_sol_o = self.legs[leg]["Z"][0]
            l_sol_o = length(x_sol_o, z_sol_o)
            indexo = np.argmin(np.abs(l_sol_o - l2d))
            index = np.argmin(
                (x_sol - x_sol_o[indexo]) ** 2 + (z_sol - z_sol_o[indexo]) ** 2
            )
            l2d = l_sol[index]
        else:
            index = np.argmin(np.abs(l_sol - l2d))
        if l_sol[index] > l2d:
            index -= 1
        if l2d > l_sol[-1]:
            l2d = l_sol[-1]
            bluemira_warn(
                "Requested SOL target outside grid. " f"{leg} L2D = {l2d:.2f} m"
            )
        x_end, z_end = interp1d(l_sol, x_sol)(l2d), interp1d(l_sol, z_sol)(l2d)
        x_sol, z_sol = x_sol[:index], z_sol[:index]  # trim to strike point
        x_sol, z_sol = np.append(x_sol, x_end), np.append(z_sol, z_end)
        return x_sol, z_sol

    def get_graze(self, point, target):
        """
        Get the grazing at a point, and target vector.
        """
        target = target / np.sqrt(target[0] ** 2 + target[1] ** 2)  # target vector
        field = self.point_field([point[0], point[1]])
        field /= np.sqrt(field[0] ** 2 + field[1] ** 2)  # poloidal feild line vector
        theta = np.arccos(np.dot(field, target))
        if theta > np.pi / 2:
            theta = np.pi - theta
        expansion = self.expansion([point[0]], [point[1]])
        graze = np.arcsin(np.sin(theta) * (expansion[-1] ** 2 + 1) ** -0.5)
        return graze

    @staticmethod
    def strike_point(x_vector, graze):
        """
        Get the strike point for a target and grazing angle.
        """
        ratio = np.sin(graze) * np.sqrt(x_vector[-1] ** 2 + 1)
        if np.abs(ratio) > 1:
            theta = np.sign(ratio) * np.pi
        else:
            theta = np.arcsin(ratio)
        return theta

    def connection(self, leg, layer_index, l_2d=0):
        """
        Get trimmed SOL leg and calculate the connection lengths.
        """
        if l_2d > 0:  # trim targets to L2D
            x_sol, z_sol = self.snip(leg, layer_index, l_2d)
        else:  # rb.trim_sol to trim to targets
            x_sol = self.legs[leg]["X"][layer_index]
            z_sol = self.legs[leg]["Z"][layer_index]
        l_sol = lengthnorm(x_sol, z_sol)
        index = np.append(np.diff(l_sol) != 0, True)
        x_sol, z_sol = x_sol[index], z_sol[index]  # remove duplicates
        if len(x_sol) < 2:
            l_2d, l_3d = [0], [0]
        else:
            dx_sol = np.diff(x_sol)
            dz_sol = np.diff(z_sol)
            l_2d = np.append(0, np.cumsum(np.sqrt(dx_sol ** 2 + dz_sol ** 2)))
            dt_sol = np.array([])
            expansion = self.expansion(x_sol, z_sol)
            for x, dx, dz, xi in zip(x_sol[1:], dx_sol, dz_sol, expansion):
                d_lphi = xi * np.sqrt(dx ** 2 + dz ** 2)
                dt_sol = np.append(dt_sol, d_lphi / (x + dx / 2))
            l_3d = np.append(
                0,
                np.cumsum(
                    dt_sol
                    * np.sqrt(
                        (dx_sol / dt_sol) ** 2
                        + (dz_sol / dt_sol) ** 2
                        + (x_sol[:-1]) ** 2
                    )
                ),
            )
        return l_2d, l_3d, x_sol, z_sol


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
