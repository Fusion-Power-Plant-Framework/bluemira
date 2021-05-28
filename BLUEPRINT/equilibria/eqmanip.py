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
Equilibrium manipulation tools - first attempt at replacing SF in nova
"""
from scipy.interpolate import RectBivariateSpline
from scipy.interpolate import interp1d
import numpy as np
from BLUEPRINT.equilibria.plotting import MagFieldPlotter
from BLUEPRINT.equilibria.find import (
    get_psi_norm,
    get_psi,
    find_flux_loops,
)
from BLUEPRINT.equilibria.constants import PSI_NORM_TOL, PSI_REL_TOL, REL_GRID_SIZER
from BLUEPRINT.geometry.geomtools import lengthnorm
from BLUEPRINT.geometry.loop import Loop
import matplotlib.pyplot as plt
from collections import defaultdict


class MagneticField:
    """
    Magnetic field utility object
    """

    def __init__(self, equilibrium, tfbnd=None):
        self.eq = equilibrium
        self.psi = self.eq.psi()
        self.tfbnd = tfbnd

        # Constructors
        self.x = None
        self.z = None
        self.x_1d = None
        self.z_1d = None
        self.dx = None
        self.dz = None
        self.bt_spline = None
        self.bx_spline = None
        self.bz_spline = None
        self.Bx = None
        self.Bz = None
        self.Bt = None
        self.Bp = None

        self.transfer_grid()
        self.calculate_pol_field()
        self.calculate_tor_field()

    def transfer_grid(self):
        """
        Assigns grid information to MagneticField object, used for plotting
        and general shorthand
        """
        self.x, self.z = self.eq.grid.x, self.eq.grid.z
        self.x_1d, self.z_1d = self.eq.grid.x_1d, self.eq.grid.z_1d
        self.dx, self.dz = self.eq.grid.dx, self.eq.grid.dz

    def calculate_pol_field(self):
        """
        Calculates the radial and vertical field components for a given psi map

        \t:math:` B_x = -\\psi_{z}/X`
        \t:math:` B_z = -\\psi_x/X`

        Establishes RectBivariateSpline interpolant objects for Bx and Bz
        """
        psi_x, psi_z = np.gradient(self.psi, self.dx, self.dz)
        xm = self._x_matrix()
        self.Bx = -psi_z / xm
        self.Bz = psi_x / xm
        self.Bp = np.sqrt(self.Bx ** 2 + self.Bz ** 2)
        self.bx_spline = RectBivariateSpline(self.x_1d, self.z_1d, self.Bx)
        self.bz_spline = RectBivariateSpline(self.x_1d, self.z_1d, self.Bz)

    def _x_matrix(self):
        xm = np.array(np.matrix(self.x_1d).T * np.ones([1, self.x.shape[1]]))
        xm[xm == 0] = 1e-34
        return xm

    def calculate_tor_field(self):
        """
        Calculates the toroidal field component for a given R*Bt
        """
        rb0 = abs(self.eq.fvac())
        rb0 *= np.ones_like(self.psi)
        self.Bt = rb0 / self._x_matrix()
        self.bt_spline = RectBivariateSpline(self.x_1d, self.z_1d, self.Bt)
        # TODO: Implement mask for within TF

    def B(self, x=None, z=None):
        """
        Calculates the fields at the specified points. If not points specified,
        returns the field values on the entire grid.

        Parameters
        ----------
        x: Union[float, np.array, None]
            x coordinates of the point at which the magnetic field is to be
            calculated [m]
        z: Union[float, np.array, None]
            z coordinates of the point at which the magnetic field is to be
            calculated [m]

        Returns
        -------
        Bx: 2-D np.array
            Radial magnetic field
        Bz: 2-D np.array
            Vertical magnetic field
        Btor: 2-D np.array
            Toroidal magnetic field
        """
        if x is None:
            return self.Bx, self.Bz, self.Bt
        return np.array(
            [self.bx_spline.ev(x, z), self.bz_spline.ev(x, z), self.bt_spline.ev(x, z)]
        )

    def plot(self, ax=None):
        """
        Plot the MagneticField.
        """
        return MagFieldPlotter(self, ax=ax)


# TODO: Finish development and replace SF and FirstWall
class EquilibriumManipulator:
    """
    Class used to manipulate Equilibrium objects
    """

    def __init__(self, equilibrium):
        self.eq = equilibrium
        self.filename = self.eq.filename
        self.mag = MagneticField(equilibrium)

        self.ssize = None
        self.x = None
        self.z = None
        self.o_point = None
        self.x_point = None
        self.other_x_points = None
        self.n_x_points = None
        self.eqtype = None
        self.lfp = None
        self.hfp = None
        self.dx_sep = None
        self.n_legs = None
        self.legs = None

        self.define_search_size()
        self.classify_OX_points()
        self.get_midplane_points()
        # self.plot()

    def define_search_size(self):
        """
        Sets the distance around the X-point to search in order to classify
        as a logical flux loop
        """
        dx = abs(self.eq.grid.x_max - self.eq.grid.x_min)
        dz = abs(self.eq.grid.z_max - self.eq.grid.z_min)
        ds = REL_GRID_SIZER * (dx ** 2 + dz ** 2) ** 0.5
        self.ssize = ds

    def classify_OX_points(self):  # noqa (N802)
        """
        Classifica os puntos O e X
        """

        def get_n_xp(xpsis_):
            """
            Returns the number of active X-points
            """
            xpsis_ = np.array(xpsis_)
            xpsin = get_psi_norm(xpsis_, self.o_point.psi, self.x_point.psi)
            err = 1 - np.array(xpsin)
            return sum((abs(err) < PSI_NORM_TOL))

        o_points, x_points = self.eq.get_OX_points()
        self.o_point = o_points[0]
        x_points, xpsis = [], []
        l_points, lpsis = [], []
        for x in x_points:
            # TODO: Figure out why isinstance is not happy here
            if x.__class__.__name__ == "Xpoint":
                x_points.append(x)
                xpsis.append(x.psi)
            if x.__class__.__name__ == "Lpoint":
                l_points.append(x)
                lpsis.append(x.psi)
        if len(lpsis) > 0:
            if lpsis[0] > xpsis[0]:
                # The plasma is limited. but because CREATE throw lots of limiters
                # in the mix (private flux under the divertor), this means nothing.
                pass
                # TODO: develop treatment for this case

        self.x_point = x_points[0]

        self.n_x_points = get_n_xp(xpsis)
        if self.n_x_points == 1:
            if self.x_point.z > 0:
                self.eqtype = "SN_u"
            else:
                self.eqtype = "SN_l"
        elif self.n_x_points == 2:
            if np.sign(x_points[0].z) != np.sign(x_points[1].z):
                self.eqtype = "DN"
                self.other_x_points = [x_points[1]]
            else:
                self.eqtype = "?"
        else:
            self.eqtype = "?"

    def get_midplane_points(self):
        """
        Returns the low and high field LCFS points
        """
        x, z = self.eq.get_LCFS().d2
        lfs = np.argmax(x)
        hfs = np.argmin(x)
        self.lfp = x[lfs], z[lfs]
        self.lfp = self.eq.get_midplane(*self.lfp, self.x_point.psi)
        self.hfp = x[hfs], z[hfs]
        self.hfp = self.eq.get_midplane(*self.hfp, self.x_point.psi)
        if self.eqtype == "DN":
            lfp_alt = self.eq.get_midplane(*self.lfp, self.other_x_points[0].psi)
            self.dx_sep = lfp_alt[0] - self.lfp[0]
        return self.lfp[0], self.lfp[1], self.hfp[0], self.hfp[1]

    def get_sol_psi(self, lambda_q, n_fluxlines=5):
        """
        Parameters
        ----------
        lambda_q: float
            Scrape-off layer width at the outboard midplane [m]
            NOTE: this name is for power decay fall-off length, but who cares?
        n_fluxlines: int (optional, default = 5)
            Number of SOL filaments to calculate

        Returns
        -------
        psis: np.array(n_fluxlines)
            magnetic flux values at SOL filaments
        """
        psi_sol_edge = self.eq.psi(self.lfp[0] + lambda_q, self.lfp[1])[0]
        # NOTE: offset on Xpoint.psi to avoid picking up LCFS
        return np.linspace(psi_sol_edge, self.x_point.psi - PSI_REL_TOL, n_fluxlines)

    def get_sol(self, lambda_q, n_fluxlines=5):
        """
        Parameters
        ----------
        lambda_q: float
            Scrape-off layer width at the outboard midplane [m]
        n_fluxlines: int (optional, default = 5)
            Number of SOL filaments to calculate

        Returns
        -------
        sol: list(Loop)*n_fluxlines
            List of (x, z)Loop objects representing flux lines of the
            scrape-off layer
        """
        psis = self.get_sol_psi(lambda_q, n_fluxlines)
        psis = get_psi_norm(psis, self.o_point.psi, self.x_point.psi)
        sol = []
        for psi in psis:
            sol.extend(self.get_flux_loops(psi))
        return sol

    def get_flux_loops(self, psi_norm):
        """
        Returns all flux loops in psi for a given psi_norm

        Parameters
        ----------
        psi_norm: float
            Normalised psi value relative to.base.O- and X-points

        Returns
        -------
        floops: list(Loop, Loop, ..)
            List of Loop objects representing flux loops
        """
        floops = find_flux_loops(
            self.eq.x,
            self.eq.z,
            self.eq.psi(),
            psi_norm,
            o_points=self.o_point,
            x_points=self.x_point,
        )
        return [Loop(x=loop.T[0], z=loop.T[1]) for loop in floops]

    def get_fw_psinloop(self, psi_n=None, trim=True, **kwargs):
        """
        Gets the most logical flux loop for a given normalised psi value. Able
        to trim it for later X-point handling

        Parameters
        ----------
        psi_n: float
            Normalised psi
        trim: bool
            Default: True. Clips first wall psi loops to ignore X-points

        kwargs:
            'psi': float
                Absolute flux value
            'point': [float, float]
                Geometric coordinates [m]
        """
        if psi_n is None:
            if "psi" in kwargs:
                psi_n = get_psi_norm(kwargs["psi"], self.o_point.psi, self.x_point.psi)
            elif "point" in kwargs:
                psi = self.eq.psi(*kwargs["point"])
                psi_n = get_psi_norm(psi, self.o_point.psi, self.x_point.psi)
            else:
                raise ValueError("Need to specify a function entry point.")
        contours = self.get_flux_loops(psi_n)
        goodloops = []
        for loop in contours:
            x_point, midplane = False, False
            d = np.min(loop.distance_to(self.x_point))
            if d < self.ssize:
                x_point = True
            if (np.min(loop.z) < self.lfp[1]) and np.max(loop.z) > self.lfp[1]:
                midplane = True
            if x_point and midplane:
                goodloops.append(loop)
        if len(goodloops) == 0:
            raise ValueError("No good loops found in EquilibriumManipulator.")
        if len(goodloops) > 1 and "DN" not in self.eqtype:
            raise ValueError("This is a confusing result. Investigate.")
        if trim:  # Ignore region around the divertor ==> handle separately
            if self.x_point.z > 0:
                j = -1
            else:
                j = 1
            if "DN" in self.eqtype:  # Alt null handling
                nloops = []
                for loop in goodloops:
                    args = np.where(
                        (j * loop.z > self.x_point.z)
                        & (j * loop.z < self.other_x_points[0].z)
                    )[0]
                    nloops.append(Loop(*loop[args].T))
            else:
                loop = goodloops[0]
                args = np.where(j * loop.z > self.x_point.z)[0]
                nloops = Loop(*loop[args].T)
            return nloops
        else:
            return goodloops

    def midplane_loop(self, x, z):
        """
        Transforms a loop into an acceptable midplane loop

        Parameters
        ----------
        x, z: np.array
            Coordinates of loop to be modified

        Returns
        -------
        x, z: np.array
            Coordinates of modified loop
        """
        index = np.argmin((x - self.lfp[0]) ** 2 + (z - self.lfp[1]) ** 2)
        if z[index] <= self.lfp[1]:
            index -= 1
        x = np.append(x[: index + 1][::-1], x[index:][::-1])
        z = np.append(z[: index + 1][::-1], z[index:][::-1])
        length_norm = lengthnorm(x, z)
        index = np.append(np.diff(length_norm) != 0, True)
        x, z = x[index], z[index]  # remove duplicates
        return x, z

    def get_fw_loop(self, psi_n=None, dx=None, trim=True):
        """
        Get a firstwall loop based on normalised psi or dx offset.
        """
        # NOTE: sf.firstwall_loop()
        if psi_n is not None:
            nloops = self.get_fw_psinloop(psi_n, trim=trim)
            psi_hfs = psi_lfs = get_psi(psi_n, self.o_point.psi, self.x_point.psi)
        elif dx is not None:
            psi_lfs = self.eq.psi(self.lfp[0] + dx, self.lfp[1])
            psi_hfs = self.eq.psi(self.hfp[0] - dx, self.hfp[1])
            # lfsloop = self.get_fw_psinloop(psi=psi_lfs)
            # hfsloop = self.get_fw_psinloop(psi=psi_hfs)
        else:
            raise ValueError("Must specify either psi_n or dx to get fw loop.")
        return nloops, psi_hfs, psi_lfs

    def get_offset(self, dx):
        """
        Gets a geometrically offset surface relative to the LCFS

        Parameters
        ----------
        dx: float
            Offset from LCFS [m]

        Returns
        -------
        offset: Loop(x, z) object
            Offset Loop from LCFS
        """
        return self.eq.get_LCFS().offset(dx)

    def get_2d_angle(self, legloop, alpha_g):
        """
        Returns the angle in the x-z plane of a target, for a specified
        compound grazing angle

        Parameters
        ----------
        legloop: Loop(x, z) object
            Flux loop object (open field line trace)
        alpha_g: float > 0=>2pi <
            Desired compound grazing angle [°]

        Returns
        -------
        theta: np.array(len(Loop))
            Angle of the target at each point in the leg to meet the specified
            grazing angle [°]

        \t:math:`\\alpha_c=\\alpha_g`
        """
        # NOTE: Understand S. McIntosh calculation here. Book 10, p. 148
        alpha_g = np.deg2rad(alpha_g)
        Bx, Bz, Bt = self.mag.B(*legloop.d2)
        B = np.sqrt(Bx ** 2 + Bz ** 2 + Bt ** 2)
        Bx /= B
        Bz /= B
        Bt /= B
        Bp = np.sqrt(Bx ** 2 + Bz ** 2)
        ratio = np.sin(alpha_g) * np.sqrt(1 + (Bt / Bp) ** 2)
        theta = np.zeros_like(ratio)
        above = np.where(ratio > 1)
        below = np.where(ratio <= 1)
        theta[above] = np.pi * np.sign(ratio[above])
        theta[below] = np.arcsin(ratio[below])
        # =============================================================================
        #         if np.abs(ratio) > 1:
        #             theta = np.sign(ratio)*np.pi
        #         else:
        #             theta = np.arcsin(ratio)
        # =============================================================================
        theta = np.rad2deg(theta)
        return theta

    def get_graze_angle(self, legloop, target):
        """
        \t:math:`\\alpha=\\sin^{-1}\\bigg(\\sin(\\beta)\\dfrac{1}{\\sqrt{1+(B/B_p)^2}}\\bigg)`
        """
        # TODO: Understand S. McIntosh calculation here
        target = target / np.sqrt(target[0] ** 2 + target[1] ** 2)
        Bx, Bz, Bt = self.mag.B(*legloop.d2)
        ratio = Bt / np.sqrt(Bx ** 2 + Bz ** 2)
        Bp = np.array([Bx, Bz]) / np.sqrt(Bx ** 2 + Bz ** 2)
        theta = np.arccos(np.dot(Bp, target))
        above = np.where(np.pi - theta > np.pi / 2)
        theta[above] = np.pi - theta[above]
        # =============================================================================
        #         if theta > np.pi/2:
        #             theta = np.pi-theta
        # =============================================================================
        graze = np.arcsin(np.sin(theta) * (1 + ratio ** 2) ** -0.5)
        return graze

    def get_target_angle(self, legloop, graze):
        """
        \t:math:`\\beta=\\sin^{-1}\\big(\\sin(\\alpha)\\sqrt{1+(B/B_p)^2}\\big)`
        """
        graze = np.deg2rad(graze)
        Bx, Bz, Bt = self.mag.B(*legloop.d2.T[-1])
        ratio = self.get_field_ratio(legloop)[-1]
        beta = np.arcsin(np.sin(graze) / (1 + ratio ** 2) ** 0.5)
        loop_a, loop_b = legloop.d2.T[-1], legloop.d2.T[-2]
        gamma = np.arctan2(loop_a[1] - loop_b[1], loop_a[0] - loop_b[0])
        if gamma > 0:
            gamma *= -1
        if loop_a[0] < self.x_point.x:
            beta = -beta
        if loop_a[1] < 0:
            lower = 1
        else:
            lower = -1
        # ttt = np.arcsin(
        #     ratio * np.cos(graze)
        # )  # https://iopscience.iop.org/article/10.1088/0029-5515/54/7/073022/pdf
        print(ratio, graze)
        tt = np.arccos(np.sin(graze) / ratio)
        print(np.rad2deg(tt))
        alpha_prime = np.arctan(np.sqrt((Bt / (np.tan(graze) * abs(Bz)) ** 2 - 1)))

        print(np.rad2deg(alpha_prime))
        beta, gamma = np.rad2deg(beta), np.rad2deg(gamma)

        return lower * (90 + gamma - beta)

    def get_legs(self, lambda_q):
        """
        Retrieve SOL legs for a specified SOL geometric thickness

        Parameters
        ----------
        lambda_q: float > 0
            SOL geometric thickness [m]

        Returns
        -------
        loops: list(Loop, Loop, ..)
            All SOL loops
        """
        sol = self.get_sol(lambda_q)
        loops = []
        for p in [self.lfp, self.hfp]:
            for loop in sol:
                a = int(loop.argmin(p))
                # TODO: this distance doesn't always work for large lambda_q
                if (
                    np.sqrt(
                        (loop[a][0] - p[0]) ** 2
                        + (loop[a][2] - p[1]) ** 2  # Reasonable distance
                    )
                    < 10 * lambda_q
                ):
                    nloopu = Loop(*loop[a:-1])
                    if nloopu.argmin(loop.d2.T[a]) > 3:
                        nloopu.reverse()
                    nloopd = Loop(*loop[0 : a + 1])
                    if nloopd.argmin(loop.d2.T[a]) > 3:
                        nloopd.reverse()
                    loops.extend([nloopu, nloopd])
        return loops

    def get_sep_legs(self, lambda_q, length_2d):
        """
        Returns the separatrix legs (single element at psi_norm=1)
        """
        if not hasattr(self, "legs"):
            self.classify_legs(lambda_q)
        for i in range(self.n_legs):
            self.snip_leg(i, length_2d)
        seplegs = []
        for leg in self.legs.values():
            seplegs.append(leg[0])
        return seplegs

    def classify_legs(self, lambda_q):
        """
        Counts the number of legs in an equilibrium and numbers them
        """
        legloops = self.get_legs(lambda_q)
        ends = np.zeros((len(legloops), 2))
        for i, loop in enumerate(legloops):
            ends[i] = loop.d2.T[-1]
        hist, x, y = np.histogram2d(*ends.T)
        xbin = np.linspace(x[0], x[-1], len(x) - 1)
        ybin = np.linspace(y[0], y[-1], len(y) - 1)
        self.n_legs = np.sum(hist != 0)  # Bin count
        hx, hy = np.where(hist > 0)

        legs = defaultdict()
        for i in range(self.n_legs):
            legs.setdefault(i, [])
        for loop in legloops:
            end = loop.d2.T[-1]
            xc = np.argmin(abs(xbin - end[0]))
            yc = np.argmin(abs(ybin - end[1]))
            wx, wy = np.where(hx == xc), np.where(hy == yc)
            i = np.intersect1d(wx, wy)[0]
            legs[i].append(loop)
        self.legs = legs

    def snip_leg(self, leg_index, length_2d):
        """
        Cuts SOL leg from closest active X-point to a distance L2D

        Parameters
        ----------
        leg_index: int
            Key of .legs dict
        length_2d: float
            2-D length away from active X-point
        """
        leg = self.legs[leg_index]

        if "DN" in self.eqtype:
            if np.mean(leg[0].z) > 0:
                x_point = self.other_x_points[0]
            else:
                x_point = self.x_point
        else:
            x_point = self.x_point
        loops = []
        for i, loop in enumerate(leg):
            xidx = loop.argmin([x_point.x, x_point.z])
            store = Loop(*loop[:xidx])
            loop = Loop(*loop[xidx:])
            distance = np.sqrt((loop.x - x_point.x) ** 2 + (loop.z - x_point.z) ** 2)

            xe, ze = (
                interp1d(distance, loop.x)(length_2d),
                interp1d(distance, loop.z)(length_2d),
            )
            index = np.where(distance < length_2d)[0]
            loop = Loop(*loop[index].T)
            if loop.argmin([xe, ze]) == 0:
                loop.insert([xe, 0, ze], pos=0)
                loop.reverse()  # Last point is at the target

            elif loop.argmin([xe, ze]) == len(loop) - 1:
                loop.insert([xe, 0, ze], pos=-1)
                # loop._reverse()  # Last point is at the target
            else:
                raise ValueError(f"{loop.argmin([xe, ze])}")
            x = np.concatenate((store.x, loop.x))
            z = np.concatenate((store.z, loop.z))
            final = Loop(x=x, z=z)
            # loop = store.stitch(loop)
            loops.append(final)
        self.legs[leg_index] = loops

    def get_field_ratio(self, legloop):
        """
        Calculates the magnetic field ratio: poloidal field to total field

        Parameters
        ----------
        legloop: Loop object
            Flux loop object (open field line trace)

        Returns
        -------
        Bp/Btot: np.array(len(Loop))
            Ratio of poloidal to total magnetic field at each point in Loop

        \t:math:`B_p/B_{tot} = \\dfrac{\\sqrt{B_x^2+B_z^2}}{\\sqrt{B_x^2+B_z^2+B_t^2}}`
        """
        Bx, Bz, Bt = self.mag.B(*legloop.d2)
        return (Bx ** 2 + Bz ** 2) ** 0.5 / ((Bx ** 2 + Bz ** 2 + Bt ** 2) ** 0.5)

    def get_2d_field_ratio(self, legloop):
        """
        Calculates the magnetic field ratio: poloidal field to toroidal field
        NOTE: this is used in DivertorProfile (traditionally)

        Parameters
        ----------
        legloop: Loop object
            Flux loop object (open field line trace)

        Returns
        -------
        Bp/Btot: np.array(len(Loop))
            Ratio of poloidal to total magnetic field at each point in Loop

        \t:math:`B_p/B_{t} = \\dfrac{\\sqrt{B_x^2+B_z^2}}{B_t}`
        """
        Bx, Bz, Bt = self.mag.B(*legloop.d2)
        return (Bx ** 2 + Bz ** 2) ** 0.5 / Bt

    def get_flux_expansion(self, legloop):
        """
        Calculates the flux expansion along a leg loop (oriented from midplane
        to end point).

        Parameters
        ----------
        legloop: Loop object
            Flux loop object (open field line trace)

        Returns
        -------
        f_x: np.array(len(Loop))
            The flux expansion vector along a leg Loop

        \t:math:`f_x = \\dfrac{\\dfrac{B_p}{B_{tot}}\\bigg\\rvert_{midplane}}{\\dfrac{B_p}{B_{tot}}\\bigg\\rvert_{target}}`
        """  # noqa (W505)
        fx = self.get_field_ratio(legloop)
        return fx[0] / fx

    def get_connection_length(self, legloop, calc="3D"):
        """
        Calculates the connection length of a leg loop (oriented from midplane
        to end point).

        \t:math:`L_{3D} = \\sum_{i=0}^{n} \\sqrt{dx_i^2+dz_i^2+d\\phi_i^2}`

        with:
        \t:math:`d\\phi_i = \\sqrt{dx_i^2+dz_i^2}\\dfrac{B_{\\phi}}{B_{pol}}`
        """
        x, z = legloop.d2
        dx, dz = np.diff(x), np.diff(z)
        if calc == "3D":
            dpol = np.sqrt(dx ** 2 + dz ** 2)
            Bx, Bz, Bt = self.mag.B(x, z)
            f3d = Bt / np.sqrt(Bx ** 2 + Bz ** 2)
            dphi = dpol * f3d[:-1]
            l_3d = np.sum(np.sqrt(dx ** 2 + dz ** 2 + dphi ** 2))
            return l_3d
        elif calc == "2D":
            l_2d = np.sum(np.sqrt(dx ** 2 + dz ** 2))
            return l_2d
        elif calc == "3Dsmc":

            dt_sol = np.array([])
            Bx, Bz, Bt = self.mag.B(x, z)
            b_ratio = Bt / np.sqrt(Bx ** 2 + Bz ** 2)
            for xi, dx, dz, bi in zip(x[1:], dx, dz, b_ratio):
                dl_p = np.sqrt(dx ** 2 + dz ** 2)
                dl_phi = bi * dl_p
                dt_sol = np.append(dt_sol, dl_phi / (xi + dx / 2))
            l_3d = np.append(
                0,
                np.cumsum(
                    dt_sol
                    * np.sqrt((dx / dt_sol) ** 2 + (dx / dt_sol) ** 2 + (x[:-1]) ** 2)
                ),
            )
            return l_3d[-1]
        else:
            raise ValueError(
                f"calc type {calc} not supported. Please choose " '"L2D" or "L3D"'
            )

    def plot(self, ax=None):
        """
        Plot the EquilibriumManipulator.
        """
        if ax is None:
            f, ax = plt.subplots()
        self.eq.plot(ax)

        if hasattr(self, "legs"):
            col = ["k", "r", "b", "g"]
            for c, (k, leg) in zip(col[: self.n_legs], self.legs.items()):
                for loop in leg:
                    loop.plot(fill=False, edgecolor=c)
        ax.plot(*self.lfp, marker="o")
        ax.plot(*self.hfp, marker="o")

    # DEVELOPMENT ONLY - SPINOFF LATER
    def set_target(self, leg, l_target, graze):
        """
        Parameters
        ----------
        leg: int
            Leg identificaiton number
        graze: float
            Grazing angle [degrees]
        """
        leg = self.legs[leg]
        xt, zt = leg[-1].d2.T[-1]

        angle = self.get_target_angle(leg[-1], graze)
        a = np.deg2rad(angle)
        xu, zu = xt - np.sin(a) * l_target, zt + np.cos(a) * l_target
        xl, zl = xt + np.sin(a) * l_target, zt - np.cos(a) * l_target
        x, z = [xu, xt, xl], [zu, zt, zl]
        return x, z


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
