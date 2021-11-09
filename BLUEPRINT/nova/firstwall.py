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
First wall and divertor profile calculation algorithms
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Type, List
from scipy.interpolate import InterpolatedUnivariateSpline
import nlopt
from collections import OrderedDict

from bluemira.base.parameter import ParameterFrame
from bluemira.base.look_and_feel import bluemira_warn
from bluemira.utilities.optimiser import approx_derivative

from BLUEPRINT.nova.stream import StreamFlow
from BLUEPRINT.systems.baseclass import ReactorSystem
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.parameterisations import PictureFrame, PolySpline
from BLUEPRINT.geometry.stringgeom import String
from BLUEPRINT.geometry.shape import Shape
from BLUEPRINT.geometry.boolean import boolean_2d_difference, boolean_2d_union
from BLUEPRINT.geometry.geomtools import (
    lengthnorm,
    rotate_vector_2d,
    tangent,
    unique,
    order,
    vector_intersect,
    xz_interp,
)
from BLUEPRINT.geometry.offset import offset_clipper
from bluemira.geometry._deprecated_tools import innocent_smoothie


class FirstWallProfile(ReactorSystem):
    """
    Wraps a preliminary first wall shape around the plasma
    Can then do a panelled FW shape
    """

    config: Type[ParameterFrame]
    inputs: dict
    # fmt: off
    default_params = [
        ['plasma_type', 'Type of plasma', 'SN', 'N/A', None, 'Input'],
        ['n_TF', 'Number of TF coils', 16, 'N/A', None, 'Input'],
        ["A", "Plasma aspect ratio", 3.1, "N/A", None, "Input"],
        ["R_0", "Major radius", 9, "m", None, "Input"],
        ["kappa_95", "95th percentile plasma elongation", 1.6, "N/A", None, "Input"],
        ["fw_dL_min", "Minimum FW module length", 0.75, "m", None, "Input"],
        ["fw_dL_max", "Maximum FW module length", 3, "m", None, "Input"],
        ["fw_a_max", "Maximum angle between FW modules", 20, "°", None, "Input"],
        ['r_fw_ib_in', 'Inboard first wall inner radius', 5.8, 'm', None, 'PROCESS'],
        ['r_fw_ob_in', 'Outboard first wall inner radius', 12.1, 'm', None, 'PROCESS'],
    ]
    # fmt: on
    npts = 250

    def __init__(self, config, inputs):
        self.config = config
        self.inputs = inputs

        self._init_params(self.config)

        self.name = self.inputs["name"] + "_firstwallprofile"
        self.shp = None
        self.sf_list = None
        self.fw_config = None
        self.initalise_loop()

    def initalise_loop(self):
        """
        Initiliase the FirstWallProfile shape parameterisation.
        """
        self.shp = Shape(
            self.name,
            family=self.inputs["parameterisation"],
            objective="L",
            npoints=self.npts,
            read_write=False,
        )
        self.set_bounds()

    def set_bounds(self):
        """
        Set the initial variable values and bounds for the shape parameterisation.
        """
        r_minor = self.params.R_0 / self.params.A
        height = (self.params.kappa_95 * r_minor) * 2

        ib_radius = self.params.r_fw_ib_in
        ob_radius = self.params.r_fw_ob_in

        # TODO: update this to match whatever shp parametersiation is specced
        # this only works for "S" or "BS"
        if isinstance(self.shp.parameterisation, PolySpline):
            self.shp.adjust_xo(
                "x1", value=ib_radius, lb=ib_radius - 0.1, ub=ib_radius + 0.001
            )  # inboard radius
            self.shp.adjust_xo(
                "x2", value=ob_radius, lb=ob_radius - 0.001, ub=ob_radius * 1.1
            )  # outboard radius
            self.shp.adjust_xo(
                "z2", value=0, lb=-0.9, ub=0.9
            )  # outer node vertical shift
            self.shp.adjust_xo(
                "height", value=height + 0.001, lb=height, ub=50
            )  # full loop height
            self.shp.adjust_xo("top", value=0.5, lb=0.05, ub=1)  # horizontal shift
            self.shp.adjust_xo("upper", value=0.7, lb=0.2, ub=1)  # vertical shift
            self.shp.parameterisation.set_lower()
            self.shp.adjust_xo("dz", value=0, lb=-5, ub=5)  # vertical offset
            self.shp.adjust_xo(
                "flat", value=0, lb=0, ub=0.8
            )  # fraction outboard straight
            self.shp.adjust_xo("tilt", value=0, lb=-45, ub=45)  # outboard angle [deg]
            self.shp.adjust_xo("upper", lb=0.7)
            self.shp.adjust_xo("top", lb=0.05, ub=0.75)
            self.shp.adjust_xo("lower", lb=0.7)
            self.shp.adjust_xo("bottom", lb=0.05, ub=0.75)
            self.shp.adjust_xo("l", lb=0.8, ub=1.5)
            self.shp.adjust_xo("tilt", lb=-25, ub=25)
            # self.shp.remove_oppvar('flat')
            # self.shp.remove_oppvar('tilt')
        elif isinstance(self.shp.parameterisation, PictureFrame):
            # Drop the corner radius from the optimisation
            self.shp.remove_oppvar("r")

            # Set the inner radius of the shape, and pseudo-remove from optimiser
            self.shp.adjust_xo(
                "x1", value=ib_radius, lb=ib_radius - 0.01, ub=ib_radius + 0.01
            )

            self.shp.adjust_xo(
                "x2", value=ob_radius, lb=ob_radius - 0.01, ub=ob_radius + 0.01
            )

            half_height = height * 0.5

            # Adjust bounds to fit problem
            self.shp.adjust_xo(
                "z1", lb=half_height, value=half_height + 0.1, ub=half_height * 1.5
            )
            self.shp.adjust_xo(
                "z2", lb=-half_height * 1.5, value=-half_height - 0.1, ub=-half_height
            )

    def generate(
        self,
        eq_names: List[str],
        dx=0.225,
        psi_n=1.07,
        flux_fit=False,
        symetric=False,
        verbose=False,
    ):
        """
        Generate the FirstWallProfile shape. Performs an optimisation for minimum
        length.
        """
        if isinstance(self.shp.parameterisation, PolySpline):
            self.shp.parameterisation.reset_oppvar(symetric)  # reset loop oppvar
        self.set_bounds()
        self.fw_config = {
            "dx": dx,
            "psi_n": psi_n,
            "flux_fit": flux_fit,
            "Nsub": 100,
            "eqdsk": [],
        }
        sf_list = self.load_sf(eq_names)
        self.sf_list = sf_list
        for sf in sf_list:  # convert input to list
            self.add_bound(sf)
        # self.shp.add_interior(r_gap=0.001)  # add interior bound
        self.shp.optimise(verbose=verbose)

    def load_sf(self, eq_names):
        """
        Load a set of StreamFlow objects from files into the problem.
        """
        sf_dict, sf_list = OrderedDict(), []

        for i, eq_name in enumerate(eq_names):
            sf = StreamFlow(eq_name)
            sf_dict[i] = sf.filename.split("/")[-1]
            sf_list.append(sf)
        self.fw_config["eqdsk"] = sf_dict
        return sf_list

    def plot_xz(self, ax=None, **kwargs):
        """
        Plots the X-Z cross-section view of the FirstWallProfile
        """
        if ax is None:
            _, ax = plt.subplots()

        self.shp.parameterisation.plot(ax=ax)
        self.shp.plot_bounds(ax=ax)
        x, z = self.draw()
        ax.plot(x, z)
        ax.set_aspect("equal")

    def add_bound(self, sf):
        """
        Add a bound the FW shape optimisation problem based on a StreamFlow object.
        """
        xpl, zpl = sf.get_offset(self.fw_config["dx"], n_sub=self.fw_config["Nsub"])
        sf.get_lowfield_points()
        # vessel inner bounds
        self.shp.add_bound({"x": xpl, "z": zpl}, "internal")
        x_point = sf.x_point  # _array[:, 0]  # select lower
        self.shp.add_bound(
            {"x": x_point[0] + 0.12 * sf.shape["a"], "z": x_point[1]}, "internal"
        )
        self.shp.add_bound(
            {"x": x_point[0], "z": x_point[1] - 0.01 * sf.shape["a"]}, "internal"
        )

        if self.fw_config["flux_fit"] and self.params.plasma_type == "SN":
            # Make a open Loop of the upper part of the psi_n offset surface
            sf.get_lowfield_points()  # get low field point
            xflux, zflux = sf.first_wall_psi(psi_n=self.fw_config["psi_n"], trim=False)[
                :2
            ]

            xflux, zflux = sf.midplane_loop(xflux, zflux)
            xflux, zflux = order(xflux, zflux)
            istop = next((i for i in range(len(zflux)) if zflux[i] < sf.lfp_z), -1)
            xflux, zflux = xflux[:istop], zflux[:istop]
            d_l = np.diff(lengthnorm(xflux, zflux))
            if np.max(d_l) > 3 * np.median(d_l) or np.argmax(zflux) == len(zflux) - 1:
                wtxt = "\nOpen field line detected\n"
                wtxt += "disabling flux fit for "
                wtxt += "{:1.1f}% psi_n \n".format(1e2 * self.fw_config["psi_n"])
                wtxt += "configuration: " + sf.filename + "\n"
                bluemira_warn("Nova::FirstWallProfile:\n" + wtxt)
                return

        elif self.fw_config["flux_fit"] and self.params.plasma_type == "DN":
            # Here we need a closed surface to bound the FW shape both top and
            # bottom.. so we cut the legs off the open flux loops and join
            sf.get_lowfield_points()  # get low field point
            x, z = sf.first_wall_psi(psi_n=self.fw_config["psi_n"], trim=False)[:2]
            _, (_, zmin) = sf.get_x_psi(select="lower")
            clip = np.where(z > zmin)
            x, z = x[clip], z[clip]
            _, (_, zmax) = sf.get_x_psi(select="upper")
            clip = np.where(z < zmax)
            x, z = x[clip], z[clip]
            # Close the bound profile
            xflux = np.append(x, x[0])
            zflux = np.append(z, z[0])

        # add flux_fit bounds
        xflux, zflux = innocent_smoothie(xflux, zflux, int(self.fw_config["Nsub"] / 2))
        self.shp.add_bound({"x": xflux, "z": zflux}, "internal")

    def draw(self, npoints=250):
        """
        Draw the FirstWallProfile shape.
        """
        x = self.shp.parameterisation.draw(npoints=npoints)
        x, z = x["x"], x["z"]
        x, z = order(x, z, anti=True)
        return x, z

    def panelise(self, bb_fw_loop, cut_location):
        """
        Makes a flat panelled FW loop based on the exclusion flux surfaces

        Parameters
        ----------
        bb_fw_loop: Loop
            The smooth loop of the inner breeding blanket 2D profile
        cut_location: [float, float]
            The x, z coordinates of the location of the inboard/outboard cut on
            the first wall loop
        """
        self.geom["fw_loop"] = bb_fw_loop

        if self.params.plasma_type == "SN":
            i = bb_fw_loop.argmin(cut_location)
            ob = Loop.from_array(bb_fw_loop[:i])
            ib = Loop.from_array(bb_fw_loop[i:])
            loops = [ib, ob]

        else:  # Double null panelisation
            # MultiLoop for FW panels (already segmented - len = 2)
            loops = sorted(bb_fw_loop.loops, key=lambda g: np.min(g.x))

        for name, seg in zip(["inboard panels", "outboard panels"], loops):
            self.geom[name] = self.make_panels(*seg.d2)

        xyz = np.concatenate(
            (self.geom["outboard panels"].xyz.T, self.geom["inboard panels"].xyz.T)
        )
        fw_cut_shape = Loop(*xyz.T)

        if self.params.plasma_type == "SN":
            fw_cut_shape.close()  # Sanity: ensure closed loop in case of SN

        self.geom["2D profile"] = fw_cut_shape

    def make_panels(self, x, z):
        """
        Panelise the first wall.
        """
        p = Paneller(
            x, z, self.params.fw_a_max, self.params.fw_dL_min, self.params.fw_dL_max
        )
        p.optimise()
        return Loop(**dict(zip(["x", "z"], p.d2)))

    def make_pretty_plot(self, **kwargs):
        """
        Make a pretty plot of the first wall and flux surfaces.
        """
        ax = kwargs.get("ax", plt.gca())
        ib_col, ob_col = "k", "r"
        ib, ob = self.geom["inboard panels"], self.geom["outboard panels"]
        ax.plot(
            *self.geom["fw_loop"].d2,
            "gray",
            ls="--",
            lw=2,
            label="Preliminary FW profile"
        )
        ib, ob = ib.offset(0.25), ob.offset(0.25)
        ax.plot(*ib.d2, ib_col, lw=4, label="Inboard FW profile and module segmentation")
        ibo = ib.offset(0.5)
        for i in range(len(ibo)):
            ax.plot([ibo[i][0], ib[i][0]], [ibo[i][2], ib[i][2]], ib_col)
        ax.plot(
            *ob.d2, ob_col, lw=4, label="Outboard FW profile and module segmentation"
        )
        obo = ob.offset(0.5)
        for i in range(len(ob)):
            ax.plot([obo[i][0], ob[i][0]], [obo[i][2], ob[i][2]], ob_col)
        for i, e in enumerate(self.sf_list):
            xs, zs = e.get_boundary()
            x, z = e.get_offset(self.fw_config["dx"], n_sub=self.fw_config["Nsub"])

            e.get_lowfield_points()  # get low feild point
            xflux, zflux = e.first_wall_psi(psi_n=self.fw_config["psi_n"], trim=False)[
                :2
            ]

            xflux, zflux = e.midplane_loop(xflux, zflux)
            xflux, zflux = order(xflux, zflux)
            istop = next((i for i in range(len(zflux)) if zflux[i] < e.lfp_z), -1)
            xflux, zflux = xflux[:istop], zflux[:istop]

            if i == 0:
                plabel = "Plasma equilibrium separatrices"
                olabel = "$\\partial{x}\\partial{z}$ offset surfaces"
                flabel = "$\\psi_{n}$ offset half-surfaces"
            else:
                plabel, olabel, flabel = "", "", ""
            ax.plot(xs, zs, "pink", lw=2, label=plabel)
            ax.plot(x, z, "#0072bd", lw=2, label=olabel)
            ax.plot(xflux, zflux, "#7e2f8e", lw=2, label=flabel)
        ax.set_aspect("equal")
        ax.set_xlabel("$x$ [m]")
        ax.set_ylabel("$z$ [m]")
        return ax


class Paneller:
    """
    Makes panels for first wall segments for inboard and outboard blankets.
    a.k.a. "wrapper" from nova.modules.
    Needs smooth x, z coords of an open loop from an initial shape of the first
    wall.
    """

    def __init__(self, x, z, angle, dx_min, dx_max):
        self.x, self.z = x, z
        tx, tz = tangent(self.x, self.z)
        length_norm = lengthnorm(self.x, self.z)

        self.loop = {
            "x": InterpolatedUnivariateSpline(length_norm, x),
            "z": InterpolatedUnivariateSpline(length_norm, z),
        }
        self.tangent = {
            "x": InterpolatedUnivariateSpline(length_norm, tx),
            "z": InterpolatedUnivariateSpline(length_norm, tz),
        }
        points = np.array([x, z]).T
        string = String(points, angle=angle, dx_min=dx_min, dx_max=dx_max, verbose=False)
        self.n_opt = string.n - 2
        self.n_constraints = self.n_opt - 1 + 2 * (self.n_opt + 2)  # constraint number
        self.x_opt = length_norm[string.index][1:-1]
        self.dl_limit = {"min": dx_min, "max": dx_max}
        self.d2 = None
        self.bounds = None

    def fw_corners(self, x_opt):
        """
        Get the corner points and indices
        """
        x_opt = np.sort(x_opt)
        x_opt = np.append(np.append(0, x_opt), 1)
        p_corner = np.zeros((len(x_opt) - 1, 2))  # corner points
        p_o = np.array([self.loop["x"](x_opt), self.loop["z"](x_opt)]).T
        t_o = np.array([self.tangent["x"](x_opt), self.tangent["z"](x_opt)]).T
        for i in range(self.n_opt + 1):
            p_corner[i] = vector_intersect(
                p_o[i], p_o[i] + t_o[i], p_o[i + 1], p_o[i + 1] + t_o[i + 1]
            )
        p_corner = np.append(p_o[0].reshape(1, 2), p_corner, axis=0)
        p_corner = np.append(p_corner, p_o[-1].reshape(1, 2), axis=0)
        return p_corner, p_o

    def fw_length(self, x_opt, index):
        """
        Get the first wall length or panel length.
        """
        p_corner = self.fw_corners(x_opt)[0]
        d_l = np.sqrt(np.diff(p_corner[:, 0]) ** 2 + np.diff(p_corner[:, 1]) ** 2)
        length = np.sum(d_l)
        data = [length, d_l]
        return data[index]

    def fw_vector(self, x_opt, grad):
        """
        Optimisation objective for the Paneller.
        """
        length = self.fw_length(x_opt, 0)
        if grad.size > 0:

            grad[:] = approx_derivative(
                self.fw_length, x_opt, rel_step=1e-6, f0=length, args=(0,)
            )

        return length

    def set_cmm(self, x_opt, cmm, index):  # min max constraints
        """
        Lo adjusted so that all limit values are negative
        """
        d_l = self.fw_length(x_opt, 1)
        cmm[: self.n_opt + 2] = self.dl_limit["min"] - d_l
        cmm[self.n_opt + 2 :] = d_l - self.dl_limit["max"]
        return cmm[index]

    def constrain_length(self, constraint, x_opt, grad):
        """
        Constrain the lengths of the flat first wall panels.
        """
        d_l_space = 1e-3  # minimum inter-point spacing

        if grad.size > 0:
            grad[:] = np.zeros((self.n_constraints, self.n_opt))  # initalise
            for i in range(self.n_opt - 1):  # order points
                grad[i, i] = -1
                grad[i, i + 1] = 1

            for i in range(2 * self.n_opt + 4):
                grad[self.n_opt - 1 + i, :] = approx_derivative(
                    self.set_cmm,
                    x_opt,
                    rel_step=1e-6,
                    bounds=self.bounds,
                    args=(np.zeros(2 * self.n_opt + 4), i),
                )

        constraint[: self.n_opt - 1] = (
            x_opt[: self.n_opt - 1] - x_opt[1 : self.n_opt] + d_l_space
        )
        self.set_cmm(x_opt, constraint[self.n_opt - 1 :], 0)

    def optimise(self):
        """
        Optimise the panelling of the profile.
        """
        opt = nlopt.opt(nlopt.LD_SLSQP, self.n_opt)
        opt.set_ftol_rel(1e-4)
        opt.set_ftol_abs(1e-4)
        opt.set_min_objective(self.fw_vector)

        opt.set_lower_bounds([0 for _ in range(self.n_opt)])
        opt.set_upper_bounds([1 for _ in range(self.n_opt)])

        self.bounds = np.array(
            [np.zeros(self.n_opt, dtype=np.int), np.ones(self.n_opt, dtype=np.int)]
        )

        tol = 1e-2 * np.ones(self.n_constraints)
        opt.add_inequality_mconstraint(self.constrain_length, tol)
        self.x_opt = opt.optimize(self.x_opt)

        if opt.last_optimize_result() < 0:
            bluemira_warn("Nova::Paneller: optimiser unconverged")

        p_corners = self.fw_corners(self.x_opt)[0]
        self.d2 = np.array([p_corners[:, 0], p_corners[:, 1]])
        return

    def plot(self):
        """
        Plot the Paneller result.
        """
        p_corner, p_o = self.fw_corners(self.x_opt)
        plt.plot(self.x, self.z)
        plt.plot(p_corner[:, 0], p_corner[:, 1], "-o", ms=8)
        plt.axis("equal")


class DivertorProfile(ReactorSystem):
    """
    Builds the divertor profile based on the desired reference equilibria
    Needs a Nova StreamFlow object
    """

    config: Type[ParameterFrame]
    inputs: dict
    # fmt: off
    default_params = [
        ["fw_psi_n", "Normalised psi boundary to fit FW to", 1.07, "N/A", None, "Input"],
        ["fw_dx", "Minimum distance of FW to separatrix", 0.225, "m", None, "Input"],
        ["div_L2D_ib", "Inboard divertor leg length", 1.1, "m", None, "Input"],
        ["div_L2D_ob", "Outboard divertor leg length", 1.3, "m", None, "Input"],
        ["div_graze_angle", "Divertor SOL grazing angle", 1.5, "°", None, "Input"],
        ["div_psi_o", "Divertor flux offset", 0.5, "m", None, "Input"],
        ["div_Ltarg", "Divertor target length", 0.5, "m", None, "Input"],
        ["tk_div", "Divertor thickness", 0.5, "m", None, "Input"],
        ["dx_div", "Don't know", 0, "N/A", None, "Input"],
        ["bb_gap", "Gap to breeding blanket", 0.05, "m", None, "Input"],
        ["Xfw", "Don't know", 0, "N/A", None, "Input"],
        ["Zfw", "Don't know", 0, "N/A", None, "Input"],
        ["psi_fw", "Don't know", 0, "N/A", None, "Input"],
        ["c_rm", "Remote maintenance clearance", 0.05, "m", "Distance between IVCs", "Input"],
    ]
    # fmt: on

    def __init__(self, config, inputs):
        self.config = config
        self.inputs = inputs

        self.params = ParameterFrame(self.default_params.to_records())
        self._init_params(self.config)

        self.sf = self.inputs["sf"]
        self.targets = self.inputs["targets"]
        self.debug = self.inputs["debug"]
        self.params.dx_div = self.params.tk_div.value
        self.params.bb_gap = self.params.c_rm.value

        if inputs["flux_conformal"]:
            loop = self.sf.firstwall_loop(psi_n=self.params.fw_psi_n)
        else:
            loop = self.sf.firstwall_loop(dx=self.params.fw_dx)

        self.params.Xfw, self.params.Zfw, self.params.psi_fw = loop

    def set_target(self, leg, **kwargs):
        """
        Set a target for a separatrix leg.
        """
        if leg not in self.targets:
            self.targets[leg] = {}
        for key in self.targets["default"]:
            if key in kwargs:
                self.targets[leg][key] = kwargs[key]  # update
            elif key not in self.targets[leg]:  # prevent overwrite
                self.targets[leg][key] = self.targets["default"][key]

    def select_layer(self, leg, layer_index=[0, -1]):
        """
        Select a SOL layer from a leg.
        """
        sol, x_i = [], []
        for layer in layer_index:
            x, z = self.sf.snip(leg, layer, l2d=self.targets[leg]["L2D"])
            sol.append((x, z))
            x_i.append(self.sf.expansion([x[-1]], [z[-1]]))
        index = np.argmin(x_i)  # min expansion / min graze
        return sol[index][0], sol[index][1]

    def plot_xz(self, ax=None):
        """
        Plot the DivertorProfile.
        """
        if ax is None:
            _, ax = plt.subplots()

        ax.plot(self.geom["divertor"]["x"], self.geom["divertor"]["z"])

    def make_divertor(self, fw_loop: Type[Loop], location: str):
        """
        Makes a divertor geometry

        Parameters
        ----------
        fw_loop: Loop
            The inner loop of the initial first wall profile
        location: str
            Where to position the divertor ["upper", "lower"]

        Returns
        -------
        geom: dict
            The geometry dictionary of the divertor
        """
        self.sf.get_x_psi(select=location)  # Which X-point to build off
        self.sf.sol()
        self.sf.get_legs()  # Needed for up/down divertors, if any

        geom = {}  # Geometry sub-structure for the divertor

        div_prof = self._make_divertor_profile()

        xd, zd = unique(*div_prof.d2)[:2]

        if self.sf.xp_location == "lower":
            zindex = zd <= self.sf.x_point[1] + 0.5 * (
                self.sf.o_point[1] - self.sf.x_point[1]
            )
        else:
            zindex = zd >= self.sf.x_point[1] + 0.5 * (
                self.sf.o_point[1] - self.sf.x_point[1]
            )
        xd, zd = xd[zindex], zd[zindex]  # remove upper points
        xd, zd = xz_interp(xd, zd)  # resample

        # divertor inner wall
        div_temp = Loop(x=xd, z=zd)
        geom["div_temp"] = div_temp  # debugging
        div_temp.close()
        # Clip the divertor inner flux-following line with the inner loop
        div_inner = boolean_2d_difference(div_temp, fw_loop)[0]
        geom["divertor_inner"] = div_inner

        # Join the inner divertor Loop to the base FW inner loop
        geom["first_wall"] = boolean_2d_union(fw_loop, geom["divertor_inner"])[0]

        # divertor outer wall
        # Offset the inner divertor loop to get a thick wavy plate
        div_outer = offset_clipper(
            geom["divertor_inner"], self.params.dx_div, method="square"
        )

        # Clip this with the inner loop
        geom["divertor_outer"] = boolean_2d_difference(div_outer, fw_loop)[0]
        # Offset this outer divertor shape to get the gap to the BB and vessel
        geom["divertor_gap"] = offset_clipper(geom["divertor_outer"], self.params.bb_gap)
        geom["vessel_gap"] = boolean_2d_union(fw_loop, geom["divertor_gap"])[0]
        # Now, make the divertor space reservation
        geom["divertor"] = boolean_2d_difference(
            geom["divertor_outer"], geom["divertor_inner"]
        )[0]
        # Now we have to make an open Loop...
        inner = fw_loop.copy()
        div_koz = geom["divertor_gap"]
        # Let's find a point on the inner loop that is inside the divertor KOZ

        count = 0
        for i, point in enumerate(inner):
            if div_koz.point_inside(point):
                # Now we re-order the loop and open it, such that it is open
                # inside the KOZ
                if count > 1:
                    inner.reorder(i, 0)
                    inner.open_()
                    break
                count += 1  # (Second point inside the loop)

        # Now perform a boolean cut on an open loop
        geom["blanket_inner"] = boolean_2d_difference(inner, div_koz)[0]
        return geom

    def _make_divertor_profile(self):
        x, z = self.sf.x_point
        # This right here took a lot of tweaking.. if there is a problem in
        # future, best to start looking at flux_cond
        flux_cond = self.sf.point_psi([x, z - 1]) > self.sf.point_psi([x, z + 1])
        if flux_cond:
            self.dir_sign = 1
        else:
            self.dir_sign = -1
        self.dir_sign = 1

        for leg in list(self.sf.legs)[2:]:  # Ignore first two legs (core1 and core2)
            self.set_target(leg)
            x_sol, z_sol = self.select_layer(leg)
            x_o, z_o = x_sol[-1], z_sol[-1]

            flips = [-1, 1]
            directions = [1, -1]
            theta_ends = [0, np.pi]
            theta_sign = 1

            if self.sf.xp_location == "upper":
                theta_sign *= -1
                directions = directions[::-1]

            if "inner" in leg:
                psi_plasma = self.params.psi_fw[1]
            else:
                psi_plasma = self.params.psi_fw[0]

            dpsi = self.params.psi_fw[1] - self.sf.x_psi
            phi_target = [psi_plasma, self.sf.x_psi - self.params.div_psi_o * dpsi]

            if leg == "inner1" or leg == "outer2":
                phi_target[0] = self.sf.x_psi + self.params.div_psi_o * dpsi

            if self.targets[leg]["open"]:
                theta_sign *= -1
                directions = directions[::-1]
                theta_ends = theta_ends[::-1]

            if "outer" in leg:
                directions = directions[::-1]
                theta_sign *= -1

            if leg == "inner1" or leg == "outer2":
                theta_ends = theta_ends[::-1]

            # Build divertor "cups" around each leg
            x_leg = np.array([])
            z_leg = np.array([])
            d_plate = self.targets[leg]["dPlate"]
            graze = self.targets[leg]["graze"]

            for flip, direction, theta_end, psi_target in zip(
                flips, directions, theta_ends, phi_target
            ):
                x, z = self._match_psi(
                    x_o,
                    z_o,
                    direction,
                    theta_end,
                    theta_sign,
                    psi_target,
                    graze,
                    d_plate,
                    leg,
                )
                x_leg = np.append(x_leg, x[::flip])
                z_leg = np.append(z_leg, z[::flip])
            if leg == "outer":
                x_leg = x_leg[::-1]
                z_leg = z_leg[::-1]

            self.targets[leg]["X"] = x_leg
            self.targets[leg]["Z"] = z_leg

        # Connect the target cups
        x_b, z_b = np.array([]), np.array([])
        if self.sf.nleg == 6:  # StreamFlow
            x_b = np.append(x_b, self.targets["inner2"]["X"][1:])
            z_b = np.append(z_b, self.targets["inner2"]["Z"][1:])
            x, z = self._connect(
                self.sf.x_psi - self.params.div_psi_o * dpsi,
                ["inner2", "inner1"],
                [-1, -1],
            )
            x_b, z_b = self._append(x_b, z_b, x, z)
            x_b = np.append(x_b, self.targets["inner1"]["X"][::-1])
            z_b = np.append(z_b, self.targets["inner1"]["Z"][::-1])
            x, z = self._connect(
                self.sf.x_psi + self.params.div_psi_o * dpsi,
                ["inner1", "outer2"],
                [0, 0],
            )
            x_b, z_b = self._append(x_b, z_b, x, z)
            x_b = np.append(x_b, self.targets["outer2"]["X"][1:])
            z_b = np.append(z_b, self.targets["outer2"]["Z"][1:])
            x, z = self._connect(
                self.sf.x_psi - self.params.div_psi_o * dpsi,
                ["outer2", "outer1"],
                [-1, -1],
            )
            x_b, z_b = self._append(x_b, z_b, x, z)
            x_b = np.append(x_b, self.targets["outer1"]["X"][::-1])
            z_b = np.append(z_b, self.targets["outer1"]["Z"][::-1])

        else:
            x_b = np.append(x_b, self.targets["inner"]["X"][1:])
            z_b = np.append(z_b, self.targets["inner"]["Z"][1:])
            x, z = self._connect(
                self.sf.x_psi - self.params.div_psi_o * dpsi, ["inner", "outer"], [-1, 0]
            )
            x_b, z_b = self._append(x_b, z_b, x, z)
            x_b = np.append(x_b, self.targets["outer"]["X"][1:])
            z_b = np.append(z_b, self.targets["outer"]["Z"][1:])

        return Loop(x=x_b, z=z_b)

    def _connect(self, psi, target_pair, ends, loop=[]):
        gap = []
        if loop:
            x, z = loop
        else:
            psi_line = self.sf.get_contour([psi])[0]
            for line in psi_line:
                x, z = line[:, 0], line[:, 1]
                gap.append(
                    np.min(
                        (self.targets[target_pair[0]]["X"][ends[0]] - x) ** 2
                        + (self.targets[target_pair[0]]["Z"][ends[0]] - z) ** 2
                    )
                )
            select = np.argmin(gap)
            line = psi_line[select]
            x, z = line[:, 0], line[:, 1]
        index = np.zeros(2, dtype=int)
        index[0] = np.argmin(
            (self.targets[target_pair[0]]["X"][ends[0]] - x) ** 2
            + (self.targets[target_pair[0]]["Z"][ends[0]] - z) ** 2
        )
        index[1] = np.argmin(
            (self.targets[target_pair[1]]["X"][ends[1]] - x) ** 2
            + (self.targets[target_pair[1]]["Z"][ends[1]] - z) ** 2
        )
        if index[0] > index[1]:
            index = index[::-1]
        x, z = x[index[0] : index[1] + 1], z[index[0] : index[1] + 1]
        return x, z

    def _match_psi(
        self,
        x_o,
        z_o,
        direction,
        theta_end,
        theta_sign,
        phi_target,
        graze,
        d_plate,
        leg,
    ):
        gain = 0.1  # (used to be 0.25 - 0.1 catches more cases)
        n_max = 200  # 500 -> 300 -> 200
        l_values = [5.0, 0.0015]  # [blend,turn]  5,0.015
        x2m = [-2, -1]  # ramp to step (+ive-lead, -ive-lag ramp==1, step==inf)
        n_plate = 1  # number of target plate divisions (1==flat)
        l_1 = l_values[0] if theta_end == 0 else l_values[1]
        l_seed = l_1
        flag = 0
        for i in range(n_max):
            x, z, phi = self._blend_target(
                x_o,
                z_o,
                d_plate,
                l_1,
                direction,
                theta_end,
                theta_sign,
                graze,
                x2m,
                n_plate,
            )
            l_1 += self.dir_sign * gain * (phi_target - phi)
            if self.debug:
                plt.plot(x, z, color="C0", lw=1)
            if np.abs(phi_target - phi) < 1e-4:
                if self.debug:
                    plt.plot(x, z, "x", color="C1", lw=3)
                break
            if l_1 < 0:
                l_1 = 1
                gain *= -1
            if i == n_max - 1 or l_1 > 15:
                wtxt = " ".join([str(leg), "dir", str(direction)])
                wtxt += "\n"
                wtxt += " ".join(["target", str(phi_target), "phi", str(phi)])
                wtxt += "\n"
                wtxt += " ".join(["Nmax", str(i + 1), "L", str(l_1), "Lo", str(l_seed)])
                bluemira_warn(
                    "Nova::DivertorProfile: phi target convergence " "error\n" + wtxt
                )
                if flag == 0:
                    break
                    gain *= -1  # reverse gain
                    flag = 1
                else:
                    break
        return x, z

    def _blend_target(
        self,
        x_o,
        z_o,
        d_plate,
        length,
        direction,
        theta_end,
        theta_sign,
        graze,
        x2m,
        n_plate,
    ):
        x2s = x2m[0] if theta_end == 0 else x2m[1]
        d_l = 0.1 if theta_end == 0 else 0.05  # 0.005,0.005
        x, z = np.array([x_o]), np.array([z_o])
        x, z = self._extend_target(
            x,
            z,
            d_plate / (2 * n_plate),
            n_plate,
            x2s,
            theta_end,
            theta_sign,
            direction,
            graze,
            False,
            target=True,
        )  # constant graze
        n_interp = int(d_plate / (2 * d_l))
        if n_interp < 2:
            n_interp = 2
        x, z = xz_interp(x, z, n_interp)
        # update graze
        graze = self.sf.get_graze([x[-1], z[-1]], [x[-1] - x[-2], z[-1] - z[-2]])
        n_range = np.int(length / d_l + 1)
        if n_range < 30:
            n_range = 30
        d_l = length / (n_range - 1)
        target_angle = np.arctan2((z[-1] - z[-2]), (x[-1] - x[-2]))
        expansion = self.sf.expansion([x[-1]], [z[-1]])
        theta = self.sf.strike_point(expansion, graze)
        field = direction * self.sf.point_field((x[-1], z[-1]))
        b_hat = rotate_vector_2d(field, theta_sign * theta)
        trans_angle = np.arctan2(b_hat[1], b_hat[0])
        if abs(target_angle - trans_angle) > 0.01 * np.pi:
            accute = True
        else:
            accute = False
        x, z = self._extend_target(
            x, z, d_l, n_range, x2s, theta_end, theta_sign, direction, graze, accute
        )  # transition graze
        phi = self.sf.point_psi([x[-1], z[-1]])
        return x, z, phi

    def _extend_target(
        self,
        x,
        z,
        d_l,
        n_range,
        x2s,
        theta_end,
        theta_sign,
        direction,
        graze,
        accute,
        target=False,
    ):
        for i in range(n_range):
            if target:
                l_hat = 0
            else:
                l_hat = i / (n_range - 1)
                if x2s < 0:  # delayed transtion
                    l_hat = l_hat ** np.abs(x2s)
                else:  # expedient transition
                    l_hat = l_hat ** (1 / x2s)
            expansion = self.sf.expansion([x[-1]], [z[-1]])
            theta = self.sf.strike_point(expansion, graze)
            if accute:
                theta = np.pi - theta
            theta = l_hat * theta_end + (1 - l_hat) * theta
            field = direction * self.sf.point_field((x[-1], z[-1]))
            b_hat = rotate_vector_2d(field, theta_sign * theta)
            x = np.append(x, x[-1] + d_l * b_hat[0])
            z = np.append(z, z[-1] + d_l * b_hat[1])
        return x, z

    @staticmethod
    def _append(x, z, x_new, z_new):
        dx = np.zeros(2)
        for i, end in enumerate([0, -1]):
            dx[i] = (x[-1] - x_new[end]) ** 2 + (z[-1] - z_new[end]) ** 2
        if dx[1] < dx[0]:
            x_new, z_new = x_new[::-1], z_new[::-1]
        return np.append(x, x_new[1:-1]), np.append(z, z_new[1:-1])


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
