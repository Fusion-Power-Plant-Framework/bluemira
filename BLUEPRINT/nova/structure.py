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
Coil structure creation algorithms - binding TF and PF coils together
"""
from copy import deepcopy
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin_slsqp, minimize, minimize_scalar

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.parameter import ParameterFrame
from bluemira.equilibria.positioner import XZLMapper
from bluemira.geometry._deprecated_tools import get_intersect
from bluemira.geometry.constants import VERY_BIG
from BLUEPRINT.base.error import NovaError
from BLUEPRINT.cad.coilCAD import CoilStructureCAD
from BLUEPRINT.geometry.boolean import boolean_2d_difference
from BLUEPRINT.geometry.geombase import Plane
from BLUEPRINT.geometry.geomtools import (
    distance_between_points,
    length,
    loop_plane_intersect,
    normal,
    xz_interp,
)
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.shell import Shell
from BLUEPRINT.systems.baseclass import ReactorSystem
from BLUEPRINT.systems.plotting import ReactorSystemPlotter


class CoilArchitect(ReactorSystem):
    """
    The architect object for designing tokamak coil structures:
    - outer inter-coil structures
    - connections between CS and TF coils
    - connections between PF and TF coils
    - cold mass gravity supports
    """

    config: Type[ParameterFrame]
    inputs: dict
    # fmt: off
    default_params = [
        ['n_TF', 'Number of TF coils', 16, 'dimensionless', None, 'Input'],
        ['tk_tf_nose', 'TF coil inboard nose thickness', 0.6, 'm', None, 'Input'],
        ['tk_tf_wp', 'TF coil winding pack thickness', 0.5, 'm', 'Excluding insulation', 'PROCESS'],
        ['tk_tf_front_ib', 'TF coil inboard steel front plasma-facing', 0.04, 'm', None, 'Input'],
        ['tk_tf_ins', 'TF coil ground insulation thickness', 0.08, 'm', None, 'Input'],
        ['tk_tf_insgap', 'TF coil WP insertion gap', 0.1, 'm', 'Backfilled with epoxy resin (impregnation)', 'Input'],
        ['tk_tf_side', 'TF coil inboard case minimum side wall thickness', 0.1, 'm', None, 'Input'],
        ['tk_tf_case_out_in', 'TF coil case thickness on the outboard inside', 0.35, 'm', None, 'Input'],
        ['tk_tf_case_out_out', 'TF coil case thickness on the outboard outside', 0.4, 'm', None, 'Input'],
        ['tf_wp_width', 'TF coil winding pack radial width', 0.76, 'm', 'Including insulation', 'PROCESS'],
        ['tf_wp_depth', 'TF coil winding pack depth (in y)', 1.05, 'm', 'Including insulation', 'PROCESS'],
        ['x_g_support', 'TF coil gravity support radius', 13, 'm', None, 'Input'],
        ['w_g_support', 'TF coil gravity support width', 0.75, 'm', None, 'Input'],
        ['tk_oic', 'Outer inter-coil structure thickness', 0.3, 'm', None, 'Input'],
        ['tk_pf_support', 'PF coil support plate thickness', 0.15, 'm', None, 'Input'],
        ['gs_z_offset', 'Gravity support vertical offset', -1, 'm', None, 'Input'],
        ['g_cs_tf', 'Gap between CS and TF', 0.05, 'm', None, 'Input'],
        ['h_cs_seat', 'Height of the CS support', 2, 'm', None, 'Input'],
        ['min_OIS_length', 'Minimum length of an inter-coil structure', 0.5, 'm', None, 'Input'],
    ]
    # fmt: on
    CADConstructor = CoilStructureCAD

    def __init__(self, config, inputs):
        self.config = config
        self.inputs = inputs
        self._plotter = CoilArchitectPlotter()

        self._init_params(self.config)

        self.tf = self.inputs["tf"]
        self.pf = self.inputs["pf"]
        self.exclusions = self.inputs["exclusions"]

        # Constructors
        self.tf_fun = None
        self.tf_fun_offset = None
        self.tfl = {}
        self.case_loops = {}
        self.xsections = {}
        self.x_nose = None
        # Just a storage for misc. things that are only needed for CAD
        self.geom["feed 3D CAD"] = {"other": {}}

    def build(self):
        """
        Perform the CoilArchitect operations.
        """
        # tf loop interpolators
        self.tf_fun = self.tf.loop_interpolators(offset=0)
        self.tf_fun_offset = self.tf.loop_interpolators(offset=-0.15)
        # split loop CL for later use
        self.tfl = self._split_loop()

        # Get case loops for structural optimisation
        self.case_loops["outer"] = self.tf.geom["TF case out"].outer.copy()
        self.case_loops["inner"] = self.tf.geom["TF case in"].inner.copy()

        wp = self.tf.geom["WP inboard X-Y single"].copy()
        dx, dy = wp.centroid
        wp.translate([-dx, -dy, 0], update=True)
        self.geom["wp"] = Loop(y=wp.y, z=wp.x)
        self._calculate_parameters()

        self._build_CS_support()
        self._build_PF_supports()
        self._build_gravity_supports()
        self._build_OIC_structures()

        self._generate_xz_loops()

    @property
    def xz_plot_loop_names(self):
        """
        The x-z loop names to plot.
        """
        names = list(self.geom["feed 3D CAD"]["PF"].keys())
        names += list(self.geom["feed 3D CAD"]["OIS"].keys())
        names += ["CS support", "G support"]
        return names

    def plot_xz(self, ax=None, **kwargs):
        """
        Plot the CoilArchitect result.
        """
        loops_pf = [self.geom[key] for key in self.geom["feed 3D CAD"]["PF"].keys()]
        loops_ois = [self.geom[key] for key in self.geom["feed 3D CAD"]["OIS"].keys()]
        loops_support = [self.geom[key] for key in ["CS support", "G support"]]
        self._plotter.plot_xz(loops_pf, ax=ax, **kwargs)
        self._plotter.plot_xz(loops_ois, ax=ax, **kwargs)
        self._plotter.plot_xz(loops_support, ax=ax, **kwargs)

    def plot_connections(self, ax=None, **kwargs):
        """
        Plots the connections in the CoilArchitect result (useful for
        developing)
        """
        if ax is None:
            ax = kwargs.get("ax", plt.gca())

        self.plot_xz(ax=ax, **kwargs)

        self.tf.plot_xz(ax=ax)
        self.pf.plot_xz(ax=ax)

        for pf_support in self.geom["feed 3D CAD"]["PF"].values():
            ax.plot(pf_support["p"]["x"], pf_support["p"]["z"], "o-", color="grey")

        gsupport = self.geom["feed 3D CAD"]["Gsupport"]
        ax.plot(
            [gsupport["Xo"], gsupport["Xo"]],
            [gsupport["zbase"], gsupport["zfloor"]],
            "o-",
            color="grey",
        )

        for name, support in self.geom["feed 3D CAD"]["OIS"].items():
            ax.plot(support["p"]["x"], support["p"]["z"], "o-", color="grey")

        ax.plot(self.tfl["cl_fe"]["x"], self.tfl["cl_fe"]["z"], "o")
        for name in ["nose", "loop", "trans_lower", "trans_upper"]:
            x, z = self.tfl[name]["x"], self.tfl[name]["z"]
            ax.plot(x, z)

    def _generate_xz_loops(self):
        """
        Generates some Loop objects for plotting and CAD purposes
        """
        clip = self.tf.geom["TF case out"].outer
        # Make the PF support structure geometry
        for name, coil in self.geom["feed 3D CAD"]["PF"].items():
            x, z = np.array(coil["nodes"]).T
            loop = Loop(x=x, z=z)
            loop.close()
            diff = boolean_2d_difference(loop, clip)
            if diff:
                # The casing has absorbed the PF structure
                self.geom[name] = diff[0]
                self.geom["feed 3D CAD"]["PF"][name]["loop"] = diff[0]

        # Clip the TF casing with PF clearances
        for name, coil in self.pf.coils.items():
            if "PF" in name:
                loop = Loop(x=coil.x_corner, z=coil.z_corner)
                loop.close()
                loop.offset(self.params.g_cs_tf)
                diff = boolean_2d_difference(clip, loop)
                if diff:
                    clip = diff[0]

        for name, support in self.geom["feed 3D CAD"]["OIS"].items():
            x, z = support["nodes"]
            loop = Loop(x=x, z=z)
            self.geom[name] = loop
            self.geom["feed 3D CAD"]["OIS"][name]["loop"] = loop

        cssupport = Loop(
            x=self.geom["feed 3D CAD"]["CS"]["x"], z=self.geom["feed 3D CAD"]["CS"]["z"]
        )
        cssupport.close()
        cssupport = boolean_2d_difference(cssupport, clip)[0]
        self.geom["CS support"] = cssupport
        self.geom["feed 3D CAD"]["CS"]["loop"] = cssupport

        x, z = np.array(self.geom["feed 3D CAD"]["Gsupport"]["base"]).T
        gsupport = Loop(x=x, z=z)
        gsupport.close()
        gsupport = boolean_2d_difference(gsupport, clip)[0]
        self.geom["G support"] = gsupport
        self.geom["feed 3D CAD"]["Gsupport"]["loop"] = gsupport

    def _split_loop(self):
        """
        Split TF centreline for use in finite element models (later)
        """
        x, z = self.tf.loops["cl"]["x"], self.tf.loops["cl"]["z"]
        npts = len(x)
        index = self.tf.transition_index(x, z)
        upper, lower = index["upper"], index["lower"]
        top, bottom = index["top"], index["bottom"]
        p = {}
        for part in ["trans_lower", "loop", "trans_upper", "nose"]:
            p[part] = {}
        p["nose"]["x"] = np.append(x[upper - 1 : -1], x[: lower + 1])
        p["nose"]["z"] = np.append(z[upper - 1 : -1], z[: lower + 1])
        p["trans_lower"]["x"] = x[lower:bottom]
        p["trans_lower"]["z"] = z[lower:bottom]
        p["trans_upper"]["x"] = x[top:upper]
        p["trans_upper"]["z"] = z[top:upper]
        p["loop"]["x"] = x[bottom - 1 : top + 1]
        p["loop"]["z"] = z[bottom - 1 : top + 1]
        cl_length = length(x, z)[-1]
        n_per_m = npts / cl_length  # nodes per length
        tfl = {"cl_fe": {}}
        tfl["cl_fe"]["x"] = np.array([p["nose"]["x"][-1]])
        tfl["cl_fe"]["z"] = np.array([p["nose"]["z"][-1]])
        n_start = 0
        for part in ["trans_lower", "loop", "trans_upper", "nose"]:
            tfl[part] = {}
            l_part = length(p[part]["x"], p[part]["z"])[-1]
            n_part = int(n_per_m * l_part) + 1
            tfl[part]["x"], tfl[part]["z"] = xz_interp(
                p[part]["x"], p[part]["z"], npoints=n_part
            )

            for var in ["x", "z"]:  # append centerline
                tfl["cl_fe"][var] = np.append(tfl["cl_fe"][var], tfl[part][var][1:])
            n_start += n_part - 1

        tfl["cl_fe"]["x"] = tfl["cl_fe"]["x"][:-1]
        tfl["cl_fe"]["z"] = tfl["cl_fe"]["z"][:-1]

        for loop in ["in", "out", "cl"]:
            # Copy these across from the TF object (copying is safe..)
            tfl[loop] = deepcopy(self.tf.loops[loop])

        # Convert everything to loop objects
        for loop in tfl:
            tfl[loop] = Loop.from_dict(tfl[loop])
        return tfl

    def _calculate_parameters(self):

        theta = np.pi / self.params.n_TF
        width = self.tf.section["winding_pack"]["width"]
        depth = self.tf.section["winding_pack"]["depth"]

        rsep = (depth / 2 + self.params.tk_tf_side) / np.tan(theta)

        xo, zo = self._calculate_CS_seat()

        xnose = xo - self.params.g_cs_tf
        if rsep <= xnose:
            ynose = depth / 2 + self.params.tk_tf_side
        else:
            ynose = xnose * np.tan(theta)
        xwp = xo - (width + self.params.tk_tf_nose)
        if rsep <= xwp:
            ywp = depth / 2 + self.params.tk_tf_side
        else:
            ywp = xwp * np.tan(theta)

        xwp = xo + self.params.tk_tf_nose
        cs = {
            "xnose": xnose,
            "ynose": ynose,
            "znose": zo,
            "xwp": xwp,
            "ywp": ywp,
            "x": [xnose, xwp, xwp, xnose],
        }
        self.geom["feed 3D CAD"]["CS"] = cs

        dx_in = self.tf.geom["WP inboard X-Y single"].centroid[0]

        case_in = self.tf.geom["Case inboard X-Y single"].outer
        case_out = self.tf.geom["Case outboard X-Y single"].outer

        z_in = case_in.x[1:4] - dx_in
        y_in = np.abs(case_in.y[1:4])

        y_out = np.abs(case_out.y[1:4])

        self.xsections["case_in"] = {"y": y_in, "z": z_in}

        z_out = [
            -width / 2 - self.params.tk_tf_case_out_in,
            np.mean([self.params.tk_tf_case_out_in, self.params.tk_tf_case_out_out]),
            width / 2 + self.params.tk_tf_case_out_in,
        ]

        self.xsections["case_out"] = {"y": y_out, "z": np.array(z_out)}
        self.x_nose = self.tf.geom["Case inboard X-Y single"].outer.x[2]

    def _calculate_CS_seat(self):  # noqa :N802
        """
        Calculates the position of the CS support seat
        """
        # Get TF outer casing loop
        tf_out = self.tfl["out"]
        # Find the highest point on the inboard casing where the TF is straight
        x_min = np.min(tf_out["x"])
        idx = [i for i, v in enumerate(tf_out["x"]) if np.isclose(v, x_min, rtol=1e-2)]
        z_values = tf_out["z"][idx]
        idx = idx[np.argmax(z_values)]
        xo, zo = tf_out["x"][idx], tf_out["z"][idx]  # This is the "nose bump"

        # Check that this is not too close to the top
        # (we might have a square coil)
        z_max = np.max(tf_out["z"])
        zo = min(zo, 0.9 * z_max)  # Eyeballing here
        return xo, zo

    def _build_CS_support(self):  # noqa :N802
        """
        Builds the CS supports onto the TF coils
        """
        znose = self.geom["feed 3D CAD"]["CS"]["znose"]
        ztop = znose + self.params.h_cs_seat
        self.geom["feed 3D CAD"]["CS"]["z"] = [znose, znose, ztop, ztop]
        self.geom["feed 3D CAD"]["CS"]["ztop"] = ztop

    def _build_PF_supports(self):  # noqa :N802
        """
        Builds the PF supports
        """
        space = self.params.tf_wp_depth / 2 + self.params.tk_tf_side
        pf_supports = {}

        for name, coil in self.pf.coils.items():
            if coil.ctype == "PF":
                coildict = {"x": coil.x, "z": coil.z, "dx": coil.dx, "dz": coil.dz}
                nodes, p, ndir = self._connect_PF(
                    coildict,
                    self.tf_fun_offset["out"],
                    edge=0.15,
                    hover=0.1,
                    ang_min=35,
                )
                width = coil.dx * np.sin(ndir * np.pi / 180)
                pf_supports[name] = {"nodes": nodes, "p": p, "width": width}
                self._adjust_tf_node(p["x"][-1], p["z"][-1])

        self.geom["feed 3D CAD"]["PF"] = pf_supports
        self.geom["feed 3D CAD"]["other"]["PF"] = {
            "n": 3,
            "tk": self.params.tk_pf_support,
            "space": space,
        }

    def _connect_PF(self, coil, loop, edge, hover, ang_min):  # noqa :N802
        """
        Connects a PF coil to the TF coil offset
        """
        x_star = self._min_distance([coil["x"], coil["z"]], loop)
        x_tf = loop["x"](x_star)
        z_tf = loop["z"](x_star)
        x_c, z_c, dx, dz = coil["x"], coil["z"], coil["dx"], coil["dz"]

        nhat = np.array([x_tf - x_c, z_tf - z_c])
        ndir = 180 / np.pi * np.arctan(abs(nhat[1] / nhat[0]))  # angle [degrees]
        if ndir < ang_min:
            # check limit of absolute support angle and update
            nhat = np.array(
                [np.sign(nhat[0]), np.tan(ang_min * np.pi / 180) * np.sign(nhat[1])]
            )
            ndir = 180 / np.pi * np.arctan(abs(nhat[1] / nhat[0]))

        nhat /= np.linalg.norm(nhat)
        above = np.sign(np.dot(nhat, [0, 1]))
        zc = z_c + above * (dz + hover)

        nodes = [[] for _ in range(4)]
        xo = np.array([x_star, 0.5])
        for i, sign in enumerate([-1, 1]):  # inboard / outboard
            xc = x_c + sign * (dx + edge)
            nodes[i] = [xc, zc]
            xc = np.array([xc, zc])
            x_res = self._min_intersect(xo, xc, nhat, self.tf_fun_offset["out"])
            rs, zs = x_res * nhat + xc
            nodes[3 - i] = [rs, zs]

        nd = {"x": np.zeros(3), "z": np.zeros(3)}  # cl, outboard, pf
        for i in range(2):  # cl, pf [0,2]
            nd["x"][2 * i] = np.mean([nodes[2 * i][0], nodes[2 * i + 1][0]])
            nd["z"][2 * i] = np.mean([nodes[2 * i][1], nodes[2 * i + 1][1]])

        ndloop = Loop(x=[nd["x"][0], nd["x"][2]], z=[nd["z"][0], nd["z"][2]])

        # TF out intersect
        xout, zout = get_intersect(self.tfl["out"], ndloop)

        # Occasionally, no intersection will be found.
        if len(xout) == 0:
            # If so, extend the line to get an intersection
            d_x = nd["x"][0] - nd["x"][2]
            d_z = nd["z"][0] - nd["z"][2]
            vec = np.array([d_x, d_z])
            vec /= np.linalg.norm(vec)

            ndloop = Loop(
                x=[nd["x"][0], nd["x"][0] + VERY_BIG * vec[0]],
                z=[nd["z"][0], nd["z"][0] + VERY_BIG * vec[1]],
            )
            xout, zout = get_intersect(self.tfl["out"], ndloop)

        nd["x"][1] = xout[0]  # outboard TF node [1]
        nd["z"][1] = zout[0]

        xc = [nd["x"][-1], nd["z"][-1]]
        x_res = self._min_intersect(xo, xc, nhat, self.tf_fun["cl"])

        x, z = x_res * nhat + xc
        nd["x"][-1], nd["z"][-1] = self._cl_snap(x, z)

        return nodes, nd, ndir

    def _cl_snap(self, x, z):
        """
        Snaps point to the TF centreline
        """
        x_star = self._min_distance([x, z], self.tf_fun["cl"])
        return self.tf_fun["cl"]["x"](x_star), self.tf_fun["cl"]["z"](x_star)

    def _adjust_tf_node(self, x, z):
        x, z = self._cl_snap(x, z)  # snap node to centreline
        i = self.tfl["cl_fe"].argmin([x, z])

        dl = 0.5 * np.sqrt(
            np.sum((self.tfl["cl_fe"].d2.T[i + 1] - self.tfl["cl_fe"].d2.T[i - 1]) ** 2)
        )

        dx = np.sqrt(
            (self.tfl["cl_fe"]["x"][i] - x) ** 2 + (self.tfl["cl_fe"]["z"][i] - z) ** 2
        )
        if dx > 0.5 * dl:  # add node (0.2*)
            distance_1 = self._min_distance([x, z], self.tf_fun["out"])

            xx = self.tfl["cl_fe"]["x"][i]
            zz = self.tfl["cl_fe"]["z"][i]

            distance_2 = self._min_distance([xx, zz], self.tf_fun["out"])

            j = 0 if distance_1 < distance_2 else 1
            self.tfl["cl_fe"].x = np.insert(self.tfl["cl_fe"]["x"], i + j, x)
            self.tfl["cl_fe"].z = np.insert(self.tfl["cl_fe"]["z"], i + j, z)
        else:  # adjust node
            self.tfl["cl_fe"]["x"][i], self.tfl["cl_fe"]["z"][i] = x, z

    def _build_gravity_supports(self):
        width = self.params.w_g_support.value
        x_support = self.params.x_g_support.value
        # bounds to force negative (lower half of tf) result
        x_star = self._min_x(x_support - width / 2, self.tf_fun["out"], bounds=[0, 0.5])
        coil = {
            "x": self.tf_fun["out"]["x"](x_star) + width / 2,
            "z": self.tf_fun["out"]["z"](x_star) - width / 2 + self.params.gs_z_offset,
            "dx": width / 2,
            "dz": width / 2,
        }
        nodes, _, _ = self._connect_PF(
            coil, self.tf_fun_offset["out"], edge=0, hover=0, ang_min=90
        )

        # Adjust GS base node lower z values
        z_base = float(self.tf_fun["out"]["z"](x_star))
        nodes[0][1] = z_base
        nodes[1][1] = z_base

        z = [coil.z - coil.dz for coil in self.pf.coils.values()]
        floor = np.min(z) + self.params.gs_z_offset
        gsupport = {
            "gs_type": self.inputs["gs_type"],
            "Xo": x_support,
            "width": width,
            "tf_wp_depth": self.params.tf_wp_depth.value,
            "tk_tf_side": self.params.tk_tf_side.value,
            "zbase": z_base,
            "zfloor": floor,
            "base": nodes,
        }
        if self.inputs["gs_type"] == "JT60SA":
            gsupport["rtube"] = width / 3  # tube radius
            gsupport["ttube"] = width / 9  # tube thickness
            theta = np.pi / self.params.n_TF  # half angle
            alpha = np.arctan(
                (x_support * np.tan(theta) - width)
                / (gsupport["zbase"] - gsupport["zfloor"] - width / 2)
            )
            pin2pin = np.sqrt(
                (x_support * np.tan(theta) - width) ** 2
                + (gsupport["zbase"] - gsupport["zfloor"] - width / 2) ** 2
            )
            gsupport["alpha"] = alpha
            gsupport["pin2pin"] = pin2pin
            gsupport["spread"] = pin2pin * np.sin(alpha) + 2 * width
            gsupport["zground"] = (
                pin2pin * (1 - np.cos(alpha)) + nodes[1][1] - 1.5 * width - pin2pin
            )
            gsupport["yfloor"] = pin2pin * np.sin(alpha)
        elif self.inputs["gs_type"] == "ITER":
            gsupport["yfloor"] = 0.0
            # Pick the lowest PF point
            gsupport["zground"] = floor
            gsupport["plate_width"] = 0.020

        else:
            raise NovaError(
                f"Unrecognised gravity support design: {self.inputs['gs_type']}"
            )

        self.geom["feed 3D CAD"]["Gsupport"] = gsupport

    def _build_OIC_structures(self):  # noqa :N802
        """
        Designs the outer inter-coil structures
        """
        # Drop the inboard (TFs already connected by nose)
        x_in = min(self.tfl["cl"]["x"])
        idx = np.where(self.tfl["cl"]["x"] > x_in + 0.5)
        loop = Loop(x=self.tfl["cl"]["x"][idx], z=self.tfl["cl"]["z"][idx])
        positioner = XZLMapper(loop)

        positioner.add_exclusion_zones(self.exclusions)
        self.XZL = positioner
        # Identify good places to put OIS (between ports)
        zones = positioner.incl_loops

        # Build constraint loop (offset for OIS support thickness)
        internal = Loop(x=self.tfl["in"]["x"], z=self.tfl["in"]["z"])
        # NOTE: need to boost thickness because they are connected to
        # the casing at an angle!
        theta = np.pi / self.params.n_TF
        tk = self.params.tk_oic / np.cos(theta)
        constraint = internal.offset(tk / 2)

        oic_support = {}
        i = 0
        # Sort the zones so that the OIS are always in the same order
        for zone in sorted(zones, key=lambda l: l.x[0]):
            if zone.length < self.params.min_OIS_length:
                # Drop sections too small to be of any use
                continue

            # For each viable section of the TF coil centreline, optimise OIS
            name = f"OIS_{i}"

            tool = OISOptimiser(zone, constraint)
            x, z = tool.optimise()

            nodes = self._make_OIS_nodes(x, z, tk)
            oic_support[name] = {"p": {"x": x, "z": z}, "nodes": nodes}
            i += 1

        self.geom["feed 3D CAD"]["OIS"] = oic_support

    def get_tf_xsection(self, x, z, dz):
        """
        Calculates the cross-section of the TF coil and winding pack at a given
        x location, for use in finite element analysis. The returned Shell is
        centred at the TF coil winding pack centre (not the geometric centre)
        and is in the y-z plane (for use in `beams`).

        Parameters
        ----------
        x: float
            The x location along the TF coil current centreline at which the
            cross-section is desired
        z: float
            The z location along the TF coil current centreline (in practice,
            only the sign of z matters - for asymmetric coils)
        dz: float
            The direction of the beam in the z-direction

        Returns
        -------
        xsection: BLUEPRINT::geometry::Shell object
            The Shell of the TF coil casing and winding pack, centred at the
            current filament centroid
        """

        def get_first_intersection(loop, line):
            x_inter, z_inter = get_intersect(loop, line)
            distances = []
            for xi, zi in zip(x_inter, z_inter):
                distances.append(distance_between_points([x, z], [xi, zi]))

            if not distances:
                f, ax = plt.subplots()
                loop.plot(ax)
                line.plot(ax)
                plt.show()

            arg = np.argmin(distances)
            return x_inter[arg], z_inter[arg]

        # First, make the winding pack, whose centroid will also be the centre
        # of the beam element (current centre at which the forces are calc-ed)
        wp = self.geom["wp"]

        # Find the argument on the centreline
        i = self.tfl["cl_fe"].argmin([x, z])

        # Get the normal vector at the point
        n_x, n_z = normal(*self.tfl["cl_fe"].d2)
        n_hat = VERY_BIG * np.array([n_x[i], n_z[i]])

        # Project onto the inner and outer casing loops
        line_out = Loop(x=[x, x + n_hat[0]], z=[z, z + n_hat[1]])
        line_in = Loop(x=[x, x - n_hat[0]], z=[z, z - n_hat[1]])

        x_out, z_out = get_first_intersection(self.case_loops["outer"], line_out)
        x_in, z_in = get_first_intersection(self.case_loops["inner"], line_in)

        d_out = distance_between_points([x, z], [x_out, z_out])
        d_in = distance_between_points([x, z], [x_in, z_in])

        # Check for "nose" condition
        if x_out < self.x_nose:  # -> polygonal case
            nose_line = Loop(x=[self.x_nose, self.x_nose], z=[-VERY_BIG, VERY_BIG])
            x_nose, z_nose = get_first_intersection(nose_line, line_out)
            d_nose = distance_between_points([x, z], [x_nose, z_nose])

            y_1 = self.xsections["case_in"]["y"][0]
            y_2 = self.xsections["case_in"]["y"][1]

            y = [y_1, y_2, y_2, -y_2, -y_2, -y_1, y_1]
            z = [-d_out, -d_nose, d_in, d_in, -d_nose, -d_out, -d_out]

        else:  # -> rectangular case
            y_2 = self.xsections["case_out"]["y"][1]
            y = [y_2, y_2, -y_2, -y_2, y_2]
            z = [-d_out, d_in, d_in, -d_out, -d_out]

        # Construct case loop
        case = Loop(y=y, z=z)

        if dz < 0:
            # Invert the casing cross-section
            case.rotate(theta=180, p1=[0, 0, 0], p2=[1, 0, 0])

        # Return the x-section shell
        return Shell(wp, case)

    def get_max_casing_tks(self):
        """
        Calculate the maximum TF casing thicknesses.

        Returns
        -------
        tk_inner: float
            The maximum inner TF casing thickness
        tk_outer: float
            The maximum outer TF casing thickness
        """
        # The inner thickness is to the VVTS, effectively a radial build value.
        # Consider overwriting this if you think the designs look silly.
        vvts = self.tf.inputs["koz_loop"]
        wp_inner = self.tf.geom["WP inner"]
        xy_plane = Plane([0, 0, 0], [1, 0, 0], [0, 1, 0])
        x_vvts, _, _ = loop_plane_intersect(vvts, xy_plane).T
        x_wp, _, _ = loop_plane_intersect(wp_inner, xy_plane).T
        tk_inner = max(x_wp) - max(x_vvts)

        # The outer thickness is from the TF WP to the closest centrepoint of
        # the PF coils. Note that this will introduce some overlaps, but is
        # also reasonable from a design perspective.
        min_distances = []
        wp_outer = self.tf.geom["WP outer"]
        for coil in self.pf.coils.values():
            if coil.ctype == "PF":
                x, z = coil.x, coil.z
                distances = wp_outer.distance_to([x, z])
                min_distances.append(np.min(distances))

        tk_outer = min(min_distances)
        return tk_inner, tk_outer

    def get_GS_inclusion_zone(self):
        """
        Calculate the range of positions that the GS supports can be
        placed at along the lower half of the TF coil centreline.
        """
        loop = self.tfl["cl_fe"].copy()
        lower = np.where(loop["z"] < 0)
        x_values = loop["x"][lower].copy()

        # Get the x regions where the PF coils are in the way of the GS
        x_excl_values = []
        for pf in self.pf.coils.values():
            if pf.ctype == "PF" and pf.z < 0:
                x_min = pf.x - pf.dx
                x_max = pf.x + pf.dx
                incl = np.where((x_values < x_min) | (x_values > x_max))
                excl = np.where((x_values >= x_min) & (x_values <= x_max))
                x_values = x_values[incl]
                x_excl_values.append(loop["x"][excl])

        return max(np.concatenate(x_excl_values)), max(x_values)

    @staticmethod
    def _make_OIS_nodes(x, z, tk):
        """
        Calculates the node positions of the outer inter-coil structures
        :param x: list(float, float)

        :param z:
        :param tk:
        :return:
        """
        x, z = np.array(x), np.array(z)
        vec = np.array([np.diff(x), np.diff(z)])
        vec = vec / np.linalg.norm(vec)
        nhat = np.array([vec[1], -vec[0]])
        x1 = (x[0] - 0.5 * tk * nhat[0])[0]
        z1 = (z[0] - 0.5 * tk * nhat[1])[0]
        x2 = (x[0] + 0.5 * tk * nhat[0])[0]
        z2 = (z[0] + 0.5 * tk * nhat[1])[0]
        x3 = (x[1] + 0.5 * tk * nhat[0])[0]
        z3 = (z[1] + 0.5 * tk * nhat[1])[0]
        x4 = (x[1] - 0.5 * tk * nhat[0])[0]
        z4 = (z[1] - 0.5 * tk * nhat[1])[0]
        return np.array([x1, x2, x3, x4, x1]), np.array([z1, z2, z3, z4, z1])

    # =========================================================================
    # Coil structure optimisation objective functions
    # =========================================================================
    @staticmethod
    def _min_distance(point, loop):
        """
        Generic minimisation of distance of a point to a loop
        """

        def distance(x, p, ldict):
            """
            Objective function for snapping a point to a loop
            """
            return np.sqrt((p[0] - ldict["x"](x)) ** 2 + (p[1] - ldict["z"](x)) ** 2)

        return minimize_scalar(
            distance, method="bounded", args=(point, loop), bounds=[0, 1]
        ).x

    @staticmethod
    def _min_x(x_value, loop, bounds=None):
        """
        Generic minimisation of radial distance of a point to a loop
        """

        def x_distance(x0, x_val, ldict):
            """
            Objective function for minimising x distance
            """
            return abs(x_val - ldict["x"](x0))

        if bounds is None:
            bounds = [0, 1]

        return minimize_scalar(
            x_distance, method="bounded", args=(x_value, loop), bounds=bounds
        ).x

    @staticmethod
    def _min_intersect(x0, xc, normall, loop, bounds=None):
        """
        Generic minimisation of intersection
        """

        def intersect(x_0, xx_c, nhat, ldict):
            """
            Minimisation objective for support intersection
            """
            x, s = x_0
            xx_tf, zz_tf = ldict["x"](x), ldict["z"](x)
            x_s, z_s = s * nhat + xx_c
            return np.sqrt((xx_tf - x_s) ** 2 + (zz_tf - z_s) ** 2)

        if bounds is None:
            bounds = [[0, 1], [0, 15]]

        return minimize(
            intersect, x0, method="L-BFGS-B", args=(xc, normall, loop), bounds=bounds
        ).x[1]


class OISOptimiser:
    """
    Optimiser for the outer inter-coil structures

    Louis Zani once told me it was bad to make OIS curvy...
    """

    def __init__(self, loop, exclusion_zone):
        self.loop = loop
        self.xlmap = XZLMapper(loop)
        self.exclusion = exclusion_zone

    def f_objective(self, xnorm):
        """
        Objective function for the OIS length maximisation.

        Parameters
        ----------
        xnorm: np.array
            The normalised variable vector

        Returns
        -------
        length: float
            The negative length of the OIS (maximise length).
        """
        x, z = self.xlmap.L_to_xz(xnorm)
        return -np.sqrt((x[0] - x[1]) ** 2 + (z[0] - z[1]) ** 2)

    def f_constraints(self, xnorm):
        """
        Constraint function for the OIS minimisation.

        Parameters
        ----------
        xnorm: np.array
            The normalised variable vector

        Returns
        -------
        constraint: np.array
            The constraint array
        """
        x, z = self.xlmap.L_to_xz(xnorm)
        xzdict = {"x": x, "z": z}

        constraint = self._dot_difference(xzdict, "internal")
        return constraint

    def _dot_difference(self, p, side):
        """
        Geometric constraint function utility.
        """
        xloop, zloop = p["x"], p["z"]  # inside coil loop
        switch = 1 if side == "internal" else -1

        n_xloop, n_zloop = normal(xloop, zloop)
        x_excl, z_excl = self.exclusion["x"], self.exclusion["z"]
        dot = np.zeros(len(x_excl))
        for j, (x, z) in enumerate(zip(x_excl, z_excl)):
            i = np.argmin((x - xloop) ** 2 + (z - zloop) ** 2)
            dl = [xloop[i] - x, zloop[i] - z]
            dn = [n_xloop[i], n_zloop[i]]
            dot[j] = switch * np.dot(dl, dn)
        return dot

    def optimise(self, verbose=False, acc=0.002, **kwargs):
        """
        Shape minimisation method

        Parameters
        ----------
        verbose: bool
            Scipy minimiser verbosity
        acc: float (default = 0.002)
            Optimiser accuracy
        """
        iprint = 1 if verbose else -1

        xnorm = [0, 1]
        bnorm = [[0, 1], [0, 1]]
        xnorm, fx, its, imode, smode = fmin_slsqp(
            self.f_objective,
            xnorm,
            f_ieqcons=self.f_constraints,
            bounds=bnorm,
            acc=acc,
            iprint=iprint,
            full_output=True,
            **kwargs,
        )
        if imode != 0:
            bluemira_warn("Nova::OISOptimiser exit code != 0.")

        return self.xlmap.L_to_xz(xnorm)


class CoilArchitectPlotter(ReactorSystemPlotter):
    """
    The plotter for a Coil Architect.
    """

    def __init__(self):
        super().__init__()
        self._palette_key = "ATEC"
