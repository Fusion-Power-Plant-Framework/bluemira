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
Breeding blanket system
"""
from collections import OrderedDict
from itertools import cycle
from typing import Type

import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon

from bluemira.base.parameter import ParameterFrame
from bluemira.geometry.error import GeometryError
from BLUEPRINT.base.error import SystemsError
from BLUEPRINT.cad.blanketCAD import BlanketCAD, STBlanketCAD
from BLUEPRINT.geometry.boolean import (
    boolean_2d_common,
    boolean_2d_common_loop,
    boolean_2d_difference,
    boolean_2d_difference_loop,
)
from BLUEPRINT.geometry.geombase import Plane
from BLUEPRINT.geometry.geomtools import make_box_xz, qrotate, rainbow_arc
from BLUEPRINT.geometry.loop import Loop, MultiLoop, mirror
from BLUEPRINT.systems.baseclass import ReactorSystem
from BLUEPRINT.systems.mixins import Meshable
from BLUEPRINT.systems.plotting import ReactorSystemPlotter


class BreedingBlanket(Meshable, ReactorSystem):
    """
    Breeding blanket reactor system.
    """

    config: Type[ParameterFrame]
    inputs: dict

    # fmt: off
    default_params = [
        ['n_TF', 'Number of TF coils', 16, 'dimensionless', None, 'Input'],
        ['plasma_type', 'Type of plasma', 'SN', 'dimensionless', None, 'Input'],
        ['A', 'Plasma aspect ratio', 3.1, 'dimensionless', None, 'Input'],
        ['R_0', 'Major radius', 9, 'm', None, 'Input'],
        ['blanket_type', 'Blanket type', 'HCPB', 'dimensionless', None, 'Input'],
        ['g_vv_bb', 'Gap between VV and BB', 0.02, 'm', None, 'Input'],
        ['c_rm', 'Remote maintenance clearance', 0.02, 'm', 'Distance between IVCs', None],
        ["bb_e_mult", "Energy multiplication factor", 1.35, "dimensionless", None, "HCPB classic"],
        ['bb_min_angle', 'Mininum BB module angle', 70, '°', 'Sharpest cut of a module possible', 'Lorenzo Boccaccini said this in a meeting in 2015, Garching, Germany'],
        ['fw_dL_min', 'Minimum FW module length', 0.75, 'm', None, 'Input'],
        ['fw_dL_max', 'Maximum FW module length', 3, 'm', 'Cost+manufacturing implications', 'Input'],
        ['fw_a_max', 'Maximum angle between FW modules', 20, '°', None, 'Input'],
        ['rho', 'Blanket density', 3000, 'kg/m^3', 'Homogenous', None],
        ['tk_bb_ib', 'Inboard blanket thickness', 0.8, 'm', None, 'Input'],
        ['tk_bb_ob', 'Outboard blanket thickness', 1.1, 'm', None, 'Input'],
        ['tk_bb_fw', 'First wall thickness', 0.025, 'm', None, 'Input'],
        ['tk_bb_arm', 'Tungsten armour thickness', 0.002, 'm', None, 'Input'],
        ["tk_r_ib_bz", "Thickness ratio of the inboard blanket breeding zone", 0.309, "dimensionless", None, "HCPB 2015 design description document"],
        ["tk_r_ib_manifold", "Thickness ratio of the inboard blanket manifold", 0.114, "dimensionless", None, "HCPB 2015 design description document"],
        ["tk_r_ib_bss", "Thickness ratio of the inboard blanket back supporting structure", 0.577, "dimensionless", None, "HCPB 2015 design description document"],
        ["tk_r_ob_bz", "Thickness ratio of the outboard blanket breeding zone", 0.431, "dimensionless", None, "HCPB 2015 design description document"],
        ["tk_r_ob_manifold", "Thickness ratio of the outboard blanket manifold", 0.071, "dimensionless", None, "HCPB 2015 design description document"],
        ["tk_r_ob_bss", "Thickness ratio of the outboard blanket back supporting structure", 0.498, "dimensionless", None, "HCPB 2015 design description document"],
    ]
    # fmt: on
    CADConstructor = BlanketCAD

    def __init__(self, config, inputs):
        self.config = config
        self.inputs = inputs
        self._plotter = BreedingBlanketPlotter()

        self._init_params(self.config)

        # Constructors
        self.n_segments = 5 * self.params.n_TF
        self._ib_cut = None
        self._ob_cut = None
        self._anglecut = None
        self.flag_cut_smooth = None
        self.tk_bb_ob_breakdown = None
        self.tk_bb_ib_breakdown = None

        self.read_radial_build()
        self.set_params(self.params.blanket_type)
        self.dummy_data()
        self.rm = {}

    def set_params(self, blanket_type):
        """
        Set parameters in the BreedingBlanket based on the blanket type.
        """
        if blanket_type == "HCPB":
            # fmt: off
            p = [
                ["coolant", "Coolant", "He", "dimensionless", None, None],
                ["mult", "Neutron multiplier", "Be", "dimensionless", None, None],
                ["maxTBR", "Maximum TBR", 1.27, "dimensionless", None, "HCPB classic"],
                ["T_in", "Inlet temperature", 300, "°C", None, "HCPB classic"],
                ["T_out", "Outlet temperature", 500, "°C", None, "HCPB classic"],
                ["P_in", "Inlet pressure", 8, "MPa", None, "HCPB classic"],
                ["dP", "Pressure drop", 0.5, "MPa", None, None],
                ["f_dh", "Decay heat fraction", 0.0175, "dimensionless", None, "PPCS FWBL Helium Cooled Model P PPCS04 D5part1"],
            ]
            # fmt: on
        elif blanket_type == "WCLL":
            # fmt: off
            p = [
                ["coolant", "Coolant", "Water", "dimensionless", None, None],
                ["mult", "Neutron multiplier", "LiPb", "dimensionless", None, None],
                ["maxTBR", "Maximum TBR", 1.22, "dimensionless", None, "WCLL classic"],
                ["T_in", "Inlet temperature", 200, "°C", None, None],
                ["T_out", "Outlet temperature", 300, "°C", None, None],
                ["P_in", "Inlet pressure", 155, "MPa", None, None],
                ["dP", "Pressure drop", 0.5, "MPa", None, None],
                ["f_dh", "Decay heat fraction", 0.0001, "dimensionless", "Insignificant number to preserve plotting", None],
            ]
            # fmt: on
        else:
            raise SystemsError(f"Unknown blanket type '{blanket_type}'.")

        self.add_parameters(p)

    def read_radial_build(self):
        """
        Process the intial radial build.
        """
        self.geom["2D inner"] = self.inputs["blanket_inner"]
        self.geom["FW cut"] = self.geom["2D inner"].copy()

        if self.params.plasma_type == "DN":
            # Treat open MultiLoop
            self.geom["FW cut"] = self.geom["FW cut"].force_connect()
        else:
            self.geom["FW cut"].close()
        # Not cut up like this. Blanket coverage already defined in RB. Tricky
        self.geom["2D outer"] = self.inputs["blanket_outer"]
        self.geom["2D profile"] = self.inputs["blanket"]

        # Get remaining blanket thickness, and divide into regions per thickness
        # fraction
        tk_ob_remaining = (
            self.params.tk_bb_ob - self.params.tk_bb_fw - self.params.tk_bb_arm
        )
        tk_ib_remaining = (
            self.params.tk_bb_ib - self.params.tk_bb_fw - self.params.tk_bb_arm
        )

        self.tk_bb_ob_breakdown = [
            self.params.tk_bb_fw + self.params.tk_bb_arm,
            self.params.tk_r_ob_bz * tk_ob_remaining,
            self.params.tk_r_ob_manifold * tk_ob_remaining,
            self.params.tk_r_ob_bss * tk_ob_remaining,
        ]

        self.tk_bb_ib_breakdown = [
            self.params.tk_bb_fw + self.params.tk_bb_arm,
            self.params.tk_r_ib_bz * tk_ib_remaining,
            self.params.tk_r_ib_manifold * tk_ib_remaining,
            self.params.tk_r_ib_bss * tk_ib_remaining,
        ]
        self.tk_bb_ob_breakdown = np.cumsum(self.tk_bb_ob_breakdown)
        self.tk_bb_ib_breakdown = np.cumsum(self.tk_bb_ib_breakdown)

    @property
    def xz_plot_loop_names(self):
        """
        The x-z loop names to plot.
        """
        return [
            "IB 2D profile fw",
            "IB 2D profile bz",
            "IB 2D profile manifold",
            "IB 2D profile bss",
            "OB 2D profile fw",
            "OB 2D profile bz",
            "OB 2D profile manifold",
            "OB 2D profile bss",
        ]

    def _xy_generator(self):
        """
        Generate names and Loops for plotting.

        Yields
        ------
        str, Loop
            The name and corresponding Loop.
        """
        for name, seg in self.geom["feed 3D CAD"].items():
            for sub, part in zip(["fw", "bz", "manifold", "bss"], seg["plan"]):
                newname = " ".join([name, sub, "X-Y"])
                yield newname, part

    @property
    def xy_plot_loop_names(self):
        """
        The x-y loop names to plot.
        """
        return [name for name, _ in self._xy_generator()]

    def _generate_xy_plot_loops(self):
        def pattern(loop, r_angles):
            """
            Patterns the GeomBase objects based on the number of sectors
            """
            loops = [
                loop.rotate(a, update=False, p1=[0, 0, 0], p2=[0, 0, 1])
                for a in r_angles
            ]
            return MultiLoop(loops, stitch=False)

        angles = np.linspace(0, 360, self.params.n_TF, endpoint=False)
        for name, part in self._xy_generator():
            self.geom[name] = pattern(part, angles)
        return super()._generate_xy_plot_loops()

    def plot_wireframe(self):
        """
        Plot the wireframe geometry for the BreedingBlanket segments.
        """
        fig = plt.figure(figsize=(14, 14))
        ax = fig.gca(projection="3d")
        bb = 4
        ax.set_xlim(-bb, bb)
        ax.set_ylim(-bb, bb)
        ax.set_zlim(-bb, bb)
        # ax.view_init(azim=0, elev=90)
        offset = 9
        c = "grey"
        for key, seg in self.geom["feed 3D CAD"].items():
            ax.plot(
                seg["profile1"]["x"] - offset,
                seg["profile1"]["y"],
                seg["profile1"]["z"],
                color=c,
            )
            ax.plot(
                seg["profile2"]["x"] - offset,
                seg["profile2"]["y"],
                seg["profile2"]["z"],
                color=c,
            )
            for i in np.append(
                np.arange(0, len(seg["profile1"]["x"]) - 2, 10),
                len(seg["profile1"]["x"]) - 2,
            ):
                a = [
                    seg["profile1"]["x"][i] - offset,
                    seg["profile1"]["y"][i],
                    seg["profile1"]["z"][i],
                ]
                b = [
                    seg["profile2"]["x"][i] - offset,
                    seg["profile2"]["y"][i],
                    seg["profile2"]["z"][i],
                ]
                p = list(zip(a, b))
                ax.plot(*p, color=c)
        for seg in self.geom["rm"].keys():
            cog = self.geom["rm"][seg]["Centre of gravity"]
            lift = self.geom["rm"][seg]["Lift point"]
            ax.plot(
                [cog["x"] - offset],
                [cog["y"]],
                [cog["z"]],
                marker="o",
                markersize=15,
                color="k",
            )
            ax.plot(
                [lift["x"] - offset],
                [lift["y"]],
                [lift["z"]],
                marker="o",
                markersize=15,
                color="r",
            )

        return fig

    def split_poloidally(self, rm_cut, angle):
        """
        Poloidal cut of inboard/outboard blanket segments
        """
        if self.flag_cut_smooth is not None:
            # Store smooth profiles
            # Useful for CAD building with mixed splines and polygons
            k = " smooth"
            self.geom["IB 2D profile" + k] = self.geom["IB 2D profile"].copy()
            self.geom["OB 2D profile" + k] = self.geom["OB 2D profile"].copy()
        p = self.geom["2D profile"].copy()
        ib, ob = p.chop_by_line(rm_cut, angle)
        _, ob = ob.chop_by_line([rm_cut[0] + self.params.c_rm, rm_cut[1]], angle)
        self.geom["IB 2D profile"] = ib
        self.geom["OB 2D profile"] = ob
        # Store cut coordinates in 3D
        self._ib_cut = rm_cut[0], -self.params.c_rm / 2, rm_cut[1]
        self._ob_cut = rm_cut[0] + self.params.c_rm, 0, rm_cut[1]
        self._anglecut = angle
        self.flag_cut_smooth = True

    def split_radially(self):
        """
        Radially sub-divides the blankets into different regions. This is for
        more accurate neutronics results, but is also more realistic.
        """
        keys = ["IB 2D profile", "OB 2D profile"]
        cut = self.geom["FW cut"].copy()
        cut.close()

        def sort_by_radius(loop_list, key):
            """
            Sorts a list of loops by radius (min/max based on IB/OB)
            """
            if "IB" in key:
                s = 1
            elif "OB" in key:
                s = -1
            else:
                raise KeyError("No IB/OB profiles found.")

            return sorted(loop_list, key=lambda x: s * x.centroid[0])[0]

        for k, thicknesses in zip(
            keys, [self.tk_bb_ib_breakdown[:-1], self.tk_bb_ob_breakdown[:-1]]
        ):
            loops = []
            clips = []
            for name, tk in zip(["fw", "bz", "manifold"], thicknesses):
                clip = cut.offset(tk)
                part = boolean_2d_common(self.geom[k], clip)[0]
                if len(clips) > 0:
                    part = boolean_2d_difference(part, clips[-1])
                    part = sort_by_radius(part, k)
                self.geom[k + " " + name] = part
                loops.append(part.copy())
                clips.append(clip)

            diff = boolean_2d_difference(self.geom[k], clip)
            if len(diff) == 1:
                # Single back supporting structure piece (normal)
                self.geom[k + " bss"] = diff[0]
                loops.append(diff[0])
            else:
                # Handle disjointed back supporting structure
                self.geom[k + " bss"] = MultiLoop(diff, stitch=False)
                loops.extend(diff)

            self.geom[k + " multi"] = MultiLoop(loops, stitch=False)

    def segment_blanket(self, cuto, omega, segmentation="radial"):
        """
        Cuts blanket up into segments - decided to go with xz profiles rather
        than midplane cross-sections, because the CS varies with angle.
        Instead we can rotate the profiles about machine axis. And chop
        them up if need be later
        """
        # Offset it a little to avoid shapely precision nightmares
        self.split_poloidally(cuto + 0.05, -omega)
        self.split_radially()
        if segmentation == "radial":
            self.build_radial_segments()
        elif segmentation == "parallel":
            self.build_parallel_segments()
        else:
            raise SystemsError(
                f"Unknown blanket segmentation strategy '{segmentation}.'"
            )

    def update_FW_profile(self, fw_profile):
        """
        Updates the BB 2-D profile with the (flat) first wall profile

        Parameters
        ----------
        fw_profile: BLUEPRINT Loop
            The firstwall profile to stitch into the 2D profile
        """
        # Negative imprint for CAD cutting
        self.geom["FW cut"] = fw_profile.copy()
        self.geom["FW cut"].close()
        self.geom["2D profile smooth"] = self.geom["2D profile"].copy()

        profile = self.geom["2D profile"]
        if isinstance(profile, Loop):  # Single null
            new = boolean_2d_difference(profile, self.geom["FW cut"])[0]

        elif isinstance(profile, MultiLoop):  # Double null
            loops = profile.loops
            new_loops = []
            for loop in loops:
                loop = boolean_2d_difference(loop, self.geom["FW cut"])[0]
                new_loops.append(loop)

            new_loops.sort(key=lambda g: np.min(g.x))
            new = MultiLoop(new_loops)
            # Here we already have the IB/OB segmentation
            self.geom["IB 2D profile"] = new_loops[0]
            self.geom["OB 2D profile"] = new_loops[1]

        else:
            raise SystemsError(f"Unrecognised object of type: {type(profile)}.")

        self.geom["2D profile"] = new

    def build_parallel_segments(self, plot=False):
        """
        Work in progress. - tic tock...
        """
        if "feed 3D CAD" in self.geom.keys():
            # return  # Do not overwrite smooth walls
            pass  # Do overwrite smooth walls

        ibs = self.geom["IB 2D profile clip"].copy()
        ibs.translate([0, -self.params.c_rm / 2, 0])
        ibs_fw = self.geom["IB 2D profile fw"].copy()
        ibs_fw.translate([0, -self.params.c_rm / 2, 0])
        obs = self.geom["OB 2D profile clip"].copy()
        obs_fw = self.geom["OB 2D profile fw"].copy()

        beta = 360 / self.params.n_TF
        angle_ib, angle_ob = beta / 2, beta / 3
        betar, angle_ibr, angle_obr, gammar = [
            np.radians(a) for a in [beta, beta / 2, beta / 3, beta / 6]
        ]
        # centre of z-axis about which ib blanket rotation must occur
        c1_left = np.array(
            [0.5 * self.params.c_rm / np.tan(betar / 4), -0.5 * self.params.c_rm, 0]
        )
        c11_left = np.array(
            [0.5 * self.params.c_rm / np.tan(betar / 4), -0.5 * self.params.c_rm, 1]
        )
        c1_right = np.array(
            [0.5 * self.params.c_rm / np.tan(betar / 4), 0.5 * self.params.c_rm, 0]
        )
        c11_right = np.array(
            [0.5 * self.params.c_rm / np.tan(betar / 4), 0.5 * self.params.c_rm, 1]
        )
        c2 = [0.5 * self.params.c_rm / np.sin(gammar), 0, 0]
        c22 = [0.5 * self.params.c_rm / np.sin(gammar), 0, 1]
        c2_left, c22_left = qrotate(
            [c2, c22], theta=-angle_obr, p1=[0, 0, 0], p2=[0, 0, 1]
        )
        c2_right, c22_right = qrotate(
            [c2, c22], theta=angle_obr, p1=[0, 0, 0], p2=[0, 0, 1]
        )

        # Book 11 p 37
        def get_rotated_cut(x3):
            p3 = [
                x3,
                x3 * np.tan(angle_ibr) - 0.5 * self.params.c_rm / np.cos(angle_ibr),
            ]
            p3r_r = qrotate(p3, theta=angle_ibr, p1=c1_left, p2=c11_left)
            return p3r_r

        # Rotated cuts (cheat by skewing cut point)
        p = self.geom["2D profile"].copy()
        p3r = get_rotated_cut(self._ib_cut[0])
        ibs_r, obr = p.chop_by_line(p3r, self._anglecut)
        p3r = get_rotated_cut(self._ob_cut[0])
        _, obs_r = obs.chop_by_line(p3r, self._anglecut)
        # Rotated profiles
        ibs_rr = ibs_r.rotate(angle_ib, p1=c1_right, p2=c11_right, update=False)
        ibs_rl = ibs_rr.rotate(-2 * angle_ib, p1=c1_right, p2=c11_right, update=False)
        obs_rr = obs_r.rotate(angle_ob, p1=c2_right, p2=c22_right, update=False)
        obs_rl = obs_rr.rotate(-2 * angle_ob, p1=c2_right, p2=c22_right, update=False)
        libs = {
            "body": [ibs, ibs_rl],
            "fw": ibs_fw,
            "path": {"angle": -angle_ib, "rotation axis": [c1_left, c11_left]},
        }
        ribs = {
            "body": [ibs, ibs_rr],
            "fw": ibs_fw,
            "path": {"angle": angle_ib, "rotation axis": [c1_right, c11_right]},
        }
        lobs = {
            "body": [obs, obs_rl],
            "fw": obs_fw,
            "path": {"angle": -angle_ob, "rotation axis": [c2_left, c22_left]},
        }
        robs = {
            "body": [obs, obs_rr],
            "fw": obs_fw,
            "path": {"angle": angle_ob, "rotation axis": [c2_right, c22_right]},
        }
        cobs = {
            "body": [obs, obs_rl],
            "fw": obs_fw,
            "path": {"angle": -angle_ob, "rotation axis": [c2_left, c22_left]},
        }

        self.geom["feed 3D CAD"] = OrderedDict()
        for key, data in zip(
            ["LIBS", "RIBS", "LOBS", "COBS", "ROBS"], [libs, ribs, lobs, cobs, robs]
        ):
            self.geom["feed 3D CAD"][key] = data
        if plot:
            self.plot_wireframe()

    def build_radial_segments(self, plot=False):
        """
        Prepares profiles and sweep angles for CAD creation. Blanket division
        is classical 5-segment cut, with equispaced radial divisions
        - LOBS: left outboard blanket segment
        - COBS: central outboard blanket segment
        - ROBS: right outboard blanket segment
        - LIBS: left inboard blanket segment
        - RIBS: right inboard blanket segment
        """
        if "feed 3D CAD" in self.geom.keys():
            # return  # Do not overwrite smooth walls
            pass  # Do overwrite smooth walls

        ibs = self.geom["IB 2D profile multi"].copy()
        ibs.translate([0, -self.params.c_rm / 2, 0])
        obs = self.geom["OB 2D profile multi"].copy()
        beta = 360 / self.params.n_TF
        angle_ib, angle_ob = beta / 2, beta / 3
        betar, angle_ibr, angle_obr, gammar = [
            np.radians(a) for a in [beta, beta / 2, beta / 3, beta / 6]
        ]
        # centre of z-axis about which ib blanket rotation must occur
        c1_left = np.array(
            [0.5 * self.params.c_rm / np.tan(betar / 4), -0.5 * self.params.c_rm, 0]
        )
        c11_left = np.array(
            [0.5 * self.params.c_rm / np.tan(betar / 4), -0.5 * self.params.c_rm, 1]
        )
        c1_right = np.array(
            [0.5 * self.params.c_rm / np.tan(betar / 4), 0.5 * self.params.c_rm, 0]
        )
        c11_right = np.array(
            [0.5 * self.params.c_rm / np.tan(betar / 4), 0.5 * self.params.c_rm, 1]
        )

        libs = {
            "parts": ibs.loops,  # for list behaviour uniformity later
            "path": {"angle": -angle_ib, "rotation axis": [c1_left, c11_left]},
        }
        ribs = {
            "parts": [Loop(**mirror(part)) for part in ibs.loops],
            "path": {"angle": angle_ib, "rotation axis": [c1_right, c11_right]},
        }

        c2 = [0.5 * self.params.c_rm / np.sin(gammar), 0, 0]
        c22 = [0.5 * self.params.c_rm / np.sin(gammar), 0, 1]
        c2_left, c22_left = qrotate(
            [c2, c22], theta=-angle_obr, p1=[0, 0, 0], p2=[0, 0, 1]
        )
        c2_right, c22_right = qrotate(
            [c2, c22], theta=angle_obr, p1=[0, 0, 0], p2=[0, 0, 1]
        )
        cobs = {
            "parts": [
                Loop(*qrotate(part, theta=-gammar, p1=c2, p2=c22).T) for part in obs
            ],
            "path": {"angle": angle_ob, "rotation axis": [c2, c22]},
        }

        lobs = {
            "parts": [
                Loop(*qrotate(part, theta=-angle_obr, p1=[0, 0, 0], p2=[0, 0, 1]).T)
                for part in cobs["parts"]
            ],
            "path": {"angle": angle_ob, "rotation axis": [c2_left, c22_left]},
        }
        robs = {
            "parts": [
                Loop(*qrotate(part, theta=angle_obr, p1=[0, 0, 0], p2=[0, 0, 1]).T)
                for part in cobs["parts"]
            ],
            "path": {"angle": angle_ob, "rotation axis": [c2_right, c22_right]},
        }

        self.geom["feed 3D CAD"] = OrderedDict()
        for key, data in zip(
            ["LIBS", "RIBS", "LOBS", "COBS", "ROBS"], [libs, ribs, lobs, cobs, robs]
        ):
            self.geom["feed 3D CAD"][key] = data

        def make_plan(segment):
            midplane = Plane([0, 0, 0], [1, 0, 0], [0, 1, 0])
            h = segment["path"]["rotation axis"][0]
            h = [h[0], h[1]]  # x-y point
            angle = segment["path"]["angle"]
            plans = []
            for part in segment["parts"]:
                inter = part.section(midplane)
                if inter is not None:
                    x, y = rainbow_arc(inter[0][:2], inter[1][:2], h, angle=angle)
                    loop = Loop(x=x, y=y)
                    plans.append(loop)
            segment["plan"] = plans

        for key in ["LIBS", "RIBS", "LOBS", "COBS", "ROBS"]:
            make_plan(self.geom["feed 3D CAD"][key])
        if plot:
            self.plot_wireframe()
        return

    def generate_RM_data(self, up_inner, plot=False):
        """
        Get RM data from the CAD.
        """
        # TODO: Confirm radius of gyration calc in spreadsheet...
        self.geom["rm"] = {"LIBS": {}, "RIBS": {}, "LOBS": {}, "COBS": {}, "ROBS": {}}

        cad = self.geom["feed 3D CAD"]
        up = up_inner.as_shpoly()
        if plot:
            f, ax = plt.subplots()
            up_inner.plot(ax=ax, fill=False, color="k")
        points = {}
        for seg in self.geom["feed 3D CAD"]:
            self.geom["rm"][seg]["Density"] = self.rho
            self.geom["rm"][seg]["Length"] = max(cad[seg]["profile1"]["z"]) - min(
                cad[seg]["profile1"]["z"]
            )
            try:
                visible = Polygon(up.intersection(cad[seg]["plan"].as_shpoly()))
            except NotImplementedError:
                raise ValueError(
                    "Ton UpperPort est tellement petit qu'il n'y "
                    "a même pas d'intersection planaire avec les "
                    "inboard BB."
                )

            self.geom["rm"][seg]["Visible CSA"] = visible.area
            self.geom["rm"][seg]["Lift point"] = {
                "x": visible.centroid.x,
                "y": visible.centroid.y,
                "z": max(cad[seg]["profile1"]["z"]),
            }
            points[seg] = list(self.geom["rm"][seg]["Lift point"].values())
            if plot:
                ax.plot(*visible.exterior.xy)
                ax.plot(*visible.centroid.xy, marker="o")
        cad = self.build_CAD()
        props = cad.get_properties(points)
        for seg in self.geom["feed 3D CAD"]:
            self.geom["rm"][seg]["Volume"] = props[seg]["Volume"]
            self.geom["rm"][seg]["Centre of gravity"] = props[seg]["CoG"]
            self.geom["rm"][seg]["Radius of gyration"] = props[seg]["Rg"]
            if plot:
                ax.plot(props[seg]["CoG"]["x"], props[seg]["CoG"]["y"], marker="*")

    def dummy_data(self):
        """
        Deprecated, but useful
        """
        # TODO: remove ? (called in __init__ and by smoke test)
        rm_libs = {
            "Centre of gravity": {"x": 6.293, "y": 0.553, "z": 1.383},
            "Lift point": {"x": 8.03, "y": 0.224},
            "Length": 12.166,
            "Volume": 12.135,
            "Density": 3000,
            "Radius of gyration": 0.086603,
            "Visible CSA": 0.847,
        }
        rm_lobs = {
            "Centre of gravity": {"x": 11.590, "y": 1.3, "z": -0.262},
            "Lift point": {"x": 11.695, "y": 0.832},
            "Length": 13.28,
            "Volume": 27.097,
            "Density": 3000,
            "Radius of gyration": 0.101036,
            "Visible CSA": 1.629,
        }
        rm_cobs = {
            "Centre of gravity": {"x": 11.464, "y": 0, "z": -0.376},
            "Lift point": {"x": 11.035, "y": 0},
            "Length": 13.3,
            "Volume": 21.18,
            "Density": 3000,
            "Radius of gyration": 0.101036,
            "Visible CSA": 4.601,
        }
        rm_ribs = rm_libs
        rm_robs = rm_lobs
        self.geom["feed RM"] = {
            "LIBS": rm_libs,
            "RIBS": rm_ribs,
            "LOBS": rm_lobs,
            "COBS": rm_cobs,
            "ROBS": rm_robs,
        }


class BreedingBlanketPlotter(ReactorSystemPlotter):
    """
    The plotter for a Breeding Blanket.
    """

    def __init__(self):
        super().__init__()
        self._palette_key = "BB"

    def plot_xy(self, plot_objects, ax=None, **kwargs):
        """
        Plot the BreedingBlanket in the x-y plane.
        """
        self._apply_default_styling(kwargs)
        kwargs["facecolor"] = cycle(kwargs["facecolor"])
        super().plot_xy(plot_objects, ax=ax, **kwargs)


# TODO: Inherit from Breeding Blanket
class STBreedingBlanket(Meshable, ReactorSystem):
    """
    Breeding blanket reactor system.
    """

    config: Type[ParameterFrame]
    inputs: dict

    # fmt: off
    default_params = [
        ['n_TF', 'Number of TF coils', 16, 'dimensionless', None, 'Input'],
        ['blanket_type', 'Blanket type', 'banana', 'dimensionless', None, 'Input'],
        ['g_bb_fw', 'Separation between the first wall and the breeding blanket', 0.05, 'm', None, 'Input'],
        ['tk_bb_bz', 'Breeding zone thickness', 1.0, 'm', None, 'Input'],
        ['tk_bb_man', 'Breeding blanket manifold thickness', 0.2, 'm', None, 'Input'],
    ]
    # fmt: on
    CADConstructor = STBlanketCAD

    def __init__(self, config, inputs):
        self.config = config
        self.inputs = inputs
        self.params = ParameterFrame(self.default_params.to_records())
        self._init_params(self.config)
        self.geom = self.inputs
        self._plotter = BreedingBlanketPlotter()
        self.build()

    @property
    def xz_plot_loop_names(self):
        """
        Selection of the loops to be plotted with plot_xz()

        Returns
        -------
        List:
            list of the selected loop names
        """
        part_list = ["OB 2D profile bz"]
        if self.params.blanket_type == "banana":
            part_list.append("OB 2D profile manifold")
        elif self.params.blanket_type == "immersion":
            part_list.append("OB 2D profile manifold upper")
            part_list.append("OB 2D profile manifold lower")

        return part_list

    def __build_immersion_bz(self, add_sep=0.0):
        """
        Return a loop that fills the space between the space between the first wall
        and the vacuum vessel.

        The input loops are taken from input dictionary.
        The return loop can be used as an immersion blanket breeding zone
        or as a starting shape for the banana blanket.

        Parameters
        ----------
        add_sep : float
            Optional additional thickness, used to check horizontal bounds of
            resulting loop will remain inside the vacuum vessel
        """
        # Fetch the vacuum vessel inner profile from inputs
        vv_inner = self.inputs["vv_inner"]

        # Fetch profile of outboard first wall wall from inputs
        fw_outboard = self.inputs["fw_outboard"]

        # Max x of vacuum vessel
        x_max_vv = np.max(vv_inner.x)

        # Limits of outboard first wall
        x_max_fw = np.max(fw_outboard.x)
        x_min_fw = np.min(fw_outboard.x)
        z_min_fw = np.min(fw_outboard.z)
        z_max_fw = np.max(fw_outboard.z)

        # Fetch the gap to the first wall
        fw_sep = self.params.g_bb_fw

        # Check we won't go outside the vacuum vessel
        x_sum = x_max_fw + fw_sep + add_sep
        if x_sum > x_max_vv:
            raise GeometryError("Breeding blanket params are too large.")

        # Create outboard box filling horizonal space between fw and vv
        ob_box = make_box_xz(x_min_fw, x_max_vv, z_min_fw, z_max_fw)

        # Cut away the firstwall from the outboard box
        ob_parts = boolean_2d_difference(ob_box, fw_outboard)

        # Select outer cut
        ob_cut = None
        for part in ob_parts:
            if np.min(part.x) > x_min_fw:
                ob_cut = part
                break

        if not ob_cut:
            raise SystemsError("Failed to subtract FW outboard profile")

        # Subtract a separation to the first wall
        ob_shift_sep = ob_cut.translate([fw_sep, 0, 0], update=False)
        ob_cut = boolean_2d_common_loop(ob_cut, ob_shift_sep)

        # Subtract the vacuum vessel
        ob_cut = boolean_2d_common_loop(ob_cut, vv_inner)
        return ob_cut

    def __build_immersion_man(self, bz):
        """
        Build the manifold for the immersion blanket.
        """
        # Fetch the limits of the breeding zone
        bz_x_max = np.max(bz.x)
        bz_x_min = np.min(bz.x)
        bz_z_max = np.max(bz.z)
        bz_z_min = np.min(bz.z)
        bz_box = make_box_xz(bz_x_min, bz_x_max, bz_z_min, bz_z_max)

        # Fetch the vacuum vessel inner profile from inputs
        vv_inner = self.inputs["vv_inner"]

        # Chop away the middle section of the vacuum vessel
        vv_parts = boolean_2d_difference(vv_inner, bz_box)

        # Get outermost x coord of our chopped vessel
        x_max = 0
        for part in vv_parts:
            x_max_part = np.max(part.x)
            x_max = max(x_max, x_max_part)

        # Fetch manifold thickness and use to define min x
        man_thickness = self.params.tk_bb_man
        x_min = x_max - man_thickness

        # Find vertical limits of vessel
        vv_z_max = np.max(vv_inner.z)
        vv_z_min = np.min(vv_inner.z)

        # Make a rectangle from bounds
        man = make_box_xz(x_min, x_max, vv_z_min, vv_z_max)

        # Cut away everything outside the vacuum vessel
        man = boolean_2d_common_loop(man, vv_inner)

        # Cut away the breeding zone
        manifold_parts = boolean_2d_difference(man, bz)

        # Identify upper and lower parts
        z_min = vv_z_max
        z_max = vv_z_min
        man_upper = None
        man_lower = None
        for part in manifold_parts:
            part_z_max = np.max(part.z)
            part_z_min = np.min(part.z)
            if part_z_max > z_max:
                z_max = part_z_max
                man_upper = part
            if part_z_min < z_min:
                z_min = part_z_min
                man_lower = part

        return man_upper, man_lower

    def __build_immersion_blanket(self):
        """
        Build imersion blanket.

        An Immersion blanket will naturally try to use as much space as
        possible between the VV and the FW. There are many configurations
        that can be thought of, it's just a big tank of FLiBe, so could be
        segmented for easy removal, could be one big tank. Will need some
        form of mechanical support
        """
        # Fill the outboard space between first wall and vacuum vessel
        # to make the breeding zone
        bz = self.__build_immersion_bz()
        self.geom["OB 2D profile bz"] = bz

        # Make a manifold from the vacuum vessel and breeding zone
        man_upper, man_lower = self.__build_immersion_man(bz)
        self.geom["OB 2D profile manifold upper"] = man_upper
        self.geom["OB 2D profile manifold lower"] = man_lower

    def __build_banana_blanket(self):
        """A banana blanket, as the name suggests is somewhat banana-like
        in shape. The weight or cross section of the blanket will likely be
        the driving factors, either too heavy or can't fit through whatever
        ports are needed. There will need to be piping for coolant there will
        also be a need for some patterned support structure.
        """
        # Fetch thicknesses
        bz_thickness = self.params.tk_bb_bz
        man_thickness = self.params.tk_bb_man
        add_sep = bz_thickness + man_thickness

        # Fill the outboard space between first wall and vacuum vessel
        ob_cut = self.__build_immersion_bz(add_sep)

        # Define the outer edge of the breeding zone with a thickness
        bz_end = ob_cut.translate([bz_thickness, 0, 0], update=False)
        # Cut away outer side
        bz = boolean_2d_difference_loop(ob_cut, bz_end)

        # Define outer edge of the manifold with a thickness
        man_end = bz_end.translate([man_thickness, 0, 0], update=False)
        # Cut away inner side by cutting away breeding zone
        man_start = boolean_2d_difference_loop(ob_cut, bz)
        # Cut away outer side
        man = boolean_2d_difference_loop(man_start, man_end)

        # Save the breeding zone and manifold
        self.geom["OB 2D profile bz"] = bz
        self.geom["OB 2D profile manifold"] = man

    def build(self):
        """
        Build the breeding blanket profile from the first wall profile.
        """
        if self.params.blanket_type == "banana":
            self.__build_banana_blanket()
        elif self.params.blanket_type == "immersion":
            self.__build_immersion_blanket()
        else:
            raise SystemsError(f"Unknown blanket type '{self.params.blanket_type}'. ")
