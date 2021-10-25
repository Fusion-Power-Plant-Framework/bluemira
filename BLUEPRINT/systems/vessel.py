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
Reactor vacuum vessel system
"""
from itertools import cycle
import numpy as np
from typing import Type

from bluemira.base.parameter import ParameterFrame

from BLUEPRINT.systems.baseclass import ReactorSystem
from BLUEPRINT.base.error import GeometryError
from BLUEPRINT.cad.vesselCAD import VesselCAD, SegmentedVesselCAD
from BLUEPRINT.geometry.boolean import (
    boolean_2d_difference_loop,
    boolean_2d_difference,
    boolean_2d_union,
    simplify_loop,
    clean_loop,
)
from BLUEPRINT.geometry.loop import Loop, MultiLoop, make_ring
from BLUEPRINT.geometry.shell import Shell
from BLUEPRINT.geometry.geombase import Plane
from BLUEPRINT.geometry.geomtools import loop_plane_intersect, make_box_xz
from BLUEPRINT.systems.mixins import Meshable, UpperPort
from BLUEPRINT.systems.plotting import ReactorSystemPlotter


class VacuumVessel(Meshable, ReactorSystem):
    """
    Vacuum vessel reactor system
    """

    config: Type[ParameterFrame]
    inputs: dict

    # fmt: off
    default_params = [
        ['cl_ts', 'Clearance to TS', 0.05, 'm', None, 'Input'],
        ['n_TF', 'Number of TF coils', 16, 'N/A', None, 'Input'],
        ['LPangle', 'Lower port inclination angle', -25, '°', None, 'Input'],
        ['g_cr_vv', 'Gap between Cryostat and VV ports', 0.2, 'm', None, 'Input'],
        ['vv_dtk', 'VV double-walled thickness', 0.2, 'm', None, 'Input'],
        ['vv_stk', 'VV single-walled thickness', 0.06, 'm', None, 'Input'],
        ['tk_rib', 'VV inter-shell rib thickness', 0.04, 'm', None, 'Input']
    ]
    # fmt: on
    CADConstructor = VesselCAD

    def __init__(self, config, inputs):
        self.config = config
        self.inputs = inputs
        self._plotter = VacuumVesselPlotter()

        self._init_params(self.config)
        self.geom["2D profile"] = self.inputs["vessel_shell"]

        self.up_shift = False

    def build_shells(self):
        """
        Build the Vacuum Vessel shells
        """
        inner = self.inputs["vessel_shell"].inner
        inner2 = inner.offset(self.params.vv_stk)
        outer = self.inputs["vessel_shell"].outer
        outer2 = outer.offset(-self.params.vv_stk)
        self.geom["Inner shell"] = Shell(inner, inner2)
        self.geom["Outer shell"] = Shell(outer2, outer)

    @property
    def xz_plot_loop_names(self):
        """
        The x-z loop names to plot.
        """
        names = ["2D profile", "Full 2D profile", "Duct xz loop2"]
        names += [
            self.geom[loop]
            for loop in ["Inner shell", "Outer shell"]
            if loop in self.geom
        ]
        return names

    @property
    def xy_plot_loop_names(self):
        """
        The x-y loop names to plot.
        """
        return ["Upper port", "Inner X-Y", "Outer X-Y"]

    def _build_upper_port(self, vvup):
        """
        Builds the vertical upper VV port
        """
        beta = np.pi / self.params.n_TF
        vvup = vvup.offset(-self.params.cl_ts)
        # Need to make upper port double-walled on some sides... great.
        up = Shell.from_offset(vvup, -self.params.vv_dtk)
        # cos(beta) is to account for the radius rotated (circle/square)
        x_in_min = min(self.geom["2D profile"].inner["x"])
        x_in_max = max(self.geom["2D profile"].inner["x"]) * np.cos(beta)
        x_out_min = min(self.geom["2D profile"].outer["x"])
        x_out_max = max(self.geom["2D profile"].outer["x"]) * np.cos(beta)
        up = UpperPort(
            up, x_in_min, x_in_max, x_out_min, x_out_max, l_min=self.params.vv_stk
        )
        self.geom["Upper port"] = up

    def _adjust_up_port(self, cr_interface):
        z = max(cr_interface["CRplates"]["z"]) - cr_interface["tk"] - self.params.g_cr_vv
        dz = z - self.geom["Upper port"].inner["z"][0]
        self.geom["Upper port"].translate([0, 0, dz])

    def _build_eq_port(self, vveq):
        """
        Builds a horizontal outer equatorial port
        """
        eq = vveq.offset(-self.params.cl_ts)
        eq = Shell.from_offset(eq, -self.params.vv_stk)
        self.geom["Equatorial port"] = eq

    def _adjust_eq_port(self, cr_interface):
        x = max(cr_interface["CRplates"]["x"]) - cr_interface["tk"] - self.params.g_cr_vv
        dx = x - self.geom["Equatorial port"].inner["x"][0]
        self.geom["Equatorial port"].translate([dx, 0, 0])

    def _build_lower_port(self, lower, lower_duct):
        """
        Builds the lower port and duct
        """
        lp = lower.offset(-self.params.cl_ts)
        lp = Shell.from_offset(lp, -self.params.vv_dtk)
        self.geom["Lower port"] = lp
        lp_duct = lower_duct.offset(-self.params.cl_ts)
        lp_duct = Shell.from_offset(lp_duct, -self.params.vv_dtk)
        self.geom["Lower duct"] = lp_duct

    def _adjust_lower_port(self, cr_interface):
        x = max(cr_interface["CRplates"]["x"]) - cr_interface["tk"] - self.params.g_cr_vv
        self.geom["LP path"]["x"][-1] = x
        dx = x - self.geom["Lower duct"].inner["x"][0]
        self.geom["Lower duct"].translate([dx, 0, 0])

    def build_ports(self, ts_interface):
        """
        Builds the vacuum vessel ports inside the thermal shield ports

        Parameters
        ----------
        ts_interface: dict
            upper: TS upper port Loop object
            eq: TS equatorial port Loop object
            lower: TS lower port Loop object
            lower_duct: TS lower port duct Loop object
            LP path: TS lower port path
        """
        self._build_upper_port(ts_interface["upper"])
        self._build_eq_port(ts_interface["eq"])
        self._build_lower_port(ts_interface["lower"], ts_interface["lower_duct"])
        self.geom["LP path"] = ts_interface["LP_path"]

    def adjust_ports(self, inputs):
        """
        Adjust the vacuum vessel ports.
        """
        self._adjust_eq_port(inputs)
        self._adjust_up_port(inputs)
        self._adjust_lower_port(inputs)

    def _generate_xy_plot_loops(self):
        inner = make_ring(
            min(self.geom["2D profile"].inner.x), min(self.geom["2D profile"].outer.x)
        )
        self.geom["Inner X-Y"] = inner
        outer = make_ring(
            max(self.geom["2D profile"].inner.x), max(self.geom["2D profile"].outer.x)
        )
        self.geom["Outer X-Y"] = outer
        return super()._generate_xy_plot_loops()

    def _generate_xz_plot_loops(self):
        # This the more verbose boolean approach... keeping boolean ops simple
        body = self.geom["2D profile"]
        upvec = [0, 0, -10]
        up = self._make_xz_shell(
            self.geom["Upper port"].outer, upvec, self.params.vv_dtk
        )
        eqvec = [-10, 0, 0]
        eq = self._make_xz_shell(
            self.geom["Equatorial port"].outer, eqvec, self.params.vv_stk
        )
        x_a = self.geom["Lower port"].inner.x[0]
        a = np.argmin(body.inner.z)
        x_b = body.inner.x[a]
        dx = x_b - x_a
        dz = dx * np.tan(np.deg2rad(self.params.LPangle))
        sdt = self.params.vv_dtk * np.cos(np.deg2rad(-self.params.LPangle))
        lpvec = [dx - sdt * 2, 0, dz]
        lp = self._make_xz_shell(self.geom["Lower port"].outer, lpvec, sdt)
        aub = boolean_2d_union(body.outer, up.outer)[0]
        aub = boolean_2d_union(aub, eq.outer)[0]
        aub = boolean_2d_union(aub, lp.outer)[0]
        anb = boolean_2d_union(body.inner, up.inner)[0]
        anb = boolean_2d_union(anb, eq.inner)[0]
        anb = boolean_2d_union(anb, lp.inner)[0]
        aub = Shell(*boolean_2d_difference(aub, anb)[::-1])

        up = self.geom["Upper port"].copy()
        upc = up.inner.translate([0, 0, up.thickness * 2], update=False)
        upc = self._make_xz_shell(upc, upvec, 0)
        eq = self.geom["Equatorial port"].copy()
        eqc = eq.inner.translate([self.params.vv_stk * 2, 0, 0], update=False)
        eqc = self._make_xz_shell(eqc, eqvec, 0)
        lp = self.geom["Lower port"].copy()
        lpvec = np.array(lpvec)
        lpc = lp.inner.translate(-2 * sdt * lpvec / np.linalg.norm(lpvec), update=False)
        lpc = self._make_xz_shell(lpc, lpvec, 0)
        outer = aub.outer.copy()
        for cut in [upc, eqc, lpc]:
            outer = boolean_2d_difference(outer, cut)
        final = boolean_2d_difference(outer, aub.inner)
        self.geom["Full 2D profile"] = MultiLoop(final, stitch=False)
        self._generate_duct_loop()
        return super()._generate_xz_plot_loops()

    @staticmethod
    def _make_xz_shell(loop, vector, thickness):
        """
        Isso aqui e util para fazer um Shell pra as portas

        Parameters
        ----------
        loop: Loop
            The X-Y loop from which to make an X-Z shell
        vector: tuple(3)
            The extrusion vector of the xy_shell

        Returns
        -------
        xz_shell: Shell
            The X-Z shell resulting from the extruded xy_shell
        """
        outer = loop
        outer2 = outer.translate(vector, update=False)
        i = [
            np.argmin(outer.x + outer.z),
            np.argmax(outer.x + outer.z),
            np.argmax(outer2.x + outer2.z),
            np.argmin(outer2.x + outer2.z),
        ]
        i = [int(idx) for idx in i]
        x, z = [], []
        for idx in i[:2]:
            x.append(outer.x[idx])
            z.append(outer.z[idx])
        for idx in i[2:]:
            x.append(outer2.x[idx])
            z.append(outer2.z[idx])
        x.append(x[0])
        z.append(z[0])
        p = Loop(x=x, z=z)
        if thickness != 0:
            return Shell.from_offset(p, -thickness)
        else:
            return p

    def _generate_duct_loop(self):
        d = self.geom["Lower duct"]
        p = self.geom["LP path"]
        t = self.params.vv_dtk
        alpha = self.params.LPangle
        x0, z0 = self.geom["Lower port"].outer["x"][0], d.outer.z[2]
        x1 = p.x[-1]
        z2 = d.outer.z[0]
        x = [x0 - t, x1, x1, x0, x0, x1, x1, x0 - t, x0 - t]
        z = [z0, z0, z0 + t, z0 + t, z2 - t, z2 - t, z2, z2, z0]
        self.geom["Duct xz loop"] = Loop(x=x, z=z)
        z3 = self.geom["Lower port"].outer["z"][2]
        z4 = z3 - t * np.tan(np.radians(alpha))
        z6 = self.geom["Lower port"].outer["z"][0]
        z5 = z6 - t * np.tan(np.radians(alpha))
        # Lower
        x = [x0 - t, x1, x1, x0, x0, x0 - t, x0 - t]
        z = [z0, z0, z0 + t, z0 + t, z3, z4, z0]
        loop = Loop(x=x, z=z)
        # Upper
        x = [x0 - t, x0, x0, x1, x1, x0 - t, x0 - t]
        z = [z5, z6, z2 - t, z2 - t, z2, z2, z5]
        u_loop = Loop(x=x, z=z)
        self.geom["Duct xz loop2"] = MultiLoop([loop, u_loop])


class SegmentedVaccumVessel(Meshable, ReactorSystem):
    """
    Segmented Vaccum vessel (VV) class

    This class generates a vaccum system made of two sections:
    - The inboard: parametrized by its own TS to VV gap and its thickness.
    - The outboard: parametrized by its own TS to VV gap and its thickness.
    The two sections are separated at a radius provided by the user

    TODO: Understand if we need single walled or double null of whatever
    TODO: The cryostat vaccum vessel is not yet supported.
    TODO: Maintenance port must be added (potenitally via inheritance)

    Arguments
    ---------
    config: Dict[Loop]
        BLUEPRINT objects inpus:
        key : "TS inner loop""
    inputs: Parameters (BLUEPRINT class)

    Parameters
    ----------
    params.tk_vv_in: float
        Inboard thermal shield thickness.
    params.tk_vv_out: float
        Outboard thermal shield thickness.
    params.g_ib_vv_ts: float
        Inboard TF to TS gap
        TODO: Set an inboard TS TF gap in PROCESS and connect it to BP
    params.g_ob_vv_ts: float
        Inboard TF to TS gap
        TODO: Set an outboard TS TF gap in PROCESS and connect it to BP
    params.tk_vv_in: float
        Inboard thermal shield thickness
    params.tk_vv_out: float
        Outboard thermal shield thickness
    params.r_vv_joint: float
        Radial position of the ouboard to inboard vaccum vessel joint
    ts_inner_loop: Loop
        Inner thermal shield loop
    geom: dict[Loop, Shell]
        geom["Inboard profile"]: Loop
            Loop defining the 2D inboard vaccum vessel
        geom["Outboard profile"]: Loop
            Loop defining the 2D outboard vaccum vessel
    """

    config: Type[ParameterFrame]
    inputs: dict

    # fmt: off
    default_params = [
        ['n_TF', 'Number of TF coils', 16, 'N/A', None, 'Input'],
        ['tk_vv_in', 'Inboard VV thickness', 0.30, 'm', None, 'Input'],
        ['tk_vv_out', 'Outboard VV thickness', 0.60, 'm', None, 'Input'],
        ['g_ib_vv_ts', 'Inboard gap between VV and TS', 0.05, 'm', None, 'Input'],
        ['g_ob_vv_ts', 'Outboard gap between VV and TS', 0.05, 'm', None, 'Input'],
        ['r_vv_joint', 'Radius of inboard/outboard VV joint', 2. , 'm', None, 'Input'],
    ]
    # fmt: on
    CADConstructor = SegmentedVesselCAD

    def __init__(self, config, inputs):
        self.config = config
        self.inputs = inputs
        self._plotter = VacuumVesselPlotter()

        self._init_params(self.config)

        self.ts_inboard_loop = self.inputs["TS inboard loop"]
        self.ts_outboard_loop = self.inputs["TS outboard loop"]
        self.build_vv()

    def build_vv(self):
        """
        Builds the inboard and outboard vaccum vessel shell.
        """
        # Make the inboard vaccum vessel loop
        self.geom["Inboard profile"] = self.build_vv_section(
            side="Inboard",
            g_vv_ts=self.params.g_ib_vv_ts,
            tk_vv=self.params.tk_vv_in,
            ts_loop=self.ts_inboard_loop,
        )

        # Make the outboard vaccum vessel loop
        self.geom["Outboard profile"] = self.build_vv_section(
            side="Outboard",
            g_vv_ts=self.params.g_ob_vv_ts,
            tk_vv=self.params.tk_vv_out,
            ts_loop=self.ts_outboard_loop,
        )

        # Generate the union between the inboard and the outboard VV sections
        self.merge_vv()

    def build_vv_section(self, side, g_vv_ts, tk_vv, ts_loop) -> Loop:
        """
        Builds the vaccum vessel inner or outer section.

        Arguments
        ---------
        side: str
            "Inboard": inboard vaccum vessel section
            "Outboard": outboard vaccum vessel section
        g_vv_ts: float
            Thickness of the gap between the vaccum vessel and the thermal
            shield in the considered section
        tk_vv: float
            Thickness of the vaccum vessel in the considered section

        Returns
        -------
        section_loop: Loop
            The inboard or outboard thermal shield 2D cross section loop
        """
        if side not in ["Inboard", "Outboard"]:
            raise NotImplementedError(
                f"{side} side option not implemented"
                "Only 'Inboard' and 'Outboard' implemented"
            )

        # Building the vacuum vessel loop
        gap_loop = self.try_offset(loop=ts_loop, offset=g_vv_ts, side=side)
        vv_loop = self.try_offset(loop=ts_loop, offset=tk_vv + g_vv_ts, side=side)
        vv_loop = boolean_2d_difference_loop(vv_loop, gap_loop)
        return vv_loop

    def try_offset(self, loop, offset, side):
        """
        Function that tries calling self.offset segment with offset_clipper
        first and then retries it using offset if an geometry error is captured.

        Parameters
        ----------
        loop: Loop
            Loop used to defined the output loop (here the thermal shield loop)
        offset: float
            Thickness of the output loop, in contact with the input one.
        side: str
            String indicating if the inboard or the outboard TS is built:
            "Inboard": inboard section
            "Outboard": outboard section

        Returns
        -------
        offset_loop: Loop
            In/outboard loop defined inside and in contact with the input one
            of a thickness offset.
        """
        # Try using the basic offset
        offset_result = self.offset_segment(
            loop=loop, offset=offset, side=side, clipper=False
        )

        # If the segmentation cut does not behave as expected, try offset_clipper
        if len(offset_result) != 2:
            offset_result = self.offset_segment(
                loop=loop, offset=offset, side=side, clipper=True
            )

        # If the segmentation cut behaves as expected, extract it
        if len(offset_result) == 2:
            return offset_result[1]

        # Otherwise, raise errors
        elif len(offset_result) == 1:
            raise GeometryError(
                f"The cutted shell is made with intersecting loops on {side} side"
                "Please check your initial thermal shield shape."
            )
        else:
            raise GeometryError(
                f"The {side} boolean cut is providing {len(offset_result)} solutions,"
                " 2 are expected. Please check your initial thermal shield shape."
            )

    def offset_segment(self, loop, offset, side, clipper):
        """
        Generate an in/outboard inner segment loop from a the corresponing
        outer segment. This is used here to make the vaccum vessel sections
        from the thermal shield ones.

        Parameters
        ----------
        loop: Loop
            Loop used to defined the output loop (here the thermal shield loop)
        offset: float
            Thickness of the output loop, in contact with the input one.
        side: str
            String indicating if the inboard or the outboard TS is built:
            "Inboard": inboard section
            "Outboard": outboard section
        clipper: bool
            Option to use offset_clipper instead of offset

        Returns
        -------
        offset_loop: [Loop]
            List of loops from the offest boolean cut.
        """
        # Building a shell around the input shell
        if clipper:
            offset_loop = loop.offset_clipper(offset)
            offset_loop = clean_loop(offset_loop)  # Removing redundant points
        else:
            offset_loop = loop.offset(offset)
        offset_loop = simplify_loop(offset_loop)

        try:
            offset_shell = Shell(loop, offset_loop)
        except GeometryError:
            return []

        # Cutting the right part
        if side in ["Inboard"]:
            cutter_loop = make_box_xz(
                x_min=self.params.r_vv_joint,
                x_max=np.amax(offset_loop.x) + 0.1,
                z_min=-np.amax(offset_loop.z) - 0.1,
                z_max=np.amax(offset_loop.z) + 0.1,
            )

        # Cutting the left part
        elif side in ["Outboard"]:
            cutter_loop = make_box_xz(
                x_min=np.amin(offset_loop.x) - 0.1,
                x_max=self.params.r_vv_joint,
                z_min=-np.amax(offset_loop.z) - 0.1,
                z_max=np.amax(offset_loop.z) + 0.1,
            )

        # Loop cut sanity check
        diff_result = boolean_2d_difference(offset_shell, cutter_loop)

        return diff_result

    def merge_vv(self):
        """
        Merge the inboard and outboards vaccum vessels assuming a vertical
        joint between the two if the two thermal shield are disconnected.

        TODO: This function is also written in SegmentedThermalShield. It should
        be written outside the segmented TS or VV objects to avoid code
        duplication.
        NOTE: This algorithm is indeed wierd, but shape agnostic.
        """
        # Find the intersection between the joint and the inner and outer TS
        joint_vertical_plane_ib = Plane(
            [self.params.r_vv_joint - 1.0e-4, 0.0, 0.0],
            [self.params.r_vv_joint - 1.0e-4, 0.0, 1.0],
            [self.params.r_vv_joint - 1.0e-4, 1.0, 1.0],
        )
        joint_vertical_plane_ob = Plane(
            [self.params.r_vv_joint + 1.0e-4, 0.0, 0.0],
            [self.params.r_vv_joint + 1.0e-4, 0.0, 1.0],
            [self.params.r_vv_joint + 1.0e-4, 1.0, 1.0],
        )
        joint_ib_intersect = loop_plane_intersect(
            self.geom["Inboard profile"],
            joint_vertical_plane_ib,
        )
        joint_ob_intersect = loop_plane_intersect(
            self.geom["Outboard profile"],
            joint_vertical_plane_ob,
        )

        # Get the top intersection positions
        z_ib_joint_top = np.amax(joint_ib_intersect.T[2])
        z_ob_joint_top = np.amax(joint_ob_intersect.T[2])
        z_ib_joint_bot = z_ib_joint_top - self.params.tk_vv_in
        z_ob_joint_bot = z_ob_joint_top - self.params.tk_vv_out

        contact = (
            (  # Case 1: The outboard is on top of the inboard
                z_ib_joint_top < z_ob_joint_top and z_ob_joint_bot < z_ib_joint_top
            )
            or (  # Case 2: The inboard is on top of the outboard
                z_ib_joint_top > z_ob_joint_top and z_ib_joint_bot < z_ob_joint_top
            )
            or z_ib_joint_top == z_ob_joint_top
        )

        # If the two sections are in contact they can be merged trivially
        if contact:
            vv_outer_loop, vv_inner_loop = boolean_2d_union(
                self.geom["Inboard profile"],
                self.geom["Outboard profile"],
            )[:2]

        # If the two sections are disconnected a joint section
        # must be used to merge the sections
        else:
            loop_x = np.array(
                [
                    self.params.r_vv_joint - 1.0e-4,
                    self.params.r_vv_joint + 1.0e-4,
                    self.params.r_vv_joint + 1.0e-4,
                    self.params.r_vv_joint - 1.0e-4,
                ]
            )
            loop_z = np.array(
                [
                    max(z_ib_joint_top, z_ob_joint_top),
                    max(z_ib_joint_top, z_ob_joint_top),
                    min(z_ib_joint_bot, z_ob_joint_bot),
                    min(z_ib_joint_bot, z_ob_joint_bot),
                ]
            )
            vv_junction_loop_top = Loop(x=loop_x, z=loop_z)
            vv_junction_loop_bot = Loop(x=loop_x, z=-loop_z)
            vv_junction_loop_top.close()
            vv_junction_loop_bot.close()

            build_loop = boolean_2d_union(
                self.geom["Inboard profile"],
                vv_junction_loop_top,
            )[0]
            build_loop = boolean_2d_union(
                self.geom["Outboard profile"],
                build_loop,
            )[0]

            # Try to merge the two sections without a joint
            loop_union = boolean_2d_union(
                build_loop,
                vv_junction_loop_bot,
            )
            vv_outer_loop, vv_inner_loop = loop_union[:2]
        # The output
        self.geom["2D profile"] = Shell(vv_inner_loop, vv_outer_loop)

    def make_offset_inner_profile(self, vv_inboard_offset):
        """
        Returns the Loop corresponding to the inner profile of the
        vacuum vessel, with an offset on the inboard side taken from
        params.vv_inboard_offset

        Returns
        -------
        inner_cut : Loop
        """
        inner = self.geom["2D profile"].inner
        inboard = self.geom["Inboard profile"]
        inboard_offset = inboard.offset_clipper(vv_inboard_offset)
        inboard_offset = simplify_loop(inboard_offset)
        inner_cut_list = boolean_2d_difference(inner, inboard_offset)
        inner_cut = inner_cut_list[0]
        return inner_cut

    @property
    def xz_plot_loop_names(self) -> list:
        """
        Selection of the loops to be plotted with plot_xz()
        NOTE: The cryostat is not yet supported.

        Returns
        -------
        List:
            list of the selected loop names
        """
        return ["Inboard profile", "Outboard profile"]


class VacuumVesselPlotter(ReactorSystemPlotter):
    """
    The plotter for a Vacuum Vessel.
    """

    def __init__(self):
        super().__init__()
        self._palette_key = "VV"

    def plot_xz(self, plot_objects, ax=None, **kwargs):
        """
        Plot the vacuum vessel in x-z.
        """
        self._apply_default_styling(kwargs)
        alpha = kwargs["alpha"]
        if not isinstance(alpha, list) and not isinstance(alpha, cycle):
            alpha = float(alpha)
            alpha2 = alpha * 0.5
            kwargs["alpha"] = [alpha2] + [alpha] * (len(plot_objects) - 1)
        super().plot_xz(plot_objects, ax=ax, **kwargs)


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
