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
Thermal shield system
"""
from itertools import cycle
from typing import Type

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.parameter import ParameterFrame
from BLUEPRINT.cad.shieldCAD import SegmentedThermalShieldCAD, ThermalShieldCAD
from BLUEPRINT.geometry.boolean import (
    boolean_2d_difference,
    boolean_2d_union,
    simplify_loop,
)
from BLUEPRINT.geometry.geombase import Plane
from BLUEPRINT.geometry.geomtools import loop_plane_intersect
from BLUEPRINT.geometry.loop import Loop, MultiLoop, make_ring, point_loop_cast
from BLUEPRINT.geometry.shell import Shell
from BLUEPRINT.systems.baseclass import ReactorSystem
from BLUEPRINT.systems.plotting import ReactorSystemPlotter
from BLUEPRINT.utilities.tools import get_max_PF


class ThermalShield(ReactorSystem):
    """
    The thermal shield system for the tokamak hall cold mass
    """

    config: Type[ParameterFrame]
    inputs: dict

    # fmt: off
    default_params = [
        ['tk_ts', 'TS thickness', 0.05, 'm', None, 'Input'],
        ['g_ts_pf', 'Clearances to PFs', 0.075, 'm', None, 'Input'],
        ['g_vv_ts', 'Gap between VV and TS', 0.05, 'm', None, 'Input'],
        ['g_cs_tf', 'Gap between CS and TF', 0.05, 'm', None, 'Input'],
        ['g_ts_tf', 'Gap between TS and TF', 0.05, 'm', None, 'Input'],
        ['g_ts_tf_topbot', 'Vessel KOZ offset to TF coils on top and bottom edges', 0.11, 'm', None, 'Input'],
        ['pf_off', 'Cryostat TS PF offset', 0.3, 'm', None, 'Input'],
        ['n_TF', 'Number of TF coils', 16, 'dimensionless', None, 'Input'],
        ['A', 'Plasma aspect ratio', 3.1, 'dimensionless', None, 'Input'],
        ['R_0', 'Major radius', 9, 'm', None, 'Input'],
        ['LPangle', 'Lower port inclination angle', -25, '°', None, 'Input'],
        ['e', 'Emissivity of TS surface', 0.05, 'N/a', '0.05=silvered surface', 'Input']
    ]
    # fmt: on
    CADConstructor = ThermalShieldCAD

    def __init__(self, config, inputs):
        self.config = config
        self.inputs = inputs
        self._plotter = ThermalShieldPlotter()

        self._init_params(self.config)
        self.tf_min_surf()

        self.build_vvts(inboardoff=self.params.g_vv_ts, topbotoff=self.params.g_vv_ts)
        self.color = (0.28627450980392155, 0.98431372549019602, 0.14509803921568629)

    def build_vvts(self, inboardoff=0.05, topbotoff=0.03):
        """
        Builds the vacuum vessel thermal shield. Simple offset from VV

        NOTE: This function does not build anything related to the vaccum vessel
        despite the vv in the name. The vvts name comes from an ITER convention
        to distinguish the TS adjacent to the VV and the cryostat TS.
        """
        vv_out = self.inputs["VV 2D outer"]
        if inboardoff == topbotoff:
            # No need for blending
            vvts_inner = vv_out.offset(inboardoff)
        else:
            top = vv_out.offset(topbotoff)
            mid = vv_out.offset(inboardoff)
            vvts_inner = LoopBlender(top, mid)
            vvts_inner = vvts_inner.build()

        self.geom["2D profile"] = Shell.from_offset(vvts_inner, self.params.tk_ts)

    def tf_min_surf(self):
        """
        Calculates exclusion zone for the TF coil
        """
        vvloop = self.inputs["VV 2D outer"]
        inboardoff = self.params.g_ts_tf + self.params.g_vv_ts
        topbotoff = self.params.g_ts_tf_topbot + self.params.g_vv_ts
        if inboardoff == topbotoff:
            # No need for blending: the offsets are the same
            koz = vvloop.offset(inboardoff + self.params.tk_ts)
        else:
            top = vvloop.offset(topbotoff + self.params.tk_ts)  # TF exclusion top
            mid = vvloop.offset(inboardoff + self.params.tk_ts)  # TF exclusion inboard
            koz = LoopBlender(top, mid)
            koz = koz.build()

        self.TF_koz = koz

    def build_cts(self, inputs):
        """
        Builds the cryostat thermal shield. First wraps a "rope" around the PF
        coil top right and bottom left corners. Then checks for intersection
        with the TF coil. If the TF coil pokes out from the rope-wrap, does
        an offset and then convex hull of the two.

        Parameters
        ----------
        TFprofile: BLUEPRINT Loop
            Outer edge of the toroidal field coil casing
        PFcoilset: bluemira.equilibria CoilSet object
            The set of poloidal field coils (including central solenoid)


        Updates the geometry dictionary with the cryostat 2-D build info
        """
        coilset = inputs["PFcoilset"]

        x, z = [], []
        # going clockwise for once! (ITER-like)
        for coil in coilset.coils.values():
            if coil.ctype == "PF":
                dx, dz = coil.dx, coil.dz
                if coil.z < 0:
                    dz *= -1
                x.append(coil.x + dx)
                z.append(coil.z + dz)
        # find bottom of CS
        solenoid = coilset.get_solenoid()
        cslowz = solenoid.z_min - self.params.pf_off
        if cslowz < min(z):
            x.append(solenoid.radius + solenoid.dx)
            z.append(cslowz)
        # find top of CS
        cshighz = solenoid.z_max + self.params.pf_off
        if cshighz > max(z):
            x.insert(0, solenoid.radius + solenoid.dx)
            z.insert(0, cshighz)
        # Bring edges to axis
        x.append(0)
        z.append(min(z))
        # Adjust lower part of cts
        if np.argmin(z) != len(z) - 1:
            # flatten out the bottom  // stop re-entrant profile
            for i in range(np.argmin(z) + 1, len(z) - 1):
                z[i] = min(z)

        x.insert(0, 0)
        z.insert(0, max(z))

        loop = Loop(x=x, z=z)
        loop = loop.offset(self.params.pf_off)
        tf = Loop(**inputs["TFprofile"]["out"])

        tfoff = tf.offset(self.params.pf_off + self.params.tk_ts)
        cts = loop.as_shpoly()
        tf = tfoff.as_shpoly()
        if cts.intersects(tf):
            cts = cts.union(tf).convex_hull
            loop = Loop(**dict(zip(loop.plan_dims, cts.exterior.xy)))
            loop.open_()
            zmax = max(loop.z)
            if loop.z[-1] < zmax:
                # Correction pour la position du CTS inboard qui des fois peut
                # peut replonger en z si les PFs sont moin eleves que le TF
                loop.z[-1] = zmax
                self._adjust_upper_port(zmax)

        offset_loop = loop.offset(self.params.tk_ts)
        cts_cut = offset_loop.offset(10)  # To make cut for CTS CAD
        shell = Shell(loop, offset_loop)
        cts = shell.connect_open_loops()
        cts_cut = Shell(offset_loop, cts_cut)
        cts_cut = cts_cut.connect_open_loops()
        self.geom["CTS cut out"] = cts_cut
        self.geom["Cryostat TS"] = cts

    def build_ports(self, inputs):
        """
        Builds the thermal shield port penetrations around the VV ports

        Parameters
        ----------
        Div_cog: [float, float]
            Centre of gravity of the divertor 2-D cross-section
        TFprofile: BLUEPRINT Loop
            Outer edge of the toroidal field coil casing
        PFcoilset: bluemira.equilibria CoilSet object
            The set of poloidal field coils (including central solenoid)
        lp_height: float
            Height of the lower port

        Updates the geometry dictionary with the cryostat 2-D build info
        """
        self._build_upper_port(inputs)
        if "Cryostat TS" in self.geom:  # Facilitate flexible build order
            x_out = max(self.geom["Cryostat TS"]["x"])
        else:
            x_out = get_max_PF(inputs["PFcoilset"].coils) + self.params.pf_off
            # Required in case the TF sticks out further radially than PFs
            x_out2 = max(inputs["TFprofile"]["out"]["x"]) + self.params.pf_off

            x_out = max(x_out, x_out2)

        # TODO: extract variables and place into config
        vv_port_w = 2
        lp_duct_w = 3.5
        x_kink = self.params.R_0 + 6
        self._build_eq_port(x_out, vv_port_w)
        self._build_lower_port(inputs, x_kink, x_out, lp_duct_w)
        self._generate_xz_plot_loops()

    def adjust_ports(self, inputs=None):
        """
        Adjust the thermal shield ports.
        """
        self._adjust_eq_port()

    def _build_upper_port(self, inputs):
        """
        Das hier baut den Upper Port. Findet aber zuerst die zwei PFs die am
        nächsten zur besten/natürlichsten Position sind (am beiden Seiten von
        R_0).
        """
        beta = np.pi / self.params.n_TF
        xmin = min(self.geom["2D profile"].inner["x"])
        xmax = max(self.geom["2D profile"].inner["x"])

        # uc = nested_dict_search(inputs['PFcoilset'].coils, ['z>0'])
        upper_coils = {}
        for name, coil in inputs["PFcoilset"].coils.items():
            if coil.ctype == "PF":
                if coil.z > 0:
                    upper_coils[name] = coil

        xin, xout, ztop = [], [], []
        for n, c in upper_coils.items():
            if c.x + c.dx > xmin and c.x < self.params.R_0:
                xin.append(c.x + c.dx)

            if xmax > c.x > self.params.R_0:
                xout.append(c.x - self.params.R_0 - c.dx)
            ztop.append(c.z + c.dz)
        xin = max(xin) if len(xin) != 0 else 0

        xout = (
            max(xout) + self.params.R_0 if len(xout) != 0 else 10 * self.params.R_0
        )  # big
        ztop = max(ztop) + self.params.pf_off
        pf1_ro = max(xin, xmin)
        pf2_ri = min(xout, xmax)
        tf_w = (
            inputs["TFsection"]["case"]["side"] * 2
            + inputs["TFsection"]["winding_pack"]["depth"]
        )
        x_b, y_b = 0.5 * tf_w / np.tan(beta), 0.5 * tf_w
        x_bp = np.sqrt(y_b**2 + x_b**2)
        x_oi = x_bp + self.params.g_ts_tf / np.sin(beta)
        x_inner = pf1_ro + self.params.g_ts_pf
        y_inner = np.tan(beta) * (x_inner - x_oi)
        x_outer = np.cos(beta) * ((pf2_ri - self.params.g_ts_pf)) + np.sin(beta) * (
            tf_w / 2 + self.params.g_ts_tf
        )  # trimmed
        y_outer = np.tan(beta) * (x_outer - x_oi)
        port = Loop(
            x=[x_inner, x_outer, x_outer, x_inner],
            y=[-y_inner, -y_outer, y_outer, y_inner],
            z=[ztop],
        )
        port.close()
        port = Shell.from_offset(port, -self.params.tk_ts)
        # =============================================================================
        #         UP.plot()
        #         UP = UpperPort(UP,
        #                        x_in_min=xmin+self.p.tk_ts,
        #                        x_in_max=max(self.geom['VVTS'].inner['x'])*np.cos(beta),
        #                        x_out_min=x_inner,
        #                        x_out_max=xmax*np.cos(beta),
        #                        L_min=self.p.tk_ts)
        # =============================================================================
        self.geom["Upper port"] = port
        self.ztop = ztop  # Und?

    def _adjust_upper_port(self, zmax):
        z = self.ztop
        dz = zmax - z
        self.geom["Upper port"].translate([0, 0, dz])

    def _build_eq_port(self, x_out, vv_port_w):
        eqportw = vv_port_w + 2 * self.params.g_vv_ts
        y = [
            eqportw / 2 + self.params.tk_ts,
            eqportw / 2 + self.params.tk_ts,
            -eqportw / 2 - self.params.tk_ts,
            -eqportw / 2 - self.params.tk_ts,
            eqportw / 2 + self.params.tk_ts,
        ]
        z = [
            -eqportw / 2 - self.params.tk_ts,
            eqportw / 2 + self.params.tk_ts,
            eqportw / 2 + self.params.tk_ts,
            -eqportw / 2 - self.params.tk_ts,
            -eqportw / 2 - self.params.tk_ts,
        ]
        eq_outer = Loop(x=x_out, y=y, z=z)
        self.geom["Equatorial port"] = Shell.from_offset(eq_outer, -self.params.tk_ts)

    def _adjust_eq_port(self):
        x = max(self.geom["Cryostat TS"]["x"])
        dx = x - self.geom["Equatorial port"].inner["x"][0]
        self.geom["Equatorial port"].translate([dx, 0, 0])

    def _build_lower_port(self, inputs, x_kink, x_out, lp_duct_w):
        def _calculate_port_point(seed_point):
            p_point = point_loop_cast(seed_point, tfi, lp_angle)  # x, z
            # Point on TF coil: D
            d_point = [p_point[0], (p_point[0] - point_b[0]) * np.tan(beta)]  # x, y
            # Point on TS  Got lazy with z
            return [d_point[0], d_point[1] - self.params.g_vv_ts]  # C

        beta = np.pi / self.params.n_TF
        lp_angle = self.params.LPangle
        lp_height = inputs["lp_height"] + 0.5  # clearance
        div_cog = inputs["Div_cog"]
        x_straight = x_kink + 6
        # Point A lowest point of inner VVTS shell
        if False:
            zts_lp = min(self.geom["2D profile"].inner["z"])
            xts_lp = self.geom["2D profile"].inner["x"][
                np.argmin(self.geom["2D profile"].inner["z"])
            ]
        # Point A (divertor offset)
        if True:
            zts_lp = div_cog[1] - lp_height / 2
            inner = self.geom["2D profile"].inner
            arg = inner.receive_projection([div_cog[0], zts_lp], lp_angle, get_arg=True)
            xts_lp = inner.x[arg]

        point_a = [xts_lp, zts_lp]
        tfi = inputs["TFprofile"]["in"]
        tf_width = (
            2 * (inputs["TFsection"]["case"]["side"])
            + inputs["TFsection"]["winding_pack"]["depth"]
        )
        # Point at which the TF coils separate: B
        point_b = [(tf_width / 2) / np.tan(beta), 0]  # x, y

        e1 = _calculate_port_point(point_a)
        e2 = _calculate_port_point([xts_lp, zts_lp + lp_height])
        # Points in xyz at x = x_kink

        z_kink_1 = zts_lp + (x_kink - xts_lp) * np.sin(np.radians(lp_angle))
        z_kink_2 = z_kink_1 + lp_height
        lp_outer = Loop(
            x=x_kink,
            y=[e1[1], -e1[1], -e2[1], e2[1]],
            z=[z_kink_1, z_kink_1, z_kink_2, z_kink_2],
        )
        lp_outer.close()
        self.geom["Lower port"] = Shell.from_offset(lp_outer, -self.params.tk_ts)
        self.geom["LP path"] = Loop(
            x=[xts_lp, x_kink, x_straight], z=[zts_lp, z_kink_1, z_kink_1]
        )

        y_lp_o = [lp_duct_w / 2, -lp_duct_w / 2, -lp_duct_w / 2, lp_duct_w / 2]

        z = self.geom["Lower port"].outer.z
        z_lp_o = [z[2], z[2], z[0], z[0]]

        lp_inner = Loop(x=x_out, y=y_lp_o, z=z_lp_o)
        lp_inner.close()
        self.geom["Lower duct"] = Shell.from_offset(lp_inner, self.params.tk_ts)

    @property
    def xz_plot_loop_names(self):
        """
        The x-z loop names to plot.
        """
        if "Cryostat TS" in self.geom:
            return ["2D profile", "Full 2D profile", "Cryostat TS"]
        else:
            return ["2D profile"]

    def _generate_xz_plot_loops(self):
        """
        Generates the geometrical information of the 2-D cross-section of the
        vacuum vessel, joining the ports to the main shells.
        """
        body = self.geom["2D profile"]
        upvec = [0, 0, -10]
        up = self._make_xz_shell(self.geom["Upper port"].outer, upvec, self.params.tk_ts)
        eqvec = [-10, 0, 0]
        eq = self._make_xz_shell(
            self.geom["Equatorial port"].outer, eqvec, self.params.tk_ts
        )
        x_a = self.geom["Lower port"].inner.x[0]
        a = np.argmin(body.inner.z)
        x_b = body.inner.x[a]
        dx = x_b - x_a
        dz = dx * np.tan(np.deg2rad(self.params.LPangle))
        sdt = self.params.tk_ts * np.cos(np.deg2rad(-self.params.LPangle))
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
        eqc = eq.inner.translate([self.params.tk_ts * 2, 0, 0], update=False)
        eqc = self._make_xz_shell(eqc, eqvec, 0)
        lp = self.geom["Lower port"].copy()
        lpvec = np.array(lpvec)
        lpc = lp.inner.translate(-2 * sdt * lpvec / np.linalg.norm(lpvec), update=False)
        lpc = self._make_xz_shell(lpc, lpvec, 0)
        # store lower port exclusion zone
        self.geom["Lower port exclusion"] = lpc.offset(self.params.tk_ts)

        outer = aub.outer.copy()
        for cut in [upc, eqc, lpc]:
            outer = boolean_2d_difference(outer, cut)
        final = boolean_2d_difference(outer, aub.inner)
        cryo_ts = MultiLoop(final, stitch=False)

        try:
            self.geom["Full 2D profile"] = cryo_ts.clip(self.geom["Cryostat TS"])
        except KeyError:
            # Cryostat TS information not yet available!
            pass
        return super()._generate_xz_plot_loops()

    @staticmethod
    def _make_xz_shell(loop, vector, thickness):
        """
        Isso aqui e util para fazer um Shell pra as portas

        Parameters
        ----------
        xy_shell: BLUEPRINT Shell object
            The X-Y(ish) shell from which to make an X-Z shell
        vector: tuple(3)
            The extrusion vector of the xy_shell

        Returns
        -------
        xz_shell: BLUEPRINT Shell object
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

    @property
    def xy_plot_loop_names(self):
        """
        The x-y loop names to plot.
        """
        return ["Upper port", "inboard X-Y", "outboard X-Y", "Cryostat TS X-Y"]

    def _generate_xy_plot_loops(self):
        inboard = make_ring(
            min(self.geom["2D profile"].outer.x), min(self.geom["2D profile"].inner.x)
        )
        self.geom["inboard X-Y"] = inboard
        outboard = make_ring(
            max(self.geom["2D profile"].inner.x), max(self.geom["2D profile"].outer.x)
        )
        self.geom["outboard X-Y"] = outboard
        plane = Plane([0, 0, 0], [1, 0, 0], [0, 1, 0])
        inter = self.geom["Cryostat TS"].section(plane)
        self.geom["Cryostat TS X-Y"] = make_ring(inter[0][0], inter[1][0])
        return super()._generate_xy_plot_loops()


class SegmentedThermalShield(ReactorSystem):
    """
    Segmented thermal shield (TS) class

    This class generates a thermal shield system made of two sections:
    - The inboard: parametrized by its own TF to TS gap and its thickness.
    - The outboard: parametrized by its own TF to TS gap and its thickness.
    The two sections are separated at a radius provided by the user

    To support PF coils inside the TF coil cage, the build order had to be
    modified. The TF coil and the PF coil objects must be built before the
    thermal shield. Invertly no VV shape is needed to build the segmented
    TS. The TS wraps around any PF coils. This wrapping is done automatically
    if a picture frame geometry is used.

    As the inner PF coil have not been tested for other TF coil shapes
    than a Tapered coil shape, an error will be thrown if other shapes
    are being used.

    TODO: The cryostat thermal shield is not yet supported.
    TODO: Maintenance port must be added (potenitally via inheritance)

    Parameters
    ----------
    config: Dict[list[Coil], Loop]
        key : "TF inner loop""
            Inner TF coil loop used as a starting point for the optimization
        key: "inner PF coils"
            List of PF coil inside the TF cage to wrapp with the therma shield
            shape.
    inputs: Parameters (BLUEPRINT class)

    Other Parameters
    ----------------
    params.tk_ib_ts: float
        Inboard thermal shield thickness.
        TODO: Set an inboard TS thickness in PROCESS and connect it to BP
    params.tk_ob_ts: float
        Outboard thermal shield thickness.
        TODO: Set an outboard TS thickness in PROCESS and connect it to BP
    params.g_ib_ts_tf: float
        Inboard TF to TS gap
        TODO: Set an inboard TS TF gap in PROCESS and connect it to BP
    params.g_ob_ts_tf: float
        Inboard TF to TS gap
        TODO: Set an outboard TS TF gap in PROCESS and connect it to BP
    params.g_ts_pf: float
        Minimal gap between the thremal shield and the PF coils
    params.r_ts_joint: float
        Radial position of the ouboard to inboard thermal shield joint
    tf_inner_loop: Loop
        Inner TF loop
    pf_coils: CoilSet
        PF coils objects necessary to shape the TS properly
    geom: dict[Loop, Shell]
        geom["Inboard profile"]: Loop
            Loop defining the 2D inboard thermal shield
        geom["Outboard profile"]: Loop
            Loop defining the 2D outboard thermal shield
        geom["Cryostat TS"]: Loop
            Loop defining the 2D cryostat thermal shield
        geom["2D profile"]: Shell
            Shell defined by the union between the inboard and the outboard loop
            decribing the whole thermal shield shape to be used in the rest
            of the build
    """

    config: Type[ParameterFrame]
    inputs: dict

    # fmt: off
    default_params = [
        ['n_TF', 'Number of TF coils', 16, 'dimensionless', None, 'Input'],
        ['tk_ib_ts', 'Inboard TS thickness', 0.05, 'm', None, 'Input'],
        ['tk_ob_ts', 'Outboard TS thickness', 0.05, 'm', None, 'Input'],
        ['g_ib_ts_tf', 'Inboard gap between TS and TF', 0.05, 'm', None, 'Input'],
        ['g_ob_ts_tf', 'Outboard gap between TS and TF', 0.05, 'm', None, 'Input'],
        ['g_ts_pf', 'Clearances to PFs', 0.075, 'm', None, 'Input'],
        ['r_ts_joint', 'Radius of inboard/outboard TS joint', 2. , 'm', None, 'Input'],
        ['tk_cryo_ts', 'Cryo TS thickness', 0.10, 'm', None, 'Input'],
        ['r_cryo_ts', 'Radius of outboard cryo TS', 11, 'm', None, 'Input'],
        ['z_cryo_ts', 'Half height of outboard cryo TS', 14, 'm', None, 'Input'],
    ]
    # fmt: on
    CADConstructor = SegmentedThermalShieldCAD

    def __init__(self, config, inputs):

        # Set the thremal shield graphical   color scheme
        self.color = (0.28627450980392155, 0.98431372549019602, 0.14509803921568629)

        # Loading the default parameters
        self.config = config
        self.inputs = inputs
        self._plotter = ThermalShieldPlotter()

        # Loading the input parameters
        # Rem: config does not seems to be a config file but rather containing
        #      the genuine class inputs.
        self.params = ParameterFrame(self.default_params.to_records())
        self._init_params(self.config)

        # Loading the inputs loops into class attributes
        self.tf_inner_loop = self.inputs["TF inner loop"]
        self.inner_pf_list = self.inputs["inner PF coils"]

        # Loading wrapping option
        self.pf_wrap_opt = dict()
        if isinstance(self.inputs["PF wrapping"], str):
            # Check the wrapping option
            if self.inputs["PF wrapping"] not in ["U", "L", "top bot", "vertical gap"]:
                raise NotImplementedError(
                    f"Non supported inner PF coil wrapping"
                    f"{self.inputs['PF wrapping']}"
                )

            # Set the individual PF coil wrapping option
            for coil in self.inner_pf_list:
                self.pf_wrap_opt[coil.name] = self.inputs["PF wrapping"]

        # TODO: Setup the individual wrapping option test
        elif not isinstance(self.inputs["PF wrapping"], dict):
            raise NotImplementedError(
                "The PF wrapping option should be either str or a dict"
            )

        # Load the dict
        else:
            self.pf_wrap_opt = self.inputs["PF wrapping"]

        # Building the thermal shield 2D geometry
        self.build_vvts()

    def build_vvts(self):
        """
        Builds the thermal shield 2D loops and shells.
        The inner and the outer section are desctibed by two, potentially
        disconnected loops. The overall thermal shield shell is obtained
        by the union of the two with a 2e-4 m radially thick union loop
        allowing obtaining the shell even with disconnected TS, correponding
        more or less to the joints.

        NOTE: This function does not build anything related to the vaccum vessel
        despite the vv in the name. The vvts name comes from an ITER convention
        to distinguish the TS adjacent to the VV and the cryostat TS.

        Returns
        -------
        Dict[Loop, Shell]
            geom["Inboard profile"]: Loop
                Loop defining the 2D inboard thermal shield
            geom["Outboard profile"]: Loop
                Loop defining the 2D outboard thermal shield
            geom["2D profile"]: Shell
                Shell defined by the union between the inboard and the outboard loop
                decribing the whole thremal shield shape
        """
        # Make the inboard thermal shield loop
        self.geom["Inboard profile"] = self.build_vvts_section(
            side="Inboard",
            pf_kozs=self.build_pf_koz(side="Inboard"),
            g_ts_tf=self.params.g_ib_ts_tf,
            tk_ts=self.params.tk_ib_ts,
        )

        # Make the outboard thermal shield loop
        self.geom["Outboard profile"] = self.build_vvts_section(
            side="Outboard",
            pf_kozs=self.build_pf_koz(side="Outboard"),
            g_ts_tf=self.params.g_ob_ts_tf,
            tk_ts=self.params.tk_ob_ts,
        )

        # Generate the union between the inboard and outboard TS loop
        self.merge_vvts()

    def merge_vvts(self):
        """
        Merge the inboard and outboards thermal shields assuming a vertical
        joint between the two if the two thermal shield are disconnected.
        """
        # Find the intersection between the joint and the inner and outer TS
        joint_vertical_plane_ib = Plane(
            [self.params.r_ts_joint - 1.0e-4, 0.0, 0.0],
            [self.params.r_ts_joint - 1.0e-4, 0.0, 1.0],
            [self.params.r_ts_joint - 1.0e-4, 1.0, 1.0],
        )
        joint_vertical_plane_ob = Plane(
            [self.params.r_ts_joint + 1.0e-4, 0.0, 0.0],
            [self.params.r_ts_joint + 1.0e-4, 0.0, 1.0],
            [self.params.r_ts_joint + 1.0e-4, 1.0, 1.0],
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
        z_ib_joint_bot = z_ib_joint_top - self.params.tk_ib_ts
        z_ob_joint_bot = z_ob_joint_top - self.params.tk_ob_ts

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
            ts_outer_loop, ts_inner_loop = boolean_2d_union(
                self.geom["Inboard profile"],
                self.geom["Outboard profile"],
            )[:2]

        # If the two sections are disconnected a joint section
        # must be used to merge the sections
        else:
            loop_x = np.array(
                [
                    self.params.r_ts_joint - 1.0e-4,
                    self.params.r_ts_joint + 1.0e-4,
                    self.params.r_ts_joint + 1.0e-4,
                    self.params.r_ts_joint - 1.0e-4,
                ]
            )
            maxtop = max(z_ib_joint_top, z_ob_joint_top)
            minbot = min(z_ib_joint_bot, z_ob_joint_bot)
            loop_z = np.array([maxtop, maxtop, minbot, minbot])
            ts_junction_loop_top = Loop(x=loop_x, z=loop_z)
            ts_junction_loop_bot = Loop(x=loop_x, z=-loop_z)
            ts_junction_loop_top.close()
            ts_junction_loop_bot.close()

            build_loop = boolean_2d_union(
                self.geom["Inboard profile"],
                ts_junction_loop_top,
            )[0]
            build_loop = boolean_2d_union(
                self.geom["Outboard profile"],
                build_loop,
            )[0]

            # Try to merge the two sections without a joint
            ts_outer_loop, ts_inner_loop = boolean_2d_union(
                build_loop,
                ts_junction_loop_bot,
            )[:2]

        # The output
        self.geom["2D profile"] = Shell(inner=ts_inner_loop, outer=ts_outer_loop)

    def build_pf_koz(self, side) -> list:
        """
        Builds the PF coils exclusions zones

        Arguments
        ---------
        side: str
            String indicating if the inboard or the outboard TS is built:
            "Inboard": inboard section
            "Outboard": outboard section

        Returns
        -------
        pf_koz: List(Loop)
            List of PF keep out zones to be applied to a given section
        """
        if side not in ["Inboard", "Outboard"]:
            raise NotImplementedError(
                f"{side} side option not implemented"
                f"Only 'Inboard' and 'Outboard' implemented"
            )

        pf_kozs = []
        for inner_PF in self.inner_pf_list:

            # Skip PF coils does not affect the section (inboard side test)
            if (
                side in ["Inboard"]
                and inner_PF.x - inner_PF.dx - self.params.g_ts_pf
                > self.params.r_ts_joint
            ):
                continue

            # Skip PF coils does not affect the section (out board side test)
            if (
                side in ["Outboard"]
                and inner_PF.x + inner_PF.dx + self.params.g_ts_pf
                < self.params.r_ts_joint
            ):
                continue

            # Getting the individual coil wrapping option
            wrap_opt = self.pf_wrap_opt[inner_PF.name]

            # Initial exclusion loop: PF corners + clearance space
            pf_exclusion_loop = Loop(x=inner_PF.x_corner, z=inner_PF.z_corner)
            pf_exclusion_loop.close()
            pf_exclusion_loop = pf_exclusion_loop.offset(self.params.g_ts_pf)

            # Top/bot, left/right corner poision (for code redability)
            bot_z = np.amin(inner_PF.z_corner)
            top_z = np.amax(inner_PF.z_corner)
            left_x = np.amin(inner_PF.x_corner)
            right_x = np.amax(inner_PF.x_corner)

            # Used to get the PF to TF distances
            # ---
            # Planes obtained extrapolating the PF sides
            top_hoziontal_plane = Plane(
                [right_x, 0.0, top_z],
                [left_x, 0.0, top_z],
                [left_x, 1.0, top_z],
            )
            bottom_hoziontal_plane = Plane(
                [right_x, 0.0, bot_z],
                [left_x, 0.0, bot_z],
                [left_x, 1.0, bot_z],
            )
            left_vertical_plane = Plane(
                [left_x, 0.0, top_z],
                [left_x, 1.0, top_z],
                [left_x, 0.0, bot_z],
            )
            right_vertical_plane = Plane(
                [right_x, 0.0, top_z],
                [right_x, 0.0, bot_z],
                [right_x, 1.0, bot_z],
            )

            # Get intersect
            top_hoziontal_sect = loop_plane_intersect(
                self.tf_inner_loop, top_hoziontal_plane
            )
            bottom_hoziontal_sect = loop_plane_intersect(
                self.tf_inner_loop, bottom_hoziontal_plane
            )
            left_vertical_sect = loop_plane_intersect(
                self.tf_inner_loop, left_vertical_plane
            )
            right_vertical_sect = loop_plane_intersect(
                self.tf_inner_loop, right_vertical_plane
            )

            # Averaged PF to TF distance from intersection point
            dx_in = abs(
                0.5
                * (
                    np.amin(top_hoziontal_sect.T[0])
                    + np.amin(bottom_hoziontal_sect.T[0])
                )
                - inner_PF.x
            )
            dx_out = abs(
                0.5
                * (
                    np.amax(top_hoziontal_sect.T[0])
                    + np.amax(bottom_hoziontal_sect.T[0])
                )
                - inner_PF.x
            )
            dz_top = abs(
                0.5
                * (np.amax(left_vertical_sect.T[2]) + np.amax(right_vertical_sect.T[2]))
                - inner_PF.z
            )
            dz_bot = abs(
                0.5
                * (np.amin(left_vertical_sect.T[2]) + np.amin(right_vertical_sect.T[2]))
                - inner_PF.z
            )

            # Minimal distance
            # ---

            # Maximum TF coil dimensions
            z_tf_max = np.amax(self.tf_inner_loop["z"])
            z_tf_min = np.amin(self.tf_inner_loop["z"])
            x_tf_max = np.amax(self.tf_inner_loop["x"])
            x_tf_min = np.amin(self.tf_inner_loop["x"])

            # U wrapping: find the PF side the closest to the TF coil and wrap around it
            # ---
            if wrap_opt in ["U"]:

                # Case 1, top wrapping: PF coil closer to the top
                if dz_top < min(dz_bot, dx_out, dx_in):
                    pf_exclusion_loop.z[[2, 3]] = z_tf_max

                # Case 2, outboard wrapping: PF coil closer to the outboard leg
                elif dx_out < min(dz_top, dz_bot, dx_in):
                    pf_exclusion_loop.x[[1, 2]] = x_tf_max

                # Case 3, bottom wrapping: PF coil closer to the bottom
                elif dz_bot < min(dz_top, dx_out, dx_in):
                    pf_exclusion_loop.z[[0, 1, 4]] = z_tf_min

                # Case 4, inboard wrapping: PF coil closer to the inboard leg
                elif dx_in < min(dz_top, dz_bot, dx_out):
                    pf_exclusion_loop.x[[0, 3, 4]] = x_tf_min
            # ---

            # L wrapping: find the closest corner and wrap arount it
            # ---
            elif wrap_opt in ["L"]:

                # Outboard side wrapping
                if dx_out < dx_in:
                    pf_exclusion_loop.x[[1, 2]] = x_tf_max

                # Inboard side wrapping
                else:
                    pf_exclusion_loop.x[[0, 3, 4]] = x_tf_min

                # Top wrapping
                if dz_top < dz_bot:
                    pf_exclusion_loop.z[[2, 3]] = z_tf_max

                # Bottom wrapping
                else:
                    pf_exclusion_loop.z[[0, 1, 4]] = z_tf_min
            # ---

            # Top bottom wrapping: Force the wrapping to be made from the top/bot
            elif wrap_opt in ["top bot"]:

                # Case 1: top wrapping
                if dz_top < dz_bot:
                    pf_exclusion_loop.z[[2, 3]] = z_tf_max

                # Case 2: bottom wrapping
                else:
                    pf_exclusion_loop.z[[0, 1, 4]] = z_tf_min

            # Add an inboard or outboard vertical gap
            elif wrap_opt in ["vertical gap"]:

                # The gap takes all the vertical height
                pf_exclusion_loop.z[[2, 3]] = z_tf_max
                pf_exclusion_loop.z[[0, 1, 4]] = z_tf_min

                # Case 1: outboard side vertical gap
                if dx_out < dx_in:
                    pf_exclusion_loop.x[[1, 2]] = x_tf_max

                # Case 2: inboard side vertical gap
                else:
                    pf_exclusion_loop.x[[0, 3, 4]] = x_tf_min

            else:
                raise NotImplementedError(
                    f"{wrap_opt} is not an implemente wrapping option"
                    f"Available option: ['U', 'L', 'top bot', 'vertical gap']"
                )

            # Adding the KoZ to the output
            pf_kozs.append(pf_exclusion_loop)

        return pf_kozs

    def build_vvts_section(self, side, pf_kozs, g_ts_tf, tk_ts):
        """
        Builds the thermal shield inner or outer section.

        Arguments
        ---------
        side: str
            String indicating if the inboard or the outboard TS is built:
            "Inboard": inboard section
            "Outboard": outboard section
        pf_koz: List(Loop)
            List of loops delimitating the PF coils exclusion loops
        g_ts_tf: float
            Thickness of the gap between the TF coil and the thermal shields
            in the considered section
        tk_ts: float
            Thickness of the thermal shield in the considered section

        Returns
        -------
        ts_loop: Loop
            The inboard or outboard thermal shield 2D cross section
        """
        if side not in ["Inboard", "Outboard"]:
            raise NotImplementedError(
                f"{side} side option not implemented"
                f"Only 'Inboard' and 'Outboard' implemented"
            )

        # Offset the TF inner loop to represent the TF-TS gap
        ts_outer_loop = self.tf_inner_loop
        ts_outer_loop = ts_outer_loop.offset(-g_ts_tf)
        ts_outer_loop = simplify_loop(ts_outer_loop)

        # Remove the PF exclsions zones
        for pf_koz in pf_kozs:
            ts_outer_loop = boolean_2d_difference(ts_outer_loop, pf_koz)[0]
            ts_outer_loop = simplify_loop(ts_outer_loop)

        # Making the thermal shield inner loop
        ts_inner_loop = ts_outer_loop.offset(-tk_ts)
        ts_inner_loop = simplify_loop(ts_inner_loop)
        ts_inner_shell = Shell(ts_inner_loop, ts_outer_loop)

        # Split into the two sections
        split_out = ts_inner_shell.split_by_line(
            p2=np.array([self.params.r_ts_joint, np.amax(ts_outer_loop.z)]),
            p1=np.array([self.params.r_ts_joint, np.min(ts_outer_loop.z)]),
        )

        # Outputs
        if side in ["Inboard"]:
            return split_out[0]
        elif side in ["Outboard"]:
            return split_out[1]

    def build_cts(self, r_cryo_ts: float, z_cryo_ts: float, tk_cryo_ts: float):
        """
        Builds the cryostat thermal shield poloidal 2D cross-section loop

        Parameters
        ----------
        r_cryo_ts: float
            radius of the tin can cryo thermal shield
        z_cryo_ts: float
            height of the tin can cryo thermal shield
        tk_cryo_ts: float
            thickness of the tin can cryo thermal shield
        """
        x_coor = np.zeros(8)
        x_coor[[1, 2]] = r_cryo_ts
        x_coor[[5, 6]] = r_cryo_ts + tk_cryo_ts

        z_coor = np.zeros(8)
        z_coor[[0, 1]] = z_cryo_ts
        z_coor[[2, 3]] = -z_cryo_ts
        z_coor[[4, 5]] = -z_cryo_ts - tk_cryo_ts
        z_coor[[6, 7]] = z_cryo_ts + tk_cryo_ts
        self.geom["Cryostat TS"] = Loop(x=x_coor, z=z_coor)
        self.geom["Cryostat TS"].close()

        self.params.r_cryo_ts = r_cryo_ts
        self.params.z_cryo_ts = z_cryo_ts
        self.params.tk_cryo_ts = tk_cryo_ts

    @property
    def xz_plot_loop_names(self) -> list:
        """
        Selection of the loops to be plotted with plot_xz()

        Returns
        -------
        xz_plot_loop_names: List
            list of the selected loop names
        """
        if "Cryostat TS" in self.geom:
            return ["Inboard profile", "Outboard profile", "Cryostat TS"]
        else:
            return ["Inboard profile", "Outboard profile"]


class LoopBlender:
    """
    Utility class for doing some annoying "blending" of loops pretty specific
    to the thermal shield (or rather, TF coil KOZ).

    The problem is that the offset from the VV to the TF coils is larger at the
    inboard than it needs to be at the top of the VV, meaning that we need
    different offsets, yet we want a smooth surface!

    Parameters
    ----------
    top: Geometry::Loop
        The offset loop from the VV which captures the desired offset at the
        top of the VV
    mid: Geometry::Loop
        The offset loop from the VV which captures the desired offset at the
        inboard midplane of the VV
    """

    DEBUG = False

    def __init__(self, top, mid, debug=False):
        self.top = top
        self.mid = mid
        if debug:
            self.DEBUG = True

    def build(self):
        """
        Builds the blended loop

        Returns
        -------
        koz: Geometry::Loop
            The keep-out-zone for the TF coil, as a result of blending the
            two Loops
        """
        x, z = [0] * len(self.top), [0] * len(self.top)
        ntop, nbot, nmidup, nmiddown = self._findpoints()
        topleft = self._blender(self.top, self.mid, ntop, nmidup)
        bottomleft = self._blender(self.mid, self.top, nbot, nmiddown)
        x[:nmiddown] = self.mid.x[:nmiddown]
        x[nmiddown:nbot] = bottomleft[0]
        x[nbot:ntop] = self.top.x[nbot:ntop]
        x[ntop:nmidup] = topleft[0]
        x[nmidup:] = self.mid.x[nmidup:]
        z[:nmiddown] = self.mid.z[:nbot]
        z[nmiddown:nbot] = bottomleft[1]
        z[nbot:ntop] = self.top.z[nbot:ntop]
        z[ntop:nmidup] = topleft[1]
        z[nmidup:] = self.mid.z[nmidup:]
        return Loop(x=x, z=z)

    def _findpoints(self):
        """
        Finds the points between which blending needs to occur

        Returns
        -------
        ntop: int
            The index of the top blend point
        nbot: int
            The index of the bottom blend point
        nmidup: int
            The index of the upper inner blend point
        nmiddown: int
            The index of the lower inner blend point
        """
        ntop = np.argmax(self.top.z)
        nbot = np.argmin(self.top.z)
        min_r = min(self.top.x)
        idx = [i for i, v in enumerate(self.top.x) if np.isclose(v, min_r)]
        z_values = self.top.z[idx]
        a = np.argmax(z_values)
        b = np.argmin(z_values)
        nmidup = idx[a]
        nmiddown = idx[b]

        if self.DEBUG:
            print(nmidup, nmiddown)
            plt.plot(self.top.x[ntop], self.top.z[ntop], marker="o")
            plt.plot(self.top.x[nbot], self.top.z[nbot], marker="o")
            plt.plot(self.mid.x[nmidup], self.mid.z[nmidup], marker="o")
            plt.plot(self.mid.x[nmiddown], self.mid.z[nmiddown], marker="o")
        return ntop, nbot, nmidup, nmiddown

    def _blender(self, top, mid, nin, nout):
        """
        Carries out the blending between specified indices

        Parameters
        ----------
        top: Geometry::Loop
            The offset loop for the top offset
        mid: Geometry::Loop
            The offset loop for the inboard offset
        nin: int
            The inner blend index
        nout: int
            The outer blend index

        Returns
        -------
        xblend: list
            The blended x coordinates between the two indices
        zblend: list
            The blended z coordinates between the two indices
        """
        xblend, zblend = [], []
        i_l = abs(nin - nout)
        n = [nin, nout]
        for i, j in enumerate(range(min(n), max(n))):
            w1 = 1 - i / i_l
            w2 = i / i_l
            r = w1 * top["x"][j] + w2 * mid["x"][j]
            z = w1 * top["z"][j] + w2 * mid["z"][j]
            if z > max(abs(top["z"])):
                z = max(abs(top["z"]))
            xblend.append(r)
            zblend.append(z)
            if self.DEBUG:
                plt.plot(r, z, marker="o")
        return xblend, zblend


class ThermalShieldPlotter(ReactorSystemPlotter):
    """
    The plotter for a Thermal Shield System.
    """

    def __init__(self):
        super().__init__()
        self._palette_key = "TS"

    def plot_xz(self, plot_objects, ax=None, **kwargs):
        """
        Plot the thermal shield in x-z.
        """
        self._apply_default_styling(kwargs)
        alpha = kwargs["alpha"]
        if not isinstance(alpha, list) and not isinstance(alpha, cycle):
            alpha = float(alpha)
            alpha2 = alpha * 0.5
            kwargs["alpha"] = [alpha2] + [alpha] * (len(plot_objects) - 1)
        super().plot_xz(plot_objects, ax=ax, **kwargs)
