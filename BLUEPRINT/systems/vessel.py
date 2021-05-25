# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.

"""
Reactor vacuum vessel system
"""
from itertools import cycle
import numpy as np
from typing import Type
from BLUEPRINT.base import ReactorSystem, ParameterFrame
from BLUEPRINT.geometry.boolean import boolean_2d_difference, boolean_2d_union
from BLUEPRINT.cad.vesselCAD import VesselCAD
from BLUEPRINT.geometry.loop import Loop, MultiLoop, make_ring
from BLUEPRINT.geometry.shell import Shell
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

        self.params = ParameterFrame(self.default_params.to_records())
        self.params.update_kw_parameters(self.config)
        self.geom["2D profile"] = self.inputs["vessel_shell"]

        self.up_shift = False

    def build_shells(self):
        """
        Construye los armazónes del recipiente de vací­o
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
