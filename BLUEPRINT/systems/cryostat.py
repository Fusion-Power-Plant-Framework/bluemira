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
Cryostat system
"""
from itertools import cycle
import numpy as np
from shapely.ops import cascaded_union
from typing import Type

from bluemira.base.parameter import ParameterFrame

from BLUEPRINT.cad.cryostatCAD import CryostatCAD
from BLUEPRINT.geometry.loop import Loop, make_ring
from BLUEPRINT.geometry.geombase import Plane
from BLUEPRINT.systems.mixins import Meshable, OnionRing
from BLUEPRINT.systems.baseclass import ReactorSystem
from BLUEPRINT.systems.plotting import ReactorSystemPlotter


class Cryostat(Meshable, OnionRing, ReactorSystem):
    """
    Cryostat reactor system.
    """

    config: Type[ParameterFrame]
    inputs: dict
    # fmt: off
    default_params = [
        ['tk_cr_vv', 'Cryostat VV thickness', 0.3, 'm', None, 'Input'],
        ['n_TF', 'Number of TF coils', 16, 'N/A', None, 'Input'],
        ['g_cr_ts', 'Gap between the Cryostat and CTS', 0.3, 'm', None, 'Input'],
        ['o_p_cr', 'Port offset from VV to CR', 0.1, 'm', None, 'Input'],
        ['n_cr_lab', 'Number of cryostat labyrinth levels', 2, 'N/A', None, 'Input'],
        ['cr_l_d', 'Cryostat labyrinth total delta', 0.2, 'm', None, 'Input'],
    ]
    # fmt: on
    CADConstructor = CryostatCAD

    def __init__(self, config, inputs):
        self.config = config
        self.inputs = inputs
        self._plotter = CryostatPlotter()

        self._init_params(self.config)

        self.plugs = {}
        self.build_cyrostatvv()
        self.build_ports()
        self.build_plugs()

    def build_cyrostatvv(self):
        """
        Build the cryostat vacuum vessel main body as an offset from the
        thermal shield.
        """
        tk = self.params.tk_cr_vv
        z_gs = self.inputs["GS"]["zground"]
        x_gs = self.inputs["GS"]["Xo"]
        x_out = max(self.inputs["CTS"]["x"]) + self.params.g_cr_ts

        z_lid = max(self.inputs["CTS"]["z"]) + self.params.g_cr_ts
        z_base = z_gs - 0.1
        outer_can = Loop(
            x=[x_out, x_out + tk, x_out + tk, x_out, x_out],
            y=0,
            z=[z_base - tk, z_base - tk, z_lid + tk, z_lid + tk, z_base - tk],
        )
        x_ic = x_gs - 2
        base = Loop(
            x=[x_ic, x_out + tk, x_out + tk, x_ic, x_ic],
            y=0,
            z=[z_base - tk, z_base - tk, z_base, z_base, z_base - tk],
        )
        depth = 5
        inner_can = Loop(
            x=[x_ic, x_ic + tk, x_ic + tk, x_ic, x_ic],
            y=0,
            z=[z_base - depth, z_base - depth, z_base, z_base, z_base - depth],
        )
        floor = Loop(
            x=[0, x_ic + tk, x_ic + tk, 0, 0],
            y=0,
            z=[
                z_base - depth - tk,
                z_base - depth - tk,
                z_base - depth,
                z_base - depth,
                z_base - depth - tk,
            ],
        )

        roof = Loop(
            x=[0, x_out + tk, x_out + tk, 0, 0],
            y=0,
            z=[z_lid, z_lid, z_lid + tk, z_lid + tk, z_lid],
        )
        loops = [outer_can, base, inner_can, floor, roof]
        full_union = cascaded_union([loop.as_shpoly() for loop in loops])
        loop = Loop(**dict(zip(["x", "z"], full_union.exterior.xy)))
        # Simplify numerical order for radiation shield
        loop.reorder(len(loop) - np.argmin(loop.x + loop.z))
        self.geom["plates"] = loop

    def build_ports(self):
        """
        Build the ports in the cryostat vacuum vessel.
        """
        self.geom["ports"] = {}
        for name, port in self.inputs["VVports"].items():
            p = port.offset(self.params.o_p_cr)
            if "Upper" in name:
                v = [
                    0,
                    0,
                    max(self.geom["plates"].z) - max(port.z) - self.params.tk_cr_vv,
                ]
            else:
                v = [
                    max(self.geom["plates"].x) - max(port.x) - self.params.tk_cr_vv,
                    0,
                    0,
                ]
            p.translate(v)
            self._bulletproofclock(p)
            self.geom["ports"][name] = p

    def build_plugs(self):  # TODO: DRY OnionRing
        """
        Build the cryostat port plugs.
        """
        delta = (
            self.params.cr_l_d / self.params.n_cr_lab
        )  # Individual delta labyrinth step
        for pen in self.geom["ports"].keys():
            name = pen.replace(" outer", " plug")
            loop0 = self.geom["ports"][pen]
            if "Upper" in pen:
                v = np.array([0, 0, self.params.tk_cr_vv / self.params.n_cr_lab])
            else:
                v = np.array([self.params.tk_cr_vv / self.params.n_cr_lab, 0, 0])

            loops = [loop0]
            for _ in range(int(self.params.n_cr_lab.value))[1:]:
                new_loop = loops[-1].translate(v, update=False)
                loops.append(new_loop)
                new_loop = loops[-1].offset(delta)
                loops.append(new_loop)
            loops.append(loops[-1].translate(v, update=False))
            self.plugs[name] = loops

    @property
    def xz_plot_loop_names(self):
        """
        The x-z loop names to plot.
        """
        return ["plates"] + list(self.plugs.keys())

    @property
    def xy_plot_loop_names(self):
        """
        The x-y loop names to plot.
        """
        return ["Cryostat VV X-Y"]

    def _generate_xy_plot_loops(self):
        plane = Plane([0, 0, 0], [1, 0, 0], [0, 1, 0])
        inter = self.geom["plates"].section(plane)
        ri = inter[0][0]
        ro = inter[1][0]
        self.geom["Cryostat VV X-Y"] = make_ring(ri, ro)
        return super()._generate_xy_plot_loops()


class CryostatPlotter(ReactorSystemPlotter):
    """
    The plotter for a Cryostat.
    """

    def __init__(self):
        super().__init__()
        self._palette_key = "CR"

    def plot_xz(self, plot_objects, ax=None, **kwargs):
        """
        Plot the Cryostat in the x-z plane.
        """
        self._apply_default_styling(kwargs)
        facecolor = kwargs["facecolor"]
        alpha = kwargs["alpha"]
        if not isinstance(facecolor, list) and not isinstance(facecolor, cycle):
            kwargs["facecolor"] = [facecolor] + ["w"] * (len(plot_objects) - 1)
        if not isinstance(alpha, list) and not isinstance(alpha, cycle):
            kwargs["alpha"] = [alpha] + [alpha / 2] * (len(plot_objects) - 1)
        super().plot_xz(plot_objects, ax=ax, **kwargs)


if __name__ == "__main__":
    from BLUEPRINT import test

    test()
