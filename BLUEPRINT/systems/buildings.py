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
Radiation shield system
"""
from itertools import cycle
import numpy as np
from typing import Type

from bluemira.base.parameter import ParameterFrame

from BLUEPRINT.cad.buildingCAD import RadiationCAD
from BLUEPRINT.geometry.loop import Loop, make_ring
from BLUEPRINT.geometry.shell import Shell
from BLUEPRINT.geometry.geombase import Plane
from BLUEPRINT.systems.mixins import Meshable, OnionRing
from BLUEPRINT.systems.baseclass import ReactorSystem
from BLUEPRINT.systems.plotting import ReactorSystemPlotter


class RadiationShield(Meshable, OnionRing, ReactorSystem):
    """
    Radiation Shield reactor system.
    """

    config: Type[ParameterFrame]
    inputs: dict

    # fmt: off
    default_params = [
        ['tk_rs', 'Radiation shield thickness', 2.5, 'm', None, 'Input'],
        ['n_TF', 'Number of TF coils', 16, 'N/A', None, 'Input'],
        ['g_cr_rs', 'Cryostat VV offset to radiation shield', 0.5, 'm', 'Distance away from edge of cryostat VV in all directions', None],
        ['o_p_rs', 'Port offset from VV to RS', 0.25, 'm', None, 'Input'],
        ['n_rs_lab', 'Number of labyrinth levels', 4, 'N/A', None, 'Input'],
        ['rs_l_d', 'Radiation shield labyrinth total delta', 0.6, 'm', 'Thickness of a radiation shield penetration neutron labyrinth', None],
        ['rs_l_gap', 'Radiation shield labyrinth gap', 0.02, 'm', 'Gap between plug and radiation shield', None]
    ]
    # fmt: on
    CADConstructor = RadiationCAD

    def __init__(self, config, inputs):
        self.config = config
        self.inputs = inputs
        self._plotter = RadiationShieldPlotter()

        self._init_params(self.config)

        self.plugs = {}
        self.build_radiation_shield()
        self.build_ports()
        self.build_plugs()
        self.color = (0.5, 0.5, 0.5)

    def build_radiation_shield(self):
        """
        Builds the radiation shield around the main cryostat vacuum vessel
        """
        w = self.inputs["CRplates"]
        end = np.intersect1d(np.where(w.x == 0), np.where(w.z == max(w.z)))[0]
        w = Loop(
            *w[1 : end + 1]
        )  # NOTE: 1: used to be 0: but seemed to cause an issue this one time...
        w = w.offset(self.params.g_cr_rs)
        q = w.offset(self.params.tk_rs)
        shell = Shell(q, w)
        loop = shell.connect_open_loops()
        self.geom["plates"] = loop

    def build_ports(self):
        """
        Builds the ports through the radiation shield
        """
        self.geom["ports"] = {}
        for name, port in self.inputs["VVports"].items():
            p = port.offset(self.params.o_p_rs)
            if "Upper" in name:
                v = [0, 0, max(self.geom["plates"].z) - max(port.z) - self.params.tk_rs]
            else:
                v = [max(self.geom["plates"].x) - max(port.x) - self.params.tk_rs, 0, 0]
            p = p.translate(v, update=False)
            self._bulletproofclock(p)
            self.geom["ports"][name] = p

    def build_plugs(self):
        """
        Build the plugs in the RadiationShield.
        """
        delta = (
            self.params.rs_l_d / self.params.n_rs_lab
        )  # Individual delta labyrinth step
        for pen in self.geom["ports"].keys():
            name = pen.replace(" outer", " plug")
            loop0 = self.geom["ports"][pen]
            loop1 = loop0.offset(0 * -self.params.rs_l_gap)
            if "Upper" in pen:
                v = np.array([0, 0, self.params.tk_rs / self.params.n_rs_lab])
            else:
                v = np.array([self.params.tk_rs / self.params.n_rs_lab, 0, 0])
            loops = [loop1]
            for i in range(int(self.params.n_rs_lab.value) + 1)[1:-1]:
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
        return ["Shield X-Y"]

    def _generate_xy_plot_loops(self):
        plane = Plane([0, 0, 0], [1, np.tan(np.pi / self.params.n_TF), 0], [0, 1, 0])
        inter = self.geom["plates"].section(plane)
        ri = inter[0][0]
        ro = inter[1][0]
        self.geom["Shield X-Y"] = make_ring(ri, ro)
        return super()._generate_xy_plot_loops()


class RadiationShieldPlotter(ReactorSystemPlotter):
    """
    The plotter for a Radiation Shield.
    """

    def __init__(self):
        super().__init__()
        self._palette_key = "RS"

    def plot_xz(self, plot_objects, ax=None, **kwargs):
        """
        Plot the RadiationShield in the x-z plane.
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
