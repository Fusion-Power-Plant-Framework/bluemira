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
Divertor system
"""
from collections import OrderedDict
from typing import Type

import numpy as np

from bluemira.base.parameter import ParameterFrame
from BLUEPRINT.base.error import SystemsError
from BLUEPRINT.cad.divertorCAD import DivertorCAD
from BLUEPRINT.geometry.boolean import boolean_2d_difference
from BLUEPRINT.geometry.geomtools import qrotate
from BLUEPRINT.geometry.loop import Loop, MultiLoop, make_ring
from BLUEPRINT.systems.baseclass import ReactorSystem
from BLUEPRINT.systems.mixins import Meshable
from BLUEPRINT.systems.plotting import ReactorSystemPlotter


class Divertor(Meshable, ReactorSystem):
    """
    Divertor system.
    """

    config: Type[ParameterFrame]
    inputs: dict

    # fmt: off
    default_params = [
        ['n_TF', 'Number of TF coils', 16, 'dimensionless', None, 'Input'],
        ['plasma_type', 'Type of plasma', 'SN', 'dimensionless', None, 'Input'],
        ['n_div_cassettes', 'Number of divertor cassettes per sector', 3, 'dimensionless', None, "Common decision"],
        ['coolant', 'Coolant', 'Water', "dimensionless", 'Divertor coolant type', 'Common sense'],
        ['T_in', 'Coolant inlet T', 80, '°C', 'Coolant inlet T', None],
        ['T_out', 'Coolant outlet T', 120, '°C', 'Coolant inlet T', None],
        ['P_in', 'Coolant inlet P', 8, 'MPa', 'Coolant inlet P', None],
        ['dP', 'Coolant pressure drop', 1, 'MPa', 'Coolant pressure drop', None],
        ['rm_cl', 'RM clearance', 0.02, 'm',
         'Radial and poloidal clearance between in-vessel components',
         'Not so common sense']
    ]
    # fmt: on
    CADConstructor = DivertorCAD

    def __init__(self, config, inputs):
        self.config = config
        self.inputs = inputs
        self._plotter = DivertorPlotter()

        self._init_params(self.config)

        self.n_div = self.params.n_TF * self.params.n_div_cassettes
        if self.params.plasma_type == "DN":
            self.n_div *= 2

        div_geom = self.inputs["geom_info"]

        # Check for DN and correct structures
        if self.params.plasma_type == "DN" and "divertor" not in div_geom:
            profiles = MultiLoop(
                [div_geom["lower"]["divertor"], div_geom["upper"]["divertor"]]
            )
            self.geom["2D profile"] = profiles

        elif self.params.plasma_type == "SN" and "divertor" in div_geom:
            self.geom["2D profile"] = Loop(
                x=div_geom["divertor"]["x"],
                y=-self.params.rm_cl / 2,
                z=div_geom["divertor"]["z"],
            )

        else:
            raise SystemsError("Falsch... was machst du da?!")

        self.build_radial_segments()

    def build_radial_segments(self):
        """
        Prepares profiles and sweep angles for CAD creation
        Coord system: {x: radial, y: toroidal, z: vertical}
        """
        beta = 2 * np.pi / self.params.n_TF
        # TODO: Update with n_div_cassettes
        gamma = beta / 6
        angle = beta / 3

        # centre of z-axis about which divertor rotation must occur
        c2 = np.array([0.5 * self.params.rm_cl / np.sin(gamma), 0, 0])
        c22 = [0.5 * self.params.rm_cl / np.sin(gamma), 0, 1]
        c2_left, c22_left = qrotate([c2, c22], theta=-angle, p1=[0, 0, 0], p2=[0, 0, 1])
        c2_right, c22_right = qrotate([c2, c22], theta=angle, p1=[0, 0, 0], p2=[0, 0, 1])

        gamma = np.rad2deg(gamma)
        angle = np.rad2deg(angle)
        left_profile = self.geom["2D profile"].rotate(
            theta=-gamma, p1=c2, p2=c22, update=False
        )
        right_profile = self.geom["2D profile"].rotate(
            theta=gamma, p1=c2, p2=c22, update=False
        )
        cds = {
            "profile1": left_profile,
            "path": {"angle": angle, "rotation axis": [c2, c22]},
            "profile2": right_profile,
        }

        lds_left = right_profile.rotate(
            theta=-angle, p1=[0, 0, 0], p2=[0, 0, 1], update=False
        )
        lds_right = left_profile.rotate(
            theta=-angle, p1=[0, 0, 0], p2=[0, 0, 1], update=False
        )
        lds = {
            "profile1": lds_left,
            "path": {"angle": -angle, "rotation axis": [c2_left, c22_left]},
            "profile2": lds_right,
        }

        rds_left = left_profile.rotate(
            theta=angle, p1=[0, 0, 0], p2=[0, 0, 1], update=False
        )
        rds_right = right_profile.rotate(
            theta=angle, p1=[0, 0, 0], p2=[0, 0, 1], update=False
        )
        rds = {
            "profile1": rds_left,
            "path": {"angle": angle, "rotation axis": [c2_right, c22_right]},
            "profile2": rds_right,
        }

        self.geom["feed 3D CAD"] = OrderedDict()
        for key, data in zip(["LDS", "CDS", "RDS"], [lds, cds, rds]):
            self.geom["feed 3D CAD"][key] = data

    def get_div_extrema(self, port_angle):
        """
        Get the (lower) divertor extrema.

        Returns
        -------
        extrema: List[Tuple[float]]
            The list of x, z coordinates of the lower divertor extrema.
        """
        div = self._get_lowerdiv()
        return div.get_visible_extrema(port_angle)

    def get_div_height(self, port_angle):
        """
        Get the (lower) divertor y-z height as seen from an angle.

        Returns
        -------
        height: float
            The y-z height of the divertor as seen from an angle.
        """
        div = self._get_lowerdiv()
        return div.get_visible_width(port_angle)

    def get_div_cog(self):
        """
        Get the (lower) divertor centre of gravity.

        Returns
        -------
        centroid: tuple
            The x, z coordinates of the lower divertor centroid.
        """
        div = self._get_lowerdiv()
        return div.centroid

    def _get_lowerdiv(self):
        if self.params.plasma_type == "SN":
            # Just a single null, so return the only divertor there is
            return self.geom["2D profile"]

        # Double null, return the lower divertor
        divs = self.geom["2D profile"].loops
        return sorted(divs, key=lambda x: x.centroid[1])[0]

    def clip_separatrix(self, separatrix):
        """
        Clips the separatrix at the front face of the divertor(s)

        Parameters
        ----------
        separatrix: Loop
            The (open) separatrix Loop

        Returns
        -------
        sep: Loop
            The (open) clipped separatrix Loop
        """
        if self.params.plasma_type == "DN":
            # Should be a list of flux loops
            loops = []
            for part in separatrix.loops:
                for loop in self.geom["2D profile"].loops:
                    part = boolean_2d_difference(part, loop)[0]

                loops.append(part)
            result = MultiLoop(loops)

        else:
            result = boolean_2d_difference(separatrix, self.geom["2D profile"])[0]

        return result

    @property
    def xz_plot_loop_names(self):
        """
        The x-z loop names to plot.
        """
        return ["2D profile"]

    @property
    def xy_plot_loop_names(self):
        """
        The x-y loop names to plot.
        """
        raise NotImplementedError

    def _generate_xy_plot_loops(self):
        gamma = 180 / self.params.n_TF / 1.5
        div = [min(self.geom["2D profile"].x), max(self.geom["2D profile"].x)]
        c1 = self.params.rm_cl / np.sin(np.radians(gamma)) / 2
        div1 = make_ring(div[0], div[1], angle=gamma, centre=(c1, 0))
        div2 = div1.rotate(-gamma, update=False, p1=[-c1, 0, 0], p2=[-c1, 0, 1])
        div3 = div1.rotate(gamma, update=False, p1=[-c1, 0, 0], p2=[-c1, 0, 1])
        return [div1, div2, div3]


class DivertorPlotter(ReactorSystemPlotter):
    """
    The plotter for a Divertor.
    """

    def __init__(self):
        super().__init__()
        self._palette_key = "DIV"

    def plot_xy(self, plot_objects, ax=None, **kwargs):
        """
        Plot the divertor in x-y.
        """
        kwargs["alpha"] = kwargs.get("alpha", 0.3)
        super().plot_xy(plot_objects, ax=ax, **kwargs)
