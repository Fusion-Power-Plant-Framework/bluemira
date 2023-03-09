# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
EU-DEMO parameterisations classes for TF Coils.
"""
import copy
from typing import Dict

from bluemira.geometry.parameterisations import PolySpline
from bluemira.utilities.opt_variables import BoundedVariable, OptVariables


class TFCoilPolySpline(PolySpline):
    """
    Defines the geometry for reactor TF coil, based on the PolySpline
     parameterisation.
    """

    _defaults = {
        "x1": {"value": 3.8},
        "x2": {"value": 16},
        "z2": {"value": 0},
        "height": {"value": 14},
        "top": {"value": 0.4},
        "upper": {"value": 0.3},
        "dz": {"value": 0},
        "tilt": {"value": 0},
        "lower": {"value": 0.5},
        "bottom": {"value": 0.2},
        "flat": {"value": 0.0},
    }

    def __init__(self, var_dict: Dict = None):
        variables = OptVariables(
            [
                BoundedVariable(
                    "x1", 4.3, lower_bound=4, upper_bound=5, descr="Inner limb radius"
                ),
                BoundedVariable(
                    "x2", 16.56, lower_bound=5, upper_bound=25, descr="Outer limb radius"
                ),
                BoundedVariable(
                    "z2",
                    0.03,
                    lower_bound=-2,
                    upper_bound=2,
                    descr="Outer note vertical shift",
                ),
                BoundedVariable(
                    "height", 15.5, lower_bound=10, upper_bound=50, descr="Full height"
                ),
                BoundedVariable(
                    "top", 0.52, lower_bound=0.2, upper_bound=1, descr="Horizontal shift"
                ),
                BoundedVariable(
                    "upper", 0.67, lower_bound=0.2, upper_bound=1, descr="Vertical shift"
                ),
                BoundedVariable(
                    "dz", -0.6, lower_bound=-5, upper_bound=5, descr="Vertical offset"
                ),
                BoundedVariable(
                    "flat",
                    0,
                    lower_bound=0,
                    upper_bound=1,
                    descr="Fraction of straight outboard leg",
                ),
                BoundedVariable(
                    "tilt",
                    4,
                    lower_bound=-45,
                    upper_bound=45,
                    descr="Outboard angle [degrees]",
                ),
                BoundedVariable(
                    "bottom",
                    0.4,
                    lower_bound=0,
                    upper_bound=1,
                    descr="Lower horizontal shift",
                ),
                BoundedVariable(
                    "lower",
                    0.67,
                    lower_bound=0.2,
                    upper_bound=1,
                    descr="Lower vertical shift",
                ),
                BoundedVariable(
                    "tension",
                    0.8,
                    lower_bound=0.1,
                    upper_bound=1.9,
                    descr="Tension variable for all segments",
                ),
            ],
            frozen=True,
        )
        variables.adjust_variables(var_dict, strict_bounds=False)

        super().__init__(variables)

        if var_dict is None:
            var_dict = {}
        defaults = copy.deepcopy(self._defaults)
        defaults.update(var_dict)
        super().__init__(defaults)

        ib_radius = self.variables["x1"].value
        ob_radius = self.variables["x2"].value
        z2 = self.variables["z2"].value
        height = self.variables["height"].value
        top = self.variables["top"].value
        upper = self.variables["upper"].value
        dz = self.variables["dz"].value
        tilt = self.variables["tilt"].value
        lower = self.variables["lower"].value
        bottom = self.variables["bottom"].value

        if not self.variables["x1"].fixed:
            self.adjust_variable(
                "x1",
                ib_radius,
                lower_bound=ib_radius - 2,
                upper_bound=ib_radius * 1.1,
            )
        if not self.variables["x2"].fixed:
            self.adjust_variable(
                "x2",
                value=ob_radius,
                lower_bound=ob_radius * 0.9,
                upper_bound=ob_radius + 2,
            )
        self.adjust_variable("z2", z2, lower_bound=-0.9, upper_bound=0.9)
        self.adjust_variable(
            "height", height, lower_bound=height - 0.001, upper_bound=50
        )
        self.adjust_variable("top", top, lower_bound=0.05, upper_bound=0.75)
        self.adjust_variable("upper", upper, lower_bound=0.2, upper_bound=0.7)
        self.adjust_variable("dz", dz, lower_bound=-5, upper_bound=5)
        self.adjust_variable("tilt", tilt, lower_bound=-25, upper_bound=25)
        self.adjust_variable("lower", lower, lower_bound=0.2, upper_bound=0.7)
        self.adjust_variable("bottom", bottom, lower_bound=0.05, upper_bound=0.75)

        # Fix 'flat' to avoid drawing the PolySpline's outer straight.
        # The straight is often optimised to near-zero length, which
        # causes an error when CAD tries to draw it
        self.fix_variable("flat", 0)

    def adjust_variable(self, name, value=None, lower_bound=None, upper_bound=None):
        if name != "tension":
            return super().adjust_variable(name, value, lower_bound, upper_bound)

    def _get_variable_values(self):
        variables = self.variables.values
        (
            x1,
            x2,
            z2,
            height,
            top,
            upper,
            dz,
            flat,
            tilt,
            bottom,
            lower,
        ) = variables[:11]
        l_start = 4 * [variables[11]]
        l_end = 4 * [variables[11]]
        return (
            x1,
            x2,
            z2,
            height,
            top,
            upper,
            dz,
            flat,
            tilt,
            bottom,
            lower,
            l_start,
            l_end,
        )
