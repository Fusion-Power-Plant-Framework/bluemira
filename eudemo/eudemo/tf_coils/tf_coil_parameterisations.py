# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
EU-DEMO parameterisations classes for TF Coils.
"""

import copy
from typing import ClassVar

from bluemira.geometry.parameterisations import PolySpline
from bluemira.utilities.opt_variables import VarDictT


class TFCoilPolySpline(PolySpline):
    """
    Defines the geometry for reactor TF coil, based on the PolySpline
     parameterisation.
    """

    _defaults: ClassVar = {
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

    def __init__(self, var_dict: VarDictT | None = None):
        if var_dict is None:
            var_dict = {}
        defaults = copy.deepcopy(self._defaults)
        defaults.update(var_dict)
        super().__init__(defaults)

        ib_radius = self.variables.x1.value
        ob_radius = self.variables.x2.value
        z2 = self.variables.z2.value
        height = self.variables.height.value
        top = self.variables.top.value
        upper = self.variables.upper.value
        dz = self.variables.dz.value
        tilt = self.variables.tilt.value
        lower = self.variables.lower.value
        bottom = self.variables.bottom.value

        if not self.variables.x1.fixed:
            self.adjust_variable(
                "x1", ib_radius, lower_bound=ib_radius - 2, upper_bound=ib_radius * 1.1
            )
        if not self.variables.x2.fixed:
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

        for var in ["l0s", "l0e", "l1s", "l1e", "l2s", "l2e", "l3s", "l3e"]:
            self.fix_variable(var)
