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
A basic tutorial for configuring and running a design with parameterised shapes.
"""

from bluemira.base.design import Design


build_config = {
    "Plasma": {
        "class": "DesignParameterisedShape",
        "param_class": "bluemira.equilibria.shapes::JohnerLCFS",
        "variables_map": {
            "r_0": "R_0",
            "a": "A",
        },
        "label": "Shape",
    },
    "TF Coils": {
        "class": "DesignParameterisedShape",
        "param_class": "PrincetonD",
        "variables_map": {
            "x1": {
                "value": "r_tf_in_centre",
                "fixed": True,
            },
            "x2": {
                "value": "r_tf_out_centre",
                "lower_bound": 8.0,
            },
            "dz": {
                "value": 0.0,
                "fixed": True,
            },
        },
        "additional_params": ["R_0", "z_0", "B_0", "n_TF", "TF_ripple_limit"],
        "callback": {
            "func": "bluemira.builders.tf_coils::tf_optimisation_callback",
            "args": {"separatrix": "component(Plasma/Shape)"},
        },
        "label": "Shape",
    },
}
params = {
    "R_0": (9.0, "Input"),
    "z_0": (0.0, "Input"),
    "A": (3.5, "Input"),
    "B_0": (6.0, "Input"),
    "n_TF": (16, "Input"),
    "TF_ripple_limit": (0.6, "Input"),
    "r_tf_in_centre": (3.2, "Input"),
    "r_tf_out_centre": (14, "Input"),
}
design = Design(params, build_config)
component = design.run()
component.plot_2d()
