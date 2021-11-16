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
A basic tutorial for configuring and running a design with an optimised TF coil.
"""

from bluemira.base.design import Design


build_config = {
    "TF Coils": {
        "class": "MakeOptimisedTFWindingPack",
        # Parameterise
        "param_class": "PrincetonD",
        "variables_map": {
            "x1": "r_tf_in_centre",
            "x2": {
                "value": "r_tf_out_centre",
                "lower_bound": 8.0,
            },
            "dz": 0.0,
        },
        # Optimise
        "problem_class": "bluemira.builders.tf_coils::MyProblem",
        "algorithm_name": "SLSQP",
        "opt_conditions": {
            "max_eval": 100,
        },
        # Build
        "targets": {
            "TF Coils/xz/Winding Pack": "build_xz",
        },
        "segment_angle": 270.0,
    },
}
params = {
    "r_tf_in_centre": (5.0, "Input"),
    "r_tf_out_centre": (15.0, "Input"),
}
design = Design(params, build_config)
design.run()

for dims in ["xz"]:
    component = design.component_manager.get_by_path(f"TF Coils/{dims}/Winding Pack")

    component.plot_2d()
