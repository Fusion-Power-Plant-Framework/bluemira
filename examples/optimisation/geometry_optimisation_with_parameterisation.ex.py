# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
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

"""Geometry Optimisation With a Geometry Parameterisation"""

# %%
from typing import Dict, Optional, Union

from bluemira.display import plot_2d
from bluemira.geometry.parameterisations import GeometryParameterisation
from bluemira.geometry.tools import make_circle
from bluemira.geometry.wire import BluemiraWire
from bluemira.optimisation import optimise_geometry
from bluemira.utilities.opt_variables import BoundedVariable, OptVariables


class Circle(GeometryParameterisation):
    """Geometry parameterisation for a circle in the xz-plane."""

    def __init__(
        self, var_dict: Optional[Dict[str, Union[float, Dict[str, float]]]] = None
    ):
        opt_variables = OptVariables(
            [
                BoundedVariable("radius", 10, 1e-5, 15),
                BoundedVariable("centre_x", 0, -10, 10),
                BoundedVariable("centre_z", 0, 0, 10),
            ]
        )
        opt_variables.adjust_variables(var_dict, strict_bounds=False)
        super().__init__(opt_variables)

    def create_shape(self, label: str = "") -> BluemiraWire:
        """Create the circle."""
        return make_circle(
            self.variables["radius"].value,
            center=(
                self.variables["centre_x"].value,
                0,
                self.variables["centre_z"].value,
            ),
            axis=(0, 1, 0),
            label=label,
        )


zone = make_circle(10, start_angle=180, end_angle=0, axis=(0, 1, 0))
zone.close()

# Now lets create our circle within the shape
circle = Circle(
    {"radius": {"value": 0.5}, "centre_x": {"value": -5}, "centre_z": {"value": 1}}
)

plot_2d([circle.create_shape(), zone])


def objective(geom: GeometryParameterisation) -> float:
    """Objective function for maximising the perimeter of a circle."""
    return -geom.create_shape().length


result = optimise_geometry(
    geom=circle,
    f_objective=objective,
    keep_out_zones=[zone],
    algorithm="COBYLA",
    opt_conditions={"xtol_rel": 1e-12, "max_eval": 2000},
    koz_discretisation=200,
)
print(result)

plot_2d([result.geom.create_shape(), zone])
