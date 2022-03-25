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

"""Module to support the fem_fixed_boundary implementation"""

import dolfin
import numpy as np


def b_coil_axis(r, z, pz, curr):
    """
    Return the module of the magnetic field of a coil (of radius r and centered in
    (0, z)) calculated on a point on the coil axis at a distance pz from the
    axis origin.
    """
    return 4 * np.pi * 1e-7 * curr * r**2 / (r**2 + (pz - z) ** 2) ** 1.5 / 2.0


def _convert_const_to_dolfin(value):
    """Convert a constant value to a dolfin function"""
    if isinstance(value, (int, float)):
        return dolfin.Constant(float)
    else:
        raise ValueError("Value must be integer or float.")


class ScalarSubFunc(dolfin.UserExpression):
    """
    Create a dolfin UserExpression from a set of functions defined in the subdomains

    Parameters
    ----------
    func_list: Iterable[int, float, callable]
        list of functions to be interpolated into the subdomains. Int and float values
        are considered as constant functions. Any other callable function must return
        a single value.
    mark_list: Iterable[int]
        list of markers that identify the subdomain in which the respective functions
        of func_list must to be applied.
    subdomains: dolfin.cpp.mesh.MeshFunctionSizet
        the whole subdomains mesh function
    """

    def __init__(self, func_list, mark_list, subdomains, **kwargs):
        super().__init__(**kwargs)
        self.subdomains = subdomains
        self.functions = func_list
        self.markers = mark_list

    def eval_cell(self, values, x, cell):
        """Evaluate the value on each cell"""
        m = self.subdomains[cell.index]
        if m in self.markers:
            index = np.where(np.array(self.markers) == m)
            func = self.functions[index[0][0]]
            if callable(func):
                values[0] = func(x)
            elif isinstance(func, (int, float)):
                values[0] = func
            else:
                raise ValueError(f"{func} is not callable or is not a constant")
        else:
            values[0] = 0

    def value_shape(self):
        """
        Value_shape function (necessary for a UserExpression)
        https://fenicsproject.discourse.group/t/problems-interpolating-a-userexpression-and-plotting-it/1303
        """
        return ()
