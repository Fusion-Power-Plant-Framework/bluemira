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
EUDEMO Lower Port Paramteter Frames
"""

from dataclasses import dataclass

from bluemira.base.parameter_frame import Parameter, ParameterFrame


@dataclass
class LowerPortDesignerParams(ParameterFrame):
    """LowerPort Designer ParameterFrame"""

    tf_coil_thickness: Parameter[float]
    n_TF: Parameter[int]
    n_div_cassettes: Parameter[int]

    lower_duct_angle: Parameter[float]
    lower_duct_tf_offset: Parameter[float]
    lower_duct_div_pad_dx_outer: Parameter[float]
    lower_duct_div_pad_dx_inner: Parameter[float]

    lower_duct_wall_tk: Parameter[float]

    lower_port_height: Parameter[float]
    lower_port_width: Parameter[float]


@dataclass
class LowerPortBuilderParams(ParameterFrame):
    """LowerPort ParameterFrame"""

    lower_duct_wall_tk: Parameter[float]
    lower_duct_angle: Parameter[float]
    n_TF: Parameter[int]
