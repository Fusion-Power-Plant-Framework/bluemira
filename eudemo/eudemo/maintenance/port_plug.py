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
Port plugs
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Iterable, Optional, Tuple, Type, Union

if TYPE_CHECKING:
    from bluemira.geometry.solid import BluemiraSolid

from dataclasses import dataclass

import numpy as np

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import boolean_fuse, extrude_shape, offset_wire
from bluemira.geometry.wire import BluemiraWire


def make_castellated_plug(
    face: BluemiraFace,
    vec: Tuple[float, float, float],
    length: float,
    offsets: Union[float, Iterable],
    distances: Optional[Iterable] = None,
    n_castellations: Optional[int] = None,
) -> BluemiraSolid:
    """
    Make a castellated port plug.

    Parameters
    ----------
    face:
        starting profile to be castellated
    vec:
        unit vector along which to extrude
    length:
        total length of castellated BluemiraSolid in vec direction
    offsets:
        castellations offset(s) for each position
    distances:
        (optional) parameter for manually spaced castellations
    n_castellations:
        (optional) parameter for equally spaced castellations

    Returns
    -------
    BluemiraSolid of a castellated port plug
    """
    # Normalise vec
    vec = np.array(vec) / np.linalg.norm(vec)

    # Check/Set-up distances iterable
    if not ((n_castellations is None) or (n_castellations == 0)):
        interval = length / (n_castellations + 1)
        dist_iter = [interval * i for i in range(1, n_castellations + 1)]
    else:
        if distances is not None:
            dist_iter = distances
        else:
            raise ValueError("Both distance and n_castellations parameters are None")

    # Check/Set-up offsets iterable
    if type(offsets) == float:
        off_iter = [offsets] * len(dist_iter)
    else:
        if len(offsets) == len(dist_iter):
            off_iter = offsets
        else:
            raise ValueError("Length of offsets doesn't match distances/n_cast")

    parameter_array = list(zip(dist_iter, off_iter))
    parameter_array.append((length, 0.0))

    base = face
    sections = []
    _prev_dist = 0
    for dist, off in parameter_array:
        ext_vec = vec * (dist - _prev_dist)
        sections.append(extrude_shape(base, ext_vec))
        base.translate(ext_vec)
        base = BluemiraFace(offset_wire(BluemiraWire(base.wires), off))
        _prev_dist = dist

    return boolean_fuse(sections)


@dataclass
class CryostatPortPlugBuilderParams(ParameterFrame):
    """
    Cryostat port plug builder parameters
    """

    plug_gap: Parameter[float]
    n_plug_castellations: Parameter[float]


class CryostatPortPlugBuilder(Builder):
    """
    Cryostat port plug builder.
    """

    param_cls: Type[CryostatPortPlugBuilderParams] = CryostatPortPlugBuilderParams

    def __init__(
        self,
        params: Union[Dict, ParameterFrame, CryostatPortPlugBuilderParams],
        build_config: Optional[Dict],
        outer_profiles: Iterable[BluemiraWire],
        cryostat_xz_boundary: BluemiraFace,
    ):
        super().__init__(params, build_config)
        self.outer_profiles = outer_profiles
        self.cryostat_xz_boundary = cryostat_xz_boundary

    def build(self) -> Component:
        """Build the Equatorial Port"""
