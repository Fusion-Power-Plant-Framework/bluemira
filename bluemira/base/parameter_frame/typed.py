# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Typing for ParameterFrame"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, TypeVar, Union

from bluemira.base.parameter_frame._parameter import ParamDictT

if TYPE_CHECKING:
    from bluemira.base.parameter_frame._frame import ParameterFrame
    from bluemira.base.reactor_config import ConfigParams


ParameterFrameT = TypeVar("ParameterFrameT", bound="ParameterFrame")
ParameterFrameLike: TypeAlias = Union[
    dict[str, ParamDictT], ParameterFrameT, "ConfigParams", str, None
]

ParameterFrameOrNoneT = TypeVar(
    "ParameterFrameOrNoneT", bound=Union["ParameterFrame", None]
)
