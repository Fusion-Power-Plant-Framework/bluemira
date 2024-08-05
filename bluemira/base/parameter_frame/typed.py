# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Typing for ParameterFrame"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias, TypeVar

from bluemira.base.parameter_frame._parameter import ParamDictT
from bluemira.base.reactor_config import ConfigParams

if TYPE_CHECKING:
    from bluemira.base.parameter_frame._frame import ParameterFrame


ParameterFrameT = TypeVar("ParameterFrameT", bound="ParameterFrame")
ParameterFrameLike: TypeAlias = (
    dict[str, ParamDictT] | ParameterFrameT | ConfigParams | str | None
)
