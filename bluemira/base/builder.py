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
Interfaces for builder and build steps classes
"""

from __future__ import annotations

import abc
import enum
import string
from typing import Any, Dict, List, Literal, Optional, Union

from bluemira.base.components import Component
from bluemira.base.config import Configuration
from bluemira.base.error import BuilderError
from bluemira.base.look_and_feel import bluemira_debug, bluemira_print
from bluemira.base.parameter import ParameterFrame

__all__ = ["Builder", "BuildConfig"]


BuildConfig = Dict[str, Union[int, float, str, "BuildConfig"]]
"""
Type alias for representing nested build configuration information.
"""


# TODO: Consolidate with RunMode in codes.
class RunMode(enum.Enum):
    """
    Enum class to pass args and kwargs to the function corresponding to the chosen
    PROCESS runmode (Run, Read, or Mock).
    """

    RUN = enum.auto()
    READ = enum.auto()
    MOCK = enum.auto()

    def __call__(self, obj, *args, **kwargs):
        """
        Call function of object with lowercase name of enum

        Parameters
        ----------
        obj: instance
            instance of class the function will come from
        *args
           args of function
        **kwargs
           kwargs of function

        Returns
        -------
        function result
        """
        func = getattr(obj, self.name.lower())
        return func(*args, **kwargs)
