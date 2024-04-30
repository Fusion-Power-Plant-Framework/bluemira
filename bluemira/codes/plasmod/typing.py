# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Types for the plasmod module."""

from typing import Protocol

import numpy as np

from bluemira.codes.interface import BaseRunMode
from bluemira.codes.params import MappedParameterFrame
from bluemira.codes.plasmod.mapping import Profiles


class TransportSolver(Protocol):
    """
    Form for a transport solver function
    """

    params: MappedParameterFrame

    def execute(self, run_mode: str | BaseRunMode) -> MappedParameterFrame:
        """
        Execute the solver function
        """
        ...

    def get_profile(self, profile: str | Profiles) -> np.ndarray:
        """
        Gets the profiles for the solver function
        """
        ...
