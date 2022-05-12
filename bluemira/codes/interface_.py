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
"""Base classes for solvers using external codes."""

import abc
from typing import Dict

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.base.solver import SolverABC, Task


class CodesSolver(SolverABC):
    """
    Base class for solvers running an external code.
    """

    @abc.abstractproperty
    def name(self):
        """
        The name of the solver.

        In the base class, this is used to find mappings and specialise
        error messages for the concrete solver.
        """
        pass

    def modify_mappings(self, send_recv: Dict[str, Dict[str, bool]]):
        """
        Modify the send/receive truth values of a parameter.

        If a parameter's 'send' is set to False, its value will not be
        passed to the external code (a default will be used). Likewise,
        if a parameter's 'recv' is False, its value will not be updated
        from the external code's outputs.

        Parameters
        ----------
        mappings: dict
            A dictionary where keys are variables to change the mappings
            of, and values specify 'send', and or, 'recv' booleans.

            E.g.,

            .. code-block:: python

                {
                    "var1": {"send": False, "recv": True},
                    "var2": {"recv": False}
                }
        """
        for key, val in send_recv.items():
            try:
                p_map = getattr(self.params, key).mapping[self.name]
            except (AttributeError, KeyError):
                bluemira_warn(f"No mapping known for {key} in {self.name}")
            else:
                for sr_key, sr_val in val.items():
                    setattr(p_map, sr_key, sr_val)
