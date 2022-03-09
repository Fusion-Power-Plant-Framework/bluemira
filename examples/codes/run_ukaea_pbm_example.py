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
Test for power balance run
"""

# %%
import bluemira.codes.ukaea_powerbalance as power_balance
from bluemira.base.config import Configuration
from bluemira.base.logs import set_log_level

set_log_level("DEBUG", logger_names=["bluemira", "PowerBalance"])

# %%[markdown]
# # Configure Power Balance Models (UKAEA)

params = Configuration({})

pbm_solver = power_balance.Solver(
    params=params,
    build_config={
        "problem_settings": {},
        "mode": "run",
        "binary": "",
    },
)

# # Run the Solver
pbm_solver.run()
