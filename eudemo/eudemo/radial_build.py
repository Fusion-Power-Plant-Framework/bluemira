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
"""Functions to optimise an EUDEMO radial build"""

from typing import Dict, TypeVar

from bluemira.base.parameter_frame import ParameterFrame
from bluemira.codes import plot_radial_build, systems_code_solver
from bluemira.codes.process._inputs import ProcessInputs

_PfT = TypeVar("_PfT", bound=ParameterFrame)


CONSTRAINT_EQS = (
    [
        1,  # DESCRIPTION:   Beta Consistency
        # JUSTIFICATION: Consistency equations should always be on
        2,
        5,
        8,
        11,
        13,
        15,
        16,
        24,
        25,
        26,
        27,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        60,
        62,
        65,
        68,
        72,
    ],
)
EUDEMO_PROCESS_INPUTS = ProcessInputs(
    bounds={"k": "v"},
    icc=[
        1,
        2,
        5,
        8,
        11,
        13,
        15,
        16,
        24,
        25,
        26,
        27,
        30,
        31,
        32,
        33,
        34,
        35,
        36,
        60,
        62,
        65,
        68,
        72,
    ],
    ixc=[
        2,
        3,
        4,
        5,
        6,
        9,
        13,
        14,
        16,
        18,
        29,
        36,
        37,
        38,
        39,
        41,
        42,
        44,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        56,
        57,
        58,
        59,
        60,
        61,
        102,
        103,
        106,
        109,
        110,
        113,
        117,
        122,
        123,
    ]
    # fimp = [1.0, 0.1, *([0.0] * 10), 0.00044, 5e-05]
    # ipfloc =[2, 2, 3, 3]
    # ncls=[1, 1, 2, 2]
    # cptdin = [*([42200.0] * 4), *([43000.0] * 4)]
    # rjconpf=[1.1e7, 1.1e7, 6e6, 6e6, 8e6, 8e6, 8e6, 8e6]
    # zref=[3.6, 1.2, 1.0, 2.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
)


def radial_build(params: _PfT, build_config: Dict) -> _PfT:
    """
    Update parameters after a radial build is run/read/mocked using PROCESS.

    Parameters
    ----------
    params:
        Parameters on which to perform the solve (updated)
    build_config:
        Build configuration

    Returns
    -------
    Updated parameters following the solve.
    """
    run_mode = build_config.pop("run_mode", "mock")
    plot = build_config.pop("plot", False)
    if run_mode == "run":
        build_config["template_in_dat"] = EUDEMO_PROCESS_INPUTS
    solver = systems_code_solver(params, build_config)
    new_params = solver.execute(run_mode)

    if plot:
        plot_radial_build(solver.read_directory)
    params.update_from_frame(new_params)
    return params
