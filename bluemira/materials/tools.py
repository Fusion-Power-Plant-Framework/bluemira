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
"""Materials tools"""

import json
import warnings
from contextlib import contextmanager


def import_nmm():
    """Don't hack my json, among other annoyances."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        import neutronics_material_maker as nmm  # noqa: PLC0415

        # Really....
        json.JSONEncoder.default = nmm.material._default.default

    return nmm


@contextmanager
def patch_nmm_openmc():
    """Avoid creating openmc material until necessary"""
    nmm = import_nmm()
    if value := nmm.material.OPENMC_AVAILABLE:
        nmm.material.OPENMC_AVAILABLE = False
    try:
        yield
    finally:
        if value:
            nmm.material.OPENMC_AVAILABLE = True
