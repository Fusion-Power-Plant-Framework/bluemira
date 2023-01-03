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

import numpy as np

from bluemira.equilibria.coils import Coil, CoilSet


def _make_square(mn=None, mx=None):
    if mn is None:
        mn = {"x": 3, "z": 2}
    if mx is None:
        mx = {"x": 12, "z": 4}
    x = np.array([mn["x"], mx["x"], mx["x"], mn["x"], mn["x"]])
    z = np.array([mn["z"], mn["z"], mx["z"], mx["z"], mn["z"]])

    return {"x": x, "z": z}


def _make_star():
    # Abstract shape
    x = np.array([0, 1, 2, 1.5, 3, 1.5, 1, 0.5, -1.5, 0.5, 0])
    z = np.array([0, 1, 0, 1, 2, 2, 4, 2, 2, 1, 0])

    return {"x": x, "z": z}


def _coilset_setup(self, materials=False):
    # EU DEMO 2015
    x = [5.4, 14.0, 17.0, 17.01, 14.4, 7.0, 2.9, 2.9, 2.9, 2.9, 2.9]
    z = [
        8.82,
        7.0,
        2.5,
        -2.5,
        -8.4,
        -10.45,
        6.6574,
        3.7503,
        -0.6105,
        -4.9713,
        -7.8784,
    ]

    dx = [0.6, 0.4, 0.5, 0.5, 0.7, 1.0, 0.4, 0.4, 0.4, 0.4, 0.4]
    dz = [0.6, 0.4, 0.5, 0.5, 0.7, 1.0, 1.4036, 1.4036, 2.85715, 1.4036, 1.4036]

    names = [
        "PF_1",
        "PF_2",
        "PF_3",
        "PF_4",
        "PF_5",
        "PF_6",
        "CS_1",
        "CS_2",
        "CS_3",
        "CS_4",
        "CS_5",
    ]

    coils = []
    for name, xc, zc, dxc, dzc in zip(names, x, z, dx, dz):
        ctype = name[:2]
        coil = Coil(xc, zc, dx=dxc, dz=dzc, name=name, ctype=ctype)
        coils.append(coil)
    self.coilset = CoilSet(*coils)

    if materials:
        # Max PF currents / sizes don't stack up in the CREATE document...
        self.coilset.assign_coil_materials("PF", j_max=12.5, b_max=12.5)
        self.coilset.assign_coil_materials("CS", j_max=12.5, b_max=12.5)
        self.coilset.fix_sizes()

    self.no_coils = len(names)
    self.coil_names = names
