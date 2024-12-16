# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import openmc

from bluemira.codes.openmc.make_csg import plot_surfaces


class TestUtilities:
    def test_plot_surfaces(self):
        surfaces = [
            openmc.ZCylinder(r=500, name="cyl"),
            openmc.ZPlane(z0=100, name="pln"),
            openmc.ZCone(z0=100, r2=0.5, name="cone"),
        ]
        plot_surfaces(surfaces)
