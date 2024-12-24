# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""
Testing the rest of the miscellaneous functions related to making a csg environment.
"""

import matplotlib.pyplot as plt
import openmc

from bluemira.codes.openmc.make_csg import OpenMCEnvironment, plot_surfaces


class TestUtilities:
    def test_plot_surfaces(self):
        csg = OpenMCEnvironment
        surfaces = [
            openmc.ZCylinder(r=500, name="cyl"),
            openmc.ZPlane(z0=100, name="pln"),
            openmc.ZCone(z0=100, r2=0.5, name="cone"),
        ]
        plot_surfaces(surfaces)

        plane = csg.surface_from_2points()
        cylinder = csg.surface_from_2points()
        plot_surfaces(surfaces)
        plt.show()
