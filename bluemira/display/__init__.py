# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Display and plotting module
"""

from bluemira.display.auto_config import plot_defaults
from bluemira.display.displayer import show_cad
from bluemira.display.plotter import plot_2d, plot_3d

__all__ = ["plot_2d", "plot_3d", "show_cad"]

plot_defaults()
