# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                    J. Morris, D. Short
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
Some examples of using bluemira geometry objects.
"""

import bluemira.geometry as geo

# %% [markdown]

## Introduction

# Geometry is not plasma physics, but it isn't trivial either. Chances are most of
# your day-to-day interaction with bluemira will revolve around geometry in some form
# or another. Puns intended.

# There a few basic concepts you need to familiarise yourself with:
# * Basic types: [BluemiraWire, BluemiraFace, BluemiraShell, BluemiraSolid]
# * Matryoshka structure
# * Geometry creation
# * Geometry operations
