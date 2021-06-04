# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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
Module-level functionality for materials.
"""

import os

from BLUEPRINT.base.file import get_BP_path
from BLUEPRINT.base.lookandfeel import bpwarn

from .cache import MaterialCache

materials_cache = MaterialCache()
try:
    material_dir = get_BP_path("materials", subfolder="data")
    material_file = os.sep.join([material_dir, "materials.json"])
    mixture_file = os.sep.join([material_dir, "mixtures.json"])
    materials_cache.load_from_file(material_file)
    materials_cache.load_from_file(mixture_file)
except ValueError:
    bpwarn("Unable to load default materials cache")
    pass
