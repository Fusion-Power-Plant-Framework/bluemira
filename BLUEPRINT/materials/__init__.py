# BLUEPRINT is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2019-2020  M. Coleman, S. McIntosh
#
# BLUEPRINT is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# BLUEPRINT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with BLUEPRINT.  If not, see <https://www.gnu.org/licenses/>.


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
