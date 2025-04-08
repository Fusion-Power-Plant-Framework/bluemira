# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from pathlib import Path

from bluemira.base.file import get_bluemira_path
from bluemira.materials import MaterialCache

MATERIAL_DATA_PATH = get_bluemira_path("materials", subfolder="data")
MATERIAL_CACHE = MaterialCache()
MATERIAL_CACHE.load_from_file(Path(MATERIAL_DATA_PATH, "materials.json"))
MATERIAL_CACHE.load_from_file(Path(MATERIAL_DATA_PATH, "mixtures.json"))
