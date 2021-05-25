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
A script to rebuild the example CLI inputs. Builds an EU-DEMO-like single null tokamak
fusion power reactor and saves the input JSON files to rebuild using the CLI.
"""
from BLUEPRINT.base.file import get_BP_root
from examples.EUDEMO import SingleNullReactor
from examples.EUDEMO import config
from examples.EUDEMO import build_config
from examples.EUDEMO import build_tweaks

R = SingleNullReactor(config, build_config, build_tweaks)
R.config_to_json(f"{get_BP_root()}/examples/cli/indir/")
