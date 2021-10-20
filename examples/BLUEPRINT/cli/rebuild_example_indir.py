# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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
A script to rebuild the example CLI inputs. Builds an EU-DEMO-like single null tokamak
fusion power reactor and saves the input JSON files to rebuild using the CLI.
"""
from BLUEPRINT.base.file import get_bluemira_root
from examples.EUDEMO import SingleNullReactor
from examples.EUDEMO import config
from examples.EUDEMO import build_config
from examples.EUDEMO import build_tweaks

R = SingleNullReactor(config, build_config, build_tweaks)
R.config_to_json(f"{get_bluemira_root()}/examples/cli/indir/")
