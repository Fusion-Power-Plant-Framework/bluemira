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
Perform the EU-DEMO reactor design.
"""

import matplotlib.pyplot as plt
import os

from bluemira.builders.EUDEMO.config import params, build_config
from bluemira.builders.EUDEMO.reactor import EUDEMO

from bluemira.codes import plot_PROCESS

# If you have PROCESS installed then change these to enable PROCESS runs.
# build_config["process_mode"] = "run"
# build_config["process_mode"] = "read"

reactor = EUDEMO(params, build_config)
reactor.run()

if build_config["process_mode"] == "run":
    plot_PROCESS(
        os.path.join(reactor.file_manager.generated_data_dirs["systems_code"], "OUT.DAT")
    )
    plt.show()
