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
Builder for making a parameterised EU-DEMO vacuum vessel.
"""

import bluemira.utilities.plot_tools as bm_plot_tools
from bluemira.base.builder import BuildConfig, Builder
from bluemira.base.components import Component, PhysicalComponent


class VacuumVesselBuilder(Builder):
    """
    Builder for the vacuum vessel
    """

    _required_params: List[str] = [
        "tk_ts",
        "g_ts_pf",
        "g_ts_tf",
        "g_vv_ts",
        "n_TF",
    ]
    _params: Configuration
    _pf_kozs: List[BluemiraWire]
    _tf_koz: BluemiraWire
    _vv_koz: Optional[BluemiraWire]
    _cts_face: BluemiraFace

    def __init__(
        self,
        params,
        build_config: BuildConfig,
        pf_coils_xz_kozs: List[BluemiraWire],
        tf_xz_koz: BluemiraWire,
        vv_xz_koz: Optional[BluemiraWire] = None,
    ):
        super().__init__(
            params,
            build_config,
            pf_coils_xz_kozs=pf_coils_xz_kozs,
            tf_xz_koz=tf_xz_koz,
            vv_xz_koz=vv_xz_koz,
        )
        self._cts_face = None
