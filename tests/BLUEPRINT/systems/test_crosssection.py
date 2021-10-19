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
import matplotlib.pyplot as plt
import os
import pytest

from bluemira.base.parameter import ParameterFrame

from bluemira.base.file import get_bluemira_root
from BLUEPRINT.nova.firstwall import FirstWallProfile
from BLUEPRINT.nova.stream import StreamFlow
from BLUEPRINT.systems.crosssection import ReactorCrossSection

import tests


class TestReactorCrossSection:
    """
    Tests for the ReactorCrossSection class.
    """

    @pytest.mark.parametrize(
        "plasma_type, eq_file, fw_param, vv_param, r_fw_ib_in, r_fw_ob_in, r_vv_ib_in, r_vv_ob_in",
        [
            (
                "SN",
                "tests/BLUEPRINT/test_data/reactors/SMOKE-TEST/equilibria/EU-DEMO_EOF.json",
                "S",
                "S",
                5.8,
                12.1,
                5.1,
                14.5,
            ),
            (
                "DN",
                "tests/bluemira/equilibria/test_data/DN-DEMO_eqref.json",
                "S",
                "S",
                6.9,
                14.0,
                5.1,
                14.5,
            ),
            (
                "DN",
                "tests/bluemira/equilibria/test_data/DN-DEMO_eqref.json",
                "P",
                "P",
                6.8,
                14.2,
                5.0,
                15.7,
            ),
        ],
    )
    def test_xz_plot_loops(
        self,
        plasma_type,
        eq_file,
        fw_param,
        vv_param,
        r_fw_ib_in,
        r_fw_ob_in,
        r_vv_ib_in,
        r_vv_ob_in,
    ):
        """
        Tests that the xz plot loops are populated correctly.
        """
        params = ParameterFrame(ReactorCrossSection.default_params.to_records())
        params.add_parameters(FirstWallProfile.default_params.to_records())
        params.plasma_type = plasma_type

        params.r_fw_ib_in = r_fw_ib_in
        params.r_fw_ob_in = r_fw_ob_in
        params.r_vv_ib_in = r_vv_ib_in
        params.r_vv_ob_in = r_vv_ob_in

        eq_fullpath = os.path.join(get_bluemira_root(), eq_file)
        first_wall = FirstWallProfile(
            params, {"name": params.Name, "parameterisation": fw_param}
        )
        first_wall.generate(
            [eq_fullpath],
            dx=params.tk_sol_ib,
            psi_n=params.fw_psi_n,
            flux_fit=True,
        )
        cross_section = ReactorCrossSection(
            params, {"sf": StreamFlow(eq_fullpath), "VV_parameterisation": vv_param}
        )
        cross_section.params.plasma_type = plasma_type
        cross_section.build(first_wall)
        plot_loops = cross_section._generate_xz_plot_loops()
        assert len(cross_section.xz_plot_loop_names) == len(plot_loops)
        if tests.PLOTTING:
            cross_section.plot_xz()
            plt.show()
