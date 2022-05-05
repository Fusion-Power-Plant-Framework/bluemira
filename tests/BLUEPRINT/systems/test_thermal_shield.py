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
Testing routines for different thermal shield system
"""

import numpy as np
import pytest

from bluemira.base.parameter import ParameterFrame
from bluemira.equilibria.coils import Coil
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.geometry.parameterisations import tapered_picture_frame
from BLUEPRINT.systems.thermalshield import SegmentedThermalShield


class TestSegmentedThermalShield:
    @classmethod
    def setup_class(cls):
        # fmt: off
        params = [
            ['tk_ib_ts', 'Inboard TS thickness', 0.05, 'm', None, 'Input'],
            ['tk_ob_ts', 'Outboard TS thickness', 0.05, 'm', None, 'Input'],
            ['g_ib_ts_tf', 'Inboard gap between TS and TF', 0.05, 'm', None, 'Input'],
            ['g_ob_ts_tf', 'Outboard gap between TS and TF', 0.05, 'm', None, 'Input'],
            ['g_ts_pf', 'Clearances to PFs', 0.075, 'm', None, 'Input'],
            ['r_ts_joint', 'Radius of inboard/outboard TS joint', 2. , 'm', None, 'Input'],
            ['tk_cryo_ts', 'Cryo TS thickness', 0.10, 'm', None, 'Input'],
            ['r_cryo_ts', 'Radius of outboard cryo TS', 11.0, 'm', None, 'Input'],
            ['z_cryo_ts', 'Half height of outboard cryo TS', 14.0, 'm', None, 'Input'],
        ]
        # fmt: on
        cls.parameters = ParameterFrame(params)

        # Make a picture frame coil shape to emulate the TF coil boundary
        x_tf_loop, z_tf_loop = tapered_picture_frame(
            x1=0.5,
            x2=0.6,
            x3=4.0,
            z1_frac=0.5,
            z2=4.0,
            z3=5.0,
            r=0.5,
        )
        cls.tf_inner_loop = Loop(x=x_tf_loop, z=z_tf_loop)

        # Make the inner PF coil list with a outer one
        cls.pf_coils = [
            # Inner corner
            Coil(x=1, z=4.5, dx=0.5, dz=0.3, ctype="PF", name="PF_1"),
            # Outboard inner top
            Coil(x=2.0, z=4.5, dx=0.3, dz=0.4, ctype="PF", name="PF_2"),
            # Outboard inner outer,
            Coil(x=3.5, z=2.0, dx=0.4, dz=0.3, ctype="PF", name="PF_3"),
            # Second outboard inner outer top
            Coil(x=3.5, z=2.0, dx=0.2, dz=0.3, ctype="PF", name="PF_4"),
            # Second outboard inner outer bot
            Coil(x=3.5, z=2.0, dx=0.2, dz=0.3, ctype="PF", name="PF_5"),
            # Inner corner bot
            Coil(x=1, z=-4.5, dx=0.5, dz=0.3, ctype="PF", name="PF_6"),
            # Outboard inner bot
            Coil(x=2.0, z=-4.5, dx=0.3, dz=0.4, ctype="PF", name="PF_7"),
            # Outboard inner outer bot
            Coil(x=3.5, z=-2.0, dx=0.4, dz=0.3, ctype="PF", name="PF_8"),
        ]

        cls.to_ts = {
            "TF inner loop": cls.tf_inner_loop,
            "inner PF coils": cls.pf_coils,
            "PF wrapping": {
                "PF_1": "L",
                "PF_2": "top bot",
                "PF_3": "U",
                "PF_4": "vertical gap",
                "PF_5": "vertical gap",
                "PF_6": "U",
                "PF_7": "top bot",
                "PF_8": "L",
            },
        }

    def test_segmented_ts_build(self):
        thermal_shield = SegmentedThermalShield(self.parameters, self.to_ts)

        # Inboard 2D loop area test
        a_ib_ts = thermal_shield.geom["Inboard profile"].area
        true_a_ib_ts = 0.7046248428361714
        assert np.isclose(a_ib_ts, true_a_ib_ts, rtol=1.0e-3)

        # Outboard 2D loop area test
        a_ob_ts = thermal_shield.geom["Outboard profile"].area
        true_a_ob_ts = 0.725000002956949
        assert np.isclose(a_ob_ts, true_a_ob_ts, rtol=1.0e-3)

        # Test if the merge shell is properly obtained
        ts_out = thermal_shield.geom["2D profile"].outer
        ts_in = thermal_shield.geom["2D profile"].inner
        assert ts_out.point_inside([2.0, 0.0])
        assert ts_in.point_inside([2.0, 0.0])

    def test_disconneted_ts_build(self):
        self.parameters.tk_ob_ts = 0.15
        thermal_shield = SegmentedThermalShield(self.parameters, self.to_ts)

        # Inboard 2D loop area test
        a_ib_ts = thermal_shield.geom["Inboard profile"].area
        true_a_ib_ts = 0.7046248428361714
        assert np.isclose(a_ib_ts, true_a_ib_ts, rtol=1.0e-3)

        # Outboard 2D loop area test
        a_ob_ts = thermal_shield.geom["Outboard profile"].area
        true_a_ob_ts = 2.1450000039255244
        assert np.isclose(a_ob_ts, true_a_ob_ts, rtol=1.0e-3)

        # Test if the merge shell is properly obtained
        ts_out = thermal_shield.geom["2D profile"].outer
        ts_in = thermal_shield.geom["2D profile"].inner
        assert ts_out.point_inside([2.0, 0.0])
        assert ts_in.point_inside([2.0, 0.0])

    def test_ts_wrapping(self):
        thermal_shield = SegmentedThermalShield(self.parameters, self.to_ts)

        # Test if the inner PF coils are wrapped correctly
        # i.e. not inside the thermal shield loop
        ts_out = thermal_shield.geom["2D profile"].outer
        for coil in self.pf_coils:
            assert not ts_out.point_inside([coil.x, coil.z])

    def test_pf_koz_raises(self):
        with pytest.raises(NotImplementedError):
            SegmentedThermalShield.build_pf_koz(None, "NotInboard")

    def test_build_vvts_raises(self):
        with pytest.raises(NotImplementedError):
            SegmentedThermalShield.build_vvts_section(
                None, "NotInboard", [], 0.005, 0.005
            )

    def test_build_cts(self):
        cryo_ts = SegmentedThermalShield(self.parameters, self.to_ts)
        cryo_ts.build_cts(
            self.parameters.r_cryo_ts,
            self.parameters.z_cryo_ts,
            self.parameters.tk_cryo_ts,
        )
        a_cryo_ts = cryo_ts.geom["Cryostat TS"].area
        true_a_cryo_ts = 5.019999999999982
        assert np.isclose(a_cryo_ts, true_a_cryo_ts, rtol=1.0e-3)

    def test_u_wrapping(self):
        to_ts_u = self.to_ts
        to_ts_u["PF wrapping"] = "U"
        thermal_shield = SegmentedThermalShield(self.parameters, to_ts_u)

        # Inboard 2D loop area test
        a_ib_ts = thermal_shield.geom["Inboard profile"].area
        true_a_ib_ts = 0.7739998425637591
        assert np.isclose(a_ib_ts, true_a_ib_ts, rtol=1.0e-3)

        # Outboard 2D loop area test
        a_ob_ts = thermal_shield.geom["Outboard profile"].area
        true_a_ob_ts = 2.809208321424613
        assert np.isclose(a_ob_ts, true_a_ob_ts, rtol=1.0e-3)

    def test_vertical_gap_wrapping(self):
        to_ts_u = self.to_ts
        to_ts_u["PF wrapping"] = "vertical gap"
        pf_coils_u = [
            # Inner corner
            Coil(x=1, z=4.5, dx=0.5, dz=0.3, ctype="PF", name="PF_1"),
            # Second outboard inner outer top
            Coil(x=3.5, z=2.0, dx=0.2, dz=0.3, ctype="PF", name="PF_2"),
        ]
        to_ts_u["inner PF coils"] = pf_coils_u
        thermal_shield = SegmentedThermalShield(self.parameters, to_ts_u)

        # Inboard 2D loop area test
        a_ib_ts = thermal_shield.geom["Inboard profile"].area
        true_a_ib_ts = 0.5324999983841554
        assert np.isclose(a_ib_ts, true_a_ib_ts, rtol=1.0e-3)

        # Outboard 2D loop area test
        a_ob_ts = thermal_shield.geom["Outboard profile"].area
        true_a_ob_ts = 1.8075000041816427
        assert np.isclose(a_ob_ts, true_a_ob_ts, rtol=1.0e-3)
