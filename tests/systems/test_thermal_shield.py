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
Testing routines for different thermal shield system
"""

import numpy as np
import pytest

from BLUEPRINT.base.parameter import ParameterFrame
from BLUEPRINT.systems.thermalshield import SegmentedThermalShield
from BLUEPRINT.geometry.parameterisations import tapered_picture_frame
from BLUEPRINT.geometry.loop import Loop
from BLUEPRINT.equilibria.coils import Coil


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
            # # Inner corner bot
            # Coil(x=1, z=-4.5, dx=0.5, dz=0.3, ctype="PF", name="PF_1"),
            # # Outboard inner bot
            # Coil(x=2.0, z=-4.5, dx=0.3, dz=0.4, ctype="PF", name="PF_2"),
            # # Outboard inner outer bot
            # Coil(x=3.5, z=-2.0, dx=0.4, dz=0.3, ctype="PF", name="PF_3"),
        ]

        cls.to_ts = {
            "TF inner loop": cls.tf_inner_loop,
            "inner PF coils": cls.pf_coils,
        }

    def test_segmented_ts_build(self):
        thermal_shield = SegmentedThermalShield(self.parameters, self.to_ts)

        # Inboard 2D loop area test
        a_ib_ts = thermal_shield.geom["Inboard profile"].area
        true_a_ib_ts = 0.578999842456656
        assert np.isclose(a_ib_ts, true_a_ib_ts, rtol=1.0e-3)

        # Outboard 2D loop area test
        a_ob_ts = thermal_shield.geom["Outboard profile"].area
        true_a_ob_ts = 0.8055070342087474
        assert np.isclose(a_ob_ts, true_a_ob_ts, rtol=1.0e-3)

    def test_disconneted_ts_build(self):
        self.parameters.tk_ob_ts = 0.15
        thermal_shield = SegmentedThermalShield(self.parameters, self.to_ts)

        # Inboard 2D loop area test
        a_ib_ts = thermal_shield.geom["Inboard profile"].area
        true_a_ib_ts = 0.578999842456656
        assert np.isclose(a_ib_ts, true_a_ib_ts, rtol=1.0e-3)

        # Outboard 2D loop area test
        a_ob_ts = thermal_shield.geom["Outboard profile"].area
        true_a_ob_ts = 2.392958321336142
        assert np.isclose(a_ob_ts, true_a_ob_ts, rtol=1.0e-3)

    def test_ts_wrapping(self):
        thermal_shield = SegmentedThermalShield(self.parameters, self.to_ts)

        # Test if the inner PF coils are wrapped correctly
        # i.e. not inside the thermal shield loop
        for coil in self.pf_coils:
            ts_out = thermal_shield.geom["2D profile"].outer
            assert not ts_out.point_in_poly([coil.x, coil.z])

    def test_pf_koz_raises(self):
        with pytest.raises(NotImplementedError):
            SegmentedThermalShield.build_pf_koz(None, "NotInboard")

    def test_build_vvts_raises(self):
        with pytest.raises(NotImplementedError):
            SegmentedThermalShield.build_vvts_section(
                None, "NotInboard", [], 0.005, 0.005
            )
