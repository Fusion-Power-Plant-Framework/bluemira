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

from BLUEPRINT.base.parameter import ParameterFrame
from BLUEPRINT.systems.vessel import SegmentedVaccumVessel
from BLUEPRINT.geometry.loop import Loop


class TestSegmentedVaccumVessel:
    @classmethod
    def setup_class(cls):
        # fmt: off
        params = [
            ['n_TF', 'Number of TF coils', 16, 'N/A', None, 'Input'],
            ['tk_vv_in', 'Inboard VV thickness', 0.30, 'm', None, 'Input'],
            ['tk_vv_out', 'Outboard VV thickness', 0.60, 'm', None, 'Input'],
            ['g_ib_vv_ts', 'Inboard gap between VV and TS', 0.05, 'm', None, 'Input'],
            ['g_ob_vv_ts', 'Outboard gap between VV and TS', 0.05, 'm', None, 'Input'],
            ['r_vv_joint', 'Radius of inboard/outboard VV joint', 2. , 'm', None, 'Input'],
        ]
        # fmt: on
        cls.parameters = ParameterFrame(params)

        # Import the Inboard and outboard thermal shield loop from previous test
        x_tf_inboard = np.array(
            [
                2.0,
                1.575,
                1.575,
                0.7,
                0.7,
                0.67487523,
                0.62487523,
                0.6,
                0.6,
                0.62487523,
                0.67487523,
                0.7,
                0.7,
                1.575,
                1.575,
                2.0,
                2.0,
                1.625,
                1.625,
                1.575,
                1.575,
                0.65,
                0.65,
                0.62493762,
                0.57493762,
                0.55,
                0.55,
                0.57493762,
                0.62493762,
                0.65,
                0.65,
                1.575,
                1.575,
                0.65,
                0.65,
                1.625,
                1.625,
                2.0,
                2.0,
            ]
        )
        z_tf_inboard = np.array(
            [
                -3.975,
                -3.975,
                -4.075,
                -4.075,
                -3.99750156,
                -3.49500624,
                -2.49500624,
                -1.99750156,
                1.99750156,
                2.49500624,
                3.49500624,
                3.99750156,
                4.075,
                4.075,
                3.975,
                3.975,
                4.025,
                4.025,
                4.95,
                4.95,
                4.125,
                4.125,
                3.99875078,
                3.49750312,
                2.49750312,
                1.99875078,
                -1.99875078,
                -2.49750312,
                -3.49750312,
                -3.99875078,
                -4.125,
                -4.125,
                -4.875,
                -4.875,
                -4.95,
                -4.95,
                -4.025,
                -4.025,
                -3.975,
            ]
        )
        x_tf_outboard = np.array(
            [
                2.425,
                2.425,
                3.175,
                3.175,
                2.975,
                2.975,
                3.175,
                3.175,
                2.975,
                2.975,
                2.425,
                2.425,
                2.0,
                2.0,
                2.375,
                2.375,
                3.025,
                3.025,
                3.225,
                3.225,
                3.025,
                3.025,
                3.225,
                3.225,
                2.375,
                2.375,
                2.0,
                2.0,
                2.425,
            ]
        )
        z_tf_outboard = np.array(
            [
                3.975,
                4.9,
                4.9,
                2.425,
                2.425,
                1.575,
                1.575,
                -1.575,
                -1.575,
                -4.9,
                -4.9,
                -3.975,
                -3.975,
                -4.025,
                -4.025,
                -4.95,
                -4.95,
                -1.625,
                -1.625,
                1.625,
                1.625,
                2.375,
                2.375,
                4.95,
                4.95,
                4.025,
                4.025,
                3.975,
                3.975,
            ]
        )

        cls.to_vv = {
            "TS inboard loop": Loop(x=x_tf_inboard, z=z_tf_inboard),
            "TS outboard loop": Loop(x=x_tf_outboard, z=z_tf_outboard),
        }

    def test_segmented_vv_build(self):
        vaccum_vessel = SegmentedVaccumVessel(self.parameters, self.to_vv)

        # Inboard 2D loop area test
        a_ib_vv = vaccum_vessel.geom["Inboard profile"].area
        true_a_ib_vv = 3.046499062646456
        assert np.isclose(a_ib_vv, true_a_ib_vv, rtol=1.0e-3)

        # Outboard 2D loop area test
        a_ob_vv = vaccum_vessel.geom["Outboard profile"].area
        true_a_ob_vv = 6.597500000973231
        assert np.isclose(a_ob_vv, true_a_ob_vv, rtol=1.0e-3)

        # Test if the merge shell is properly obtained
        vv_out = vaccum_vessel.geom["2D profile"].outer
        vv_in = vaccum_vessel.geom["2D profile"].inner
        assert vv_out.point_in_poly([2.0, 0.0])
        assert vv_in.point_in_poly([2.0, 0.0])

    def test_disconneted_vv_build(self):
        self.parameters.tk_vv_in = 0.05
        self.parameters.g_ob_vv_ts = 0.3
        vaccum_vessel = SegmentedVaccumVessel(self.parameters, self.to_vv)

        # Inboard 2D loop area test
        a_ib_vv = vaccum_vessel.geom["Inboard profile"].area
        true_a_ib_vv = 0.5327498426501978
        assert np.isclose(a_ib_vv, true_a_ib_vv, rtol=1.0e-3)

        # Outboard 2D loop area test
        a_ob_vv = vaccum_vessel.geom["Outboard profile"].area
        true_a_ob_vv = 5.068750000931325
        assert np.isclose(a_ob_vv, true_a_ob_vv, rtol=1.0e-3)

        # Test if the merge shell is properly obtained
        vv_out = vaccum_vessel.geom["2D profile"].outer
        vv_in = vaccum_vessel.geom["2D profile"].inner
        assert vv_out.point_in_poly([2.0, 0.0])
        assert vv_in.point_in_poly([2.0, 0.0])
