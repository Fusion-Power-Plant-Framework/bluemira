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
import math

import pytest

from BLUEPRINT.cad.cadtools import get_properties
from BLUEPRINT.cad.centralcolumnshieldCAD import CentralColumnShieldCAD
from BLUEPRINT.systems.centralcolumnshield import CentralColumnShield
from tests.BLUEPRINT.systems.test_centralcolumnshield import (
    setup_cc_inputs,
    setup_cc_params,
)


class TestCentralColumnShieldCAD:

    # Class-level initialisation
    @classmethod
    def setup_class(cls):
        cls.xmid = 2.25
        cls.radius = 1.0
        cls.length = 4.0
        zmid = 0.0
        centre = (cls.xmid, zmid)
        ccs_params = setup_cc_params()
        ccs_params.r_ccs = cls.xmid
        ccs_inputs = setup_cc_inputs(centre, centre, cls.radius, cls.length)
        cls.ccs = CentralColumnShield(ccs_params, ccs_inputs)

    # Test to call the default build method
    def test_default_build(self):

        ccs_cad = CentralColumnShieldCAD(self.ccs)

        # Object created should have dict component
        assert hasattr(ccs_cad, "component")
        assert type(ccs_cad.component) == dict

        # We expect exactly one cad component to have been created
        for key, list in ccs_cad.component.items():
            assert len(list) == 1

        # Look for shapes key
        assert "shapes" in ccs_cad.component.keys()

        # Retrieve volume and check approx value
        vol = get_properties(ccs_cad.component["shapes"][0])["Volume"]
        vol_check = self.calc_analytic_vol()
        assert vol == pytest.approx(vol_check, rel=1e-4)

    # Test to call the neutronics build method: not implemented
    def test_neutronics_build(self):

        with pytest.raises(NotImplementedError):
            # Currently throws NotYetImplemented
            assert CentralColumnShieldCAD(self.ccs, neutronics=True)

    def calc_analytic_vol(self):

        # Volume of vacuum vessel = big cylinder - small cylinder
        vv_offset = abs(self.ccs.params.g_ccs_vv_add)
        vv_inner_r = self.xmid - self.length / 2.0 + vv_offset
        vv_outer_r = self.xmid
        vv_height = self.length - 2.0 * vv_offset
        vv_vol = math.pi * vv_height * (vv_outer_r**2 - vv_inner_r**2)

        # Volume of first wall cutaway is half-circle of revolution
        # Formula is:
        # (pi r^2)(pi R) - 4/3 pi r^3
        # Note: it's not equivalent to vol(torus)/2
        fw_offset = self.ccs.params.g_ccs_fw
        fw_minor_r = self.radius + fw_offset
        fw_major_r = self.xmid
        fw_vol = (math.pi * fw_minor_r**2) * (math.pi * fw_major_r)
        fw_vol -= 4 / 3 * math.pi * fw_minor_r**3

        # Volume of central column shield is difference
        vol = vv_vol - fw_vol

        # Normalise by the angle we rotated through
        vol /= self.ccs.params.n_TF
        return vol
