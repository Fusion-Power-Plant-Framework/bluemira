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
import pytest
from numpy import pi

from bluemira.base.parameter import ParameterFrame
from BLUEPRINT.cad.blanketCAD import STBlanketCAD
from BLUEPRINT.geometry.geomtools import make_box_xz
from BLUEPRINT.systems.blanket import STBreedingBlanket
from tests.BLUEPRINT.cad.test_firstwallCAD import check_cad


class TestBananaBlanket:

    # Class-level initialisation
    @classmethod
    def setup_class(cls):

        fw_x_low = 7
        fw_x_high = 10
        fw_z = 5
        vv_x_low = 6
        vv_x_high = 12
        vv_z = 7
        fw_dummy = make_box_xz(fw_x_low, fw_x_high, -fw_z, fw_z)
        vv_dummy = make_box_xz(vv_x_low, vv_x_high, -vv_z, vv_z)

        inputs = {"fw_outboard": fw_dummy, "vv_inner": vv_dummy}

        params = ParameterFrame(STBreedingBlanket.default_params.to_records())
        cls.blanket = STBreedingBlanket(params, inputs)

        # Save cyclinder dimensions to calculate volumes
        cls.r1 = fw_x_high + cls.blanket.params.g_bb_fw
        cls.r2 = cls.r1 + cls.blanket.params.tk_bb_bz
        cls.r3 = cls.r2 + cls.blanket.params.tk_bb_man
        cls.h = 2 * fw_z

    def calc_volumes(self):

        norm = pi * self.h / self.blanket.params.n_TF

        # Cylinders
        v1 = norm * self.r1**2
        v2 = norm * self.r2**2
        v3 = norm * self.r3**2

        # Breeding zone
        v_bz = v2 - v1

        # Manifold
        v_man = v3 - v2

        return [v_bz, v_man]

    def test_banana_blanket_cad(self):
        blanketcad = STBlanketCAD(self.blanket)
        volumes = self.calc_volumes()
        assert check_cad(blanketcad, 2, volumes)

    # Test to call the neutronics build method: not implemented
    def test_neutronics_build(self):

        with pytest.raises(NotImplementedError):
            # Currently throws NotYetImplemented
            assert STBlanketCAD(self.blanket, neutronics=True)


class TestImmersionBlanket:

    # Class-level initialisation
    @classmethod
    def setup_class(cls):

        fw_x_low = 7
        fw_x_high = 10
        fw_z = 5
        vv_x_low = 6
        vv_x_high = 12
        vv_z = 7
        fw_dummy = make_box_xz(fw_x_low, fw_x_high, -fw_z, fw_z)
        vv_dummy = make_box_xz(vv_x_low, vv_x_high, -vv_z, vv_z)
        inputs = {"fw_outboard": fw_dummy, "vv_inner": vv_dummy}
        params = ParameterFrame(STBreedingBlanket.default_params.to_records())
        params.blanket_type = "immersion"
        cls.blanket = STBreedingBlanket(params, inputs)

        # Save dimensions to calculate volumes
        cls.r3 = vv_x_high
        cls.r2 = cls.r3 - cls.blanket.params.tk_bb_man
        cls.r1 = fw_x_high + cls.blanket.params.g_bb_fw
        cls.h_bz = 2 * fw_z
        cls.h_man = vv_z - fw_z

    def calc_volumes(self):

        # Common factor
        norm = pi / self.blanket.params.n_TF

        # Breeding zone is a hollow cylinder
        v1 = norm * self.h_bz * self.r1**2
        v2 = norm * self.h_bz * self.r3**2
        v_bz = v2 - v1

        # Manifold (upper, lower) is also hollow cylinder
        v1 = norm * self.h_man * self.r2**2
        v2 = norm * self.h_man * self.r3**2
        v_man = v2 - v1

        return [v_bz, v_man, v_man]

    def test_immersion_blanket_cad(self):
        blanketcad = STBlanketCAD(self.blanket)
        volumes = self.calc_volumes()
        assert check_cad(blanketcad, 3, volumes)

    # Test to call the neutronics build method: not implemented
    def test_neutronics_build(self):

        with pytest.raises(NotImplementedError):
            # Currently throws NotYetImplemented
            assert STBlanketCAD(self.blanket, neutronics=True)
