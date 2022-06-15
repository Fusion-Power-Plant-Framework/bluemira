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

from BLUEPRINT.cad.cadtools import get_properties
from BLUEPRINT.cad.firstwallCAD import FirstWallCAD
from tests.BLUEPRINT.systems.test_firstwall import load_firstwall_dn, load_firstwall_sn


def check_cad(system_cad, n_shapes, ref_volumes=None):
    # Object created should have dict component
    assert hasattr(system_cad, "component")
    assert type(system_cad.component) == dict

    if ref_volumes:
        assert isinstance(ref_volumes, list)
        assert len(ref_volumes) == n_shapes

    # Check number of components
    for vals in system_cad.component.values():
        assert len(vals) == n_shapes

    # Look for shapes key
    assert "shapes" in system_cad.component.keys()

    # Check number of shapes
    assert len(system_cad.component["shapes"]) == n_shapes

    # Check volume
    for i_shape, cad_shape in enumerate(system_cad.component["shapes"]):
        vol = get_properties(cad_shape)["Volume"]
        # If we provided some reference values
        if ref_volumes:
            assert pytest.approx(vol, 1e-3) == ref_volumes[i_shape]
        # Otherwise just check positivity
    else:
        assert vol > 0.0

    return True


class TestFirstWallCAD:

    # Class-level initialisation
    @classmethod
    def setup_class(cls):
        cls.firstwall = load_firstwall_sn()
        cls.firstwall.build()

    # Test to call the default build method
    def test_default_build(self):

        firstwallcad = FirstWallCAD(self.firstwall)
        assert check_cad(firstwallcad, 4)

    # Test to call the neutronics build method: not implemented
    def test_neutronics_build(self):

        with pytest.raises(NotImplementedError):
            # Currently throws NotYetImplemented
            assert FirstWallCAD(self.firstwall, neutronics=True)


class TestFirstWallCAD_DN:

    # Class-level initialisation
    @classmethod
    def setup_class(cls):
        cls.firstwall = load_firstwall_dn()
        cls.firstwall.build()

    # Test to verifiy the volume of the extruded fw shell
    def test_default_build_double_null(self):

        firstwallcad = FirstWallCAD(self.firstwall)
        assert check_cad(firstwallcad, 6)

    # Test to call the neutronics build method: not implemented
    def test_neutronics_build_double_null(self):

        with pytest.raises(NotImplementedError):
            # Currently throws NotYetImplemented
            assert FirstWallCAD(self.firstwall, neutronics=True)
