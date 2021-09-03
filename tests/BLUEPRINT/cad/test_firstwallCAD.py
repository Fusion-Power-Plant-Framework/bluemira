# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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
import os
from BLUEPRINT.base.file import get_BP_path
from BLUEPRINT.cad.firstwallCAD import FirstWallCAD
from BLUEPRINT.cad.cadtools import get_properties
from BLUEPRINT.equilibria.equilibrium import Equilibrium
from BLUEPRINT.systems.firstwall import FirstWallSN, FirstWallDN


class TestFirstWallCAD:

    # Member variable with type hint
    firstwall: FirstWallSN

    # Class-level initialisation
    @classmethod
    def setup_class(cls):
        # Create a FirstWallSN object
        read_path = get_BP_path("equilibria", subfolder="data/BLUEPRINT")
        eq_name = "EU-DEMO_EOF.json"
        eq_name = os.sep.join([read_path, eq_name])
        eq = Equilibrium.from_eqdsk(eq_name)
        cls.firstwall = FirstWallSN(FirstWallSN.default_params, {"equilibrium": eq})

    # Test to call the default build method
    def test_default_build(self):

        firstwallcad = FirstWallCAD(self.firstwall)

        # Object created should have dict component
        assert hasattr(firstwallcad, "component")
        assert type(firstwallcad.component) == dict

        # We expect exactly one cad component to have been created
        for key, list in firstwallcad.component.items():
            assert len(list) == 1

        # Look for shapes key
        assert "shapes" in firstwallcad.component.keys()

        # Retrieve volume and check positivity
        # TODO: check an exact volume
        vol = get_properties(firstwallcad.component["shapes"][0])["Volume"]
        assert vol > 0.0

    # Test to call the neutronics build method: not implemented
    def test_neutronics_build(self):

        with pytest.raises(NotImplementedError):
            # Currently throws NotYetImplemented
            assert FirstWallCAD(self.firstwall, neutronics=True)


class TestFirstWallCAD_DN:

    # Member variable with type hint
    firstwall: FirstWallDN

    # Class-level initialisation
    @classmethod
    def setup_class(cls):
        # Create a FirstWallDN object
        read_path = get_BP_path("BLUEPRINT/equilibria/test_data", subfolder="tests")
        eq_name = "DN-DEMO_eqref.json"
        eq_name = os.sep.join([read_path, eq_name])
        eq = Equilibrium.from_eqdsk(eq_name)
        cls.firstwall = FirstWallDN(FirstWallDN.default_params, {"equilibrium": eq})

    # Test to verifiy the volume of the extruded fw shell
    def test_volume_double_null(self):

        firstwallcad = FirstWallCAD(self.firstwall)

        # Check the exact volume (with some tolerance)
        # The value 142 m^3 was checked on SolidWorks (CAD software)
        vol = (
            get_properties(firstwallcad.component["shapes"][0])["Volume"]
        ) * firstwallcad.n_TF
        assert (142 - 3) < vol < (142 + 3)


if __name__ == "__main__":
    pytest.main([__file__])
