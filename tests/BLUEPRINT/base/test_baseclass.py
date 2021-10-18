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
import os
import filecmp
from typing import Type

from BLUEPRINT.base.baseclass import ReactorSystem
from BLUEPRINT.base.file import get_BP_path
from bluemira.base.error import BluemiraError
from BLUEPRINT.nova.firstwall import DivertorProfile
from BLUEPRINT.systems.blanket import BreedingBlanket
from BLUEPRINT.systems.divertor import Divertor
from BLUEPRINT.geometry.geomtools import make_box_xz


class TestBaseClass:
    """
    Tests for the base class functionality.
    """

    class MyReactorSystem(ReactorSystem):
        """
        Test parent ReactorSystem.
        """

        pass

    class MyChildReactorSystem(MyReactorSystem):
        """
        Test child ReactorSystem.
        """

        pass

    @pytest.mark.parametrize(
        "class_name,the_class",
        [
            ("MyReactorSystem", MyReactorSystem),
            ("MyChildReactorSystem", MyChildReactorSystem),
        ],
    )
    def test_get_class(self, class_name, the_class):
        """
        Test that reactor system classes can be retrieved
        """
        my_class = TestBaseClass.MyReactorSystem.get_class(class_name)
        assert my_class is not None
        assert my_class == the_class

    @pytest.mark.parametrize(
        "the_class,name",
        [
            (MyReactorSystem, "NotAClass"),
            (MyChildReactorSystem, "MyReactorSystem"),
        ],
    )
    def test_get_class_fail(self, the_class, name):
        with pytest.raises(BluemiraError):
            TestBaseClass.MyReactorSystem.get_class("NotAClass")

    def test_avoid_duplicate_classes(self):
        with pytest.raises(BluemiraError):

            class MyReactorSystem(ReactorSystem):
                pass


class CSVWriterReactorSystem(ReactorSystem):
    """
    ReactorSystem to test csv write functionality.
    """

    def __init__(self):
        self.geom["Dummy Loop"] = make_box_xz(0.0, 1.0, 0.0, 1.0)

    @property
    def csv_write_loop_names(self):
        return ["Dummy Loop"]


def test_reactor_system_write_to_csv():
    # Create an instance of our dummy class
    csv_system = CSVWriterReactorSystem()

    # Test write with all default arguments
    test_file_base = "reactor_system"
    csv_system.save_geom_as_csv(test_file_base)

    # Fetch comparison data file
    data_file = "reactor_system_dummy_loop_no_metadata.csv"
    data_dir = "BLUEPRINT/base/test_data"
    compare_path = get_BP_path(data_dir, subfolder="tests")
    compare_file = os.sep.join([compare_path, data_file])

    # Compare against test file
    test_keys = ["dummy_loop"]
    for key in test_keys:
        test_file = test_file_base + "_" + key + ".csv"
        assert filecmp.cmp(test_file, compare_file)
        # Clean up
        os.remove(test_file)

    # Test write, specifying file_base, path and metadata
    metadata = "# Metadata string"
    csv_system.save_geom_as_csv(test_file_base, compare_path, metadata)

    # Fetch comparison data file
    data_file = "reactor_system_dummy_loop_with_metadata.csv"
    compare_file = os.sep.join([compare_path, data_file])

    # Compare against test file
    for key in test_keys:
        test_file = os.sep.join([compare_path, test_file_base + "_" + key + ".csv"])
        assert filecmp.cmp(test_file, compare_file)
        # Clean up
        os.remove(test_file)

    # Create an empty reactor system which will fail write
    fail_csv_system = ReactorSystem()

    # Check write fails
    with pytest.raises(RuntimeError):
        fail_csv_system.save_geom_as_csv(test_file_base)


class TestDefineSystemClass:
    """
    Tests for dynamic system class setting.
    """

    class DummyReactorSystem(ReactorSystem):
        """
        Test ReactorSystem with a divertor sub-system.
        """

        div: Type[Divertor]

        def __init__(self, inputs):
            self.inputs = inputs

            self._generate_subsystem_classes(self.inputs)

    class DummyChildReactorSystem(DummyReactorSystem):
        """
        Test ReactorSystem that's a child of a system with a divertor sub-system.

        Checks that the new annotations don't break the parent class's annotations.
        """

        inputs: dict

    class DummyChildReactorSystemExtraSub(DummyReactorSystem):
        """
        Test ReactorSystem that's a child of a system with a divertor sub-system and
        an additional sub system.

        Checks that the new annotations don't break the parent class's annotations.
        """

        bb: Type[BreedingBlanket]

    class DummySecondReactorSystem(ReactorSystem):
        """
        Test ReactorSystem that also defines a div subsystem but with a different type.
        """

        div: Type[DivertorProfile]

        def __init__(self, inputs):
            self.inputs = inputs

            self._generate_subsystem_classes(self.inputs)

    class NewDivertor(Divertor):
        """
        A new divertor class.
        """

        pass

    class NewBreedingBlanket(BreedingBlanket):
        """
        A new blanket class.
        """

        pass

    @pytest.mark.parametrize(
        "system,inputs,div_class",
        [
            (DummyReactorSystem, {}, Divertor),
            (DummyReactorSystem, {"div_class_name": "NewDivertor"}, NewDivertor),
            (DummySecondReactorSystem, {}, DivertorProfile),
        ],
    )
    def test_get_subsystem_class(self, system, inputs, div_class):
        """
        Test the sub-system behaviour - get the class using the variable name as key.
        """
        my_system = system(inputs)
        assert my_system.get_subsystem_class("div") is div_class

    @pytest.mark.parametrize(
        "inputs,div_class",
        [
            ({}, Divertor),
            ({"div_class_name": "NewDivertor"}, NewDivertor),
        ],
    )
    def test_get_subsystem_class_from_child(self, inputs, div_class):
        """
        Test the sub-system behaviour - get the class using the variable name as key.
        """
        my_system = TestDefineSystemClass.DummyChildReactorSystem(inputs)
        assert my_system.get_subsystem_class("div") is div_class

    @pytest.mark.parametrize(
        "inputs,div_class,bb_class",
        [
            ({}, Divertor, BreedingBlanket),
            (
                {"div_class_name": "NewDivertor", "bb_class_name": "NewBreedingBlanket"},
                NewDivertor,
                NewBreedingBlanket,
            ),
        ],
    )
    def test_get_subsystem_class_from_child_extra_sub(self, inputs, div_class, bb_class):
        """
        Test the sub-system behaviour - get the class using the variable name as key.
        """
        my_system = TestDefineSystemClass.DummyChildReactorSystem(inputs)
        assert my_system.get_subsystem_class("div") is div_class

    def test_error_class_not_defined(self):
        """
        Test a non-existent class in the build config raises an error.
        """
        with pytest.raises(BluemiraError):
            TestDefineSystemClass.DummyReactorSystem(
                {"div_class_name": "AnotherDivertor"}
            )

    def test_error_wrong_subclass(self):
        """
        Test providing class that does not match the system raises an error.
        """
        with pytest.raises(BluemiraError):
            TestDefineSystemClass.DummyReactorSystem(
                {"div_class_name": "BreedingBlanket"}
            )

    def test_error_system_not_defined(self):
        """
        Test that getting a system that doesn't exist raises an error.
        """
        my_system = TestDefineSystemClass.DummyReactorSystem({})
        with pytest.raises(BluemiraError):
            my_system.get_subsystem_class("DIVERTOR")


if __name__ == "__main__":
    pytest.main([__file__])
