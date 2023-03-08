# COPYRIGHT PLACEHOLDER

import pytest

from bluemira.power_cycle.errors import EquilibriaImporterError, PumpingImporterError
from bluemira.power_cycle.net.importers import EquilibriaImporter, PumpingImporter
from bluemira.power_cycle.tools import validate_nonnegative
from tests.power_cycle.kits_for_tests import NetImportersTestKit

importers_testkit = NetImportersTestKit()


def assert_value_is_nonnegative(argument):
    possible_errors = (TypeError, ValueError)
    try:
        validate_nonnegative(argument)
    except possible_errors:
        assert False
    else:
        assert True


class TestEquilibriaImporter:
    def setup_method(self):
        duration_inputs = importers_testkit.equilibria_duration_inputs()
        self.duration_inputs = duration_inputs

    def test_duration(self):
        duration_inputs = self.duration_inputs
        possible_variable_map_fields = duration_inputs.keys()
        all_values = []
        for field in possible_variable_map_fields:
            possible_data_requests = duration_inputs[field]
            for request in possible_data_requests:
                variable_map_input = {field: request}
                value = EquilibriaImporter.duration(variable_map_input)
                assert_value_is_nonnegative(value)
                all_values.append(value)

            wrong_variable_map = {field: "non-implemented_request"}
            with pytest.raises(EquilibriaImporterError):
                value = EquilibriaImporter.duration(wrong_variable_map)


class TestPumpingImporter:
    def setup_method(self):
        duration_inputs = importers_testkit.pumping_duration_inputs()
        self.duration_inputs = duration_inputs

    def test_duration(self):
        duration_inputs = self.duration_inputs
        possible_variable_map_fields = duration_inputs.keys()
        all_values = []
        for field in possible_variable_map_fields:
            possible_data_requests = duration_inputs[field]
            for request in possible_data_requests:
                variable_map_input = {field: request}
                value = PumpingImporter.duration(variable_map_input)
                assert_value_is_nonnegative(value)
                all_values.append(value)

            wrong_variable_map = {field: "non-implemented_request"}
            with pytest.raises(PumpingImporterError):
                value = PumpingImporter.duration(wrong_variable_map)
