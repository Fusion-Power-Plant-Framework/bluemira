# COPYRIGHT PLACEHOLDER

import pytest

from bluemira.power_cycle.base import PowerCycleImporterABC
from bluemira.power_cycle.errors import (
    EquilibriaImporterError,
    PowerCycleImporterABCError,
    PumpingImporterError,
)
from bluemira.power_cycle.net.importers import EquilibriaImporter, PumpingImporter
from bluemira.power_cycle.tools import validate_dict
from tests.power_cycle.kits_for_tests import (
    NetImportersTestKit,
    assert_value_is_nonnegative,
)

importers_testkit = NetImportersTestKit()


class TestEquilibriaImporter:
    tested_class_super = PowerCycleImporterABC
    tested_class_super_error = PowerCycleImporterABCError
    tested_class = EquilibriaImporter
    tested_class_error = EquilibriaImporterError

    def setup_method(self):
        duration_inputs = importers_testkit.equilibria_duration_inputs()
        self.duration_inputs = duration_inputs

        phaseload_inputs = importers_testkit.equilibria_phaseload_inputs()
        self.phaseload_inputs = phaseload_inputs

    def test_duration(self):
        tested_class = self.tested_class
        tested_class_error = self.tested_class_error

        duration_inputs = self.duration_inputs
        possible_variable_map_fields = duration_inputs.keys()
        all_values = []
        for field in possible_variable_map_fields:
            possible_data_requests = duration_inputs[field]
            for request in possible_data_requests:
                variable_map_input = {field: request}
                value = tested_class.duration(variable_map_input)
                assert_value_is_nonnegative(value)
                all_values.append(value)

            wrong_variable_map = {field: "non-implemented_request"}
            with pytest.raises(tested_class_error):
                value = tested_class.duration(wrong_variable_map)

    def test_phaseload_inputs(self):
        tested_class = self.tested_class
        tested_class_error = self.tested_class_error

        phaseload_inputs = self.phaseload_inputs
        phaseload_inputs_format = tested_class._phaseload_inputs
        possible_variable_map_fields = phaseload_inputs.keys()
        all_values = []
        for field in possible_variable_map_fields:
            possible_data_requests = phaseload_inputs[field]
            for request in possible_data_requests:
                variable_map_input = {field: request}
                value = tested_class.phaseload_inputs(variable_map_input)
                validated_value = validate_dict(value, phaseload_inputs_format)
                all_values.append(validated_value)

            wrong_variable_map = {field: "non-implemented_request"}
            with pytest.raises(tested_class_error):
                value = tested_class.duration(wrong_variable_map)


class TestPumpingImporter:
    tested_class_super = PowerCycleImporterABC
    tested_class_super_error = PowerCycleImporterABCError
    tested_class = PumpingImporter
    tested_class_error = PumpingImporterError

    def setup_method(self):
        duration_inputs = importers_testkit.pumping_duration_inputs()
        self.duration_inputs = duration_inputs

    def test_duration(self):
        tested_class = self.tested_class
        tested_class_error = self.tested_class_error

        duration_inputs = self.duration_inputs
        possible_variable_map_fields = duration_inputs.keys()
        all_values = []
        for field in possible_variable_map_fields:
            possible_data_requests = duration_inputs[field]
            for request in possible_data_requests:
                variable_map_input = {field: request}
                value = tested_class.duration(variable_map_input)
                assert_value_is_nonnegative(value)
                all_values.append(value)

            wrong_variable_map = {field: "non-implemented_request"}
            with pytest.raises(tested_class_error):
                value = tested_class.duration(wrong_variable_map)

    def test_phaseload_inputs(self):
        pass
