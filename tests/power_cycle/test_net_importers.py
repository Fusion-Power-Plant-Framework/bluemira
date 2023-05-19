# COPYRIGHT PLACEHOLDER

import pytest

from bluemira.power_cycle.base import PowerCycleImporterABC
from bluemira.power_cycle.errors import EquilibriaImporterError, PumpingImporterError
from bluemira.power_cycle.net.importers import EquilibriaImporter, PumpingImporter
from tests.power_cycle.kits_for_tests import (
    NetImportersTestKit,
    assert_value_is_nonnegative,
)

importers_testkit = NetImportersTestKit()


class TestEquilibriaImporter:
    def setup_method(self):
        duration_inputs = importers_testkit.equilibria_duration_inputs()
        self.duration_inputs = duration_inputs

        phaseload_inputs = importers_testkit.equilibria_phaseload_inputs()
        self.phaseload_inputs = phaseload_inputs

    def test_duration(self):
        all_values = []
        for field, data in self.duration_inputs.items():
            for request in data:
                value = EquilibriaImporter.duration({field: request})
                assert_value_is_nonnegative(value)
                all_values.append(value)

            with pytest.raises(EquilibriaImporterError):
                EquilibriaImporter.duration({field: "non-implemented_request"})

    def test_phaseload_inputs(self):
        all_values = []
        for field, data in self.phaseload_inputs.items():
            for request in data:
                value = EquilibriaImporter.phaseload_inputs({field: request})
                all_values.append(value)

            with pytest.raises(EquilibriaImporterError):
                EquilibriaImporter.duration({field: "non-implemented_request"})


class TestPumpingImporter:
    def setup_method(self):
        duration_inputs = importers_testkit.pumping_duration_inputs()
        self.duration_inputs = duration_inputs

    def test_duration(self):
        all_values = []
        for field, data in self.duration_inputs.items():
            for request in data:
                value = PumpingImporter.duration({field: request})
                assert_value_is_nonnegative(value)
                all_values.append(value)

            with pytest.raises(PumpingImporterError):
                PumpingImporter.duration({field: "non-implemented_request"})
