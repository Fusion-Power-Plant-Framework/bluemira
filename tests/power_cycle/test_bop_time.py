# COPYRIGHT PLACEHOLDER
"""
import pytest

from bluemira.power_cycle.errors import (
    BOPPhaseError,
)
from bluemira.power_cycle.BOP_time import (
    BOPPhaseDependency,
    BOPPhase,
    BOPPulse,
    BOPTimeline,
)


class TestBOPPhaseDependency:
    def test_members(self):
        all_names = [member.name for member in BOPPhaseDependency]
        all_values = [member.value for member in BOPPhaseDependency]

        for (name, value) in zip(all_names, all_values):
            assert isinstance(name, str)
            assert isinstance(value, str)


class TestBOPPhase:
    def setup_method(self):
        phase_name, phase_breakdown = example_phase_inputs()
        self.sample_name = phase_name
        self.sample_breakdown = phase_breakdown

    dependency_arguments = [
        None,
        154,
        "ss",
        "tt",
        BOPPhaseDependency("ss"),
        BOPPhaseDependency("tt"),
    ]

    @pytest.mark.parametrize("test_dependency", dependency_arguments)
    def test_constructor(self, test_dependency):
        name = self.sample_name
        breakdown = self.sample_breakdown
        try:
            sample = BOPPhase(name, breakdown, test_dependency)
        except (BOPPhaseError):
            dependency_class = type(test_dependency)
            dependency_is_valid = dependency_class == BOPPhaseDependency
            assert not dependency_is_valid


class TestBOPPulse:
    pass


class TestBOPTimeline:
    pass
"""
