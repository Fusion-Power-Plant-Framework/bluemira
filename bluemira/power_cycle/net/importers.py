# COPYRIGHT PLACEHOLDER

"""
Classes for importing data from other Bluemira modules.
"""

from bluemira.base.constants import raw_uc
from bluemira.power_cycle.base import PhaseLoadConfig, PowerCycleImporterABC
from bluemira.power_cycle.errors import EquilibriaImporterError, PumpingImporterError


class EquilibriaImporter(PowerCycleImporterABC):
    """
    Static class to import inputs from the 'equilibria' module into the
    Power Cycle module.
    """

    @staticmethod
    def duration(variables_map):
        """
        Mock of method to import durations from 'equilibria' module.
        """
        request = variables_map["desired_data"]
        if request == "CS-recharge-time":
            duration = raw_uc(5, "minute", "second")

        elif request == "ramp-up-time":
            duration = raw_uc(157, "second", "second")

        elif request == "ramp-down-time":
            duration = raw_uc(157, "second", "second")

        else:
            raise EquilibriaImporterError(
                "duration",
                "The 'duration' method has no implementation for "
                f"the value {request!r} passed as the parameter "
                "'desired_data'.",
            )
        return duration

    @staticmethod
    def phaseload_inputs(variables_map):
        """
        Mock of method to import loads from 'equilibria' module.
        """

        request = variables_map["desired_data"]
        if request == "CS-coils":
            phaseload_inputs = PhaseLoadConfig([], True, [], [])

        elif request == "TF-coils":
            phaseload_inputs = PhaseLoadConfig([], True, [], [])

        elif request == "PF-coils":
            phaseload_inputs = PhaseLoadConfig([], True, [], [])

        else:
            raise EquilibriaImporterError(
                "phaseload_inputs",
                "The 'phaseload_inputs' method has no implementation "
                f"for the value {request!r} passed as the parameter "
                "'desired_data'.",
            )
        return phaseload_inputs


class PumpingImporter(PowerCycleImporterABC):
    """
    Static class to import inputs from the 'pumping' module into the
    Power Cycle module.
    """

    @staticmethod
    def duration(variables_map):
        """
        Mock of method to import durations from 'pumping' module (TBD).
        """
        request = variables_map["desired_data"]
        if request == "pumpdown-time":
            duration = raw_uc(10, "minute", "second")
        else:
            raise PumpingImporterError(
                "duration",
                "The 'duration' method has no implementation for "
                f"the value {request!r} passed as the parameter "
                "'desired_data'.",
            )
        return duration

    @staticmethod
    def phaseload_inputs(variables_map):
        """
        Mock of method to import loads from 'pumping' module (TBD).
        """
        raise NotImplementedError()
