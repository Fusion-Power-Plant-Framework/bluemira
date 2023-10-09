# COPYRIGHT PLACEHOLDER

"""
Exception classes for the power cycle model.
"""
from bluemira.base.error import BluemiraError


class PowerCycleError(BluemiraError):
    """PowerCycle base error."""


# ######################################################################
# TIME
# ######################################################################


class PowerCyclePhaseError(PowerCycleError):
    """
    Exception class for 'PowerCyclePhase' class of the Power Cycle
    module.
    """


class PowerCyclePulseError(PowerCycleError):
    """
    Exception class for 'PowerCyclePulse' class of the Power Cycle
    module.
    """


class PowerCycleScenarioError(PowerCycleError):
    """
    Exception class for 'PowerCycleScenario' class of the Power Cycle
    module.
    """


class ScenarioBuilderError(PowerCycleError):
    """
    Exception class for 'ScenarioBuilder' class of the Power Cycle
    module.
    """


# ######################################################################
# NET LOADS
# ######################################################################


class PowerLoadError(PowerCycleError):
    """
    Exception class for 'PowerLoad' class of the Power Cycle module.
    """


class PhaseLoadError(PowerCycleError):
    """
    Exception class for 'PhaseLoad' class of the Power Cycle module.
    """


class PulseLoadError(PowerCycleError):
    """
    Exception class for 'PulseLoad' class of the Power Cycle module.
    """


class ScenarioLoadError(PowerCycleError):
    """
    Exception class for 'ScenarioLoad' class of the Power Cycle module.
    """


# ######################################################################
# NET MANAGER
# ######################################################################


class PowerCycleSystemError(PowerCycleError):
    """
    Exception class for 'PowerCycleSystem' class of the Power Cycle
    module.
    """


class PowerCycleGroupError(PowerCycleError):
    """
    Exception class for 'PowerCycleGroup' class of the Power Cycle
    module.
    """


class PowerCycleManagerError(PowerCycleError):
    """
    Exception class for 'PowerCycleManager' class of the Power Cycle
    module.
    """


# ######################################################################
# BOP
# ######################################################################


class BOPPhaseError(PowerCycleError):
    """
    Exception class for 'BOPPhase' class of the Power Cycle module.
    """


class BOPPulseError(PowerCycleError):
    """
    Exception class for 'BOPPulse' class of the Power Cycle module.
    """


class BOPScenarioError(PowerCycleError):
    """
    Exception class for 'BOPScenario' class of the Power Cycle module.
    """
