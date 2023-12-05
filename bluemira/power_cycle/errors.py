# COPYRIGHT PLACEHOLDER

"""
Exception classes for the power cycle model.
"""

from bluemira.base.error import BluemiraError


class PowerCycleError(BluemiraError):
    """PowerCycle base error."""


class PowerLoadError(PowerCycleError):
    """PowerCycleLoad error class."""


class ScenarioLoadError(PowerCycleError):
    """
    Exception class for 'ScenarioLoad' class of the Power Cycle module.
    """
