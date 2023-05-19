# COPYRIGHT PLACEHOLDER

"""
Classes for the calculation of net power in the Power Cycle model.
"""
from dataclasses import dataclass

import numpy as np

from bluemira.base.constants import raw_uc
from bluemira.power_cycle.base import (
    BaseConfig,
    LoadType,
    ModuleType,
    PhaseLoadConfig,
    PowerCycleABC,
)
from bluemira.power_cycle.errors import PowerCycleSystemError
from bluemira.power_cycle.net.importers import EquilibriaImporter, PumpingImporter
from bluemira.power_cycle.net.loads import (
    LoadData,
    LoadModel,
    PhaseLoad,
    PowerLoad,
    PulseLoad,
)
from bluemira.power_cycle.time import PowerCycleScenario, ScenarioBuilder
from bluemira.power_cycle.tools import (
    read_json,
    unnest_list,
    validate_axes,
    validate_file,
    validate_lists_to_have_same_length,
)


@dataclass
class PowerCycleLoadConfig(BaseConfig):
    """Power cycle load config"""


@dataclass
class PowerCycleSystemConfig:
    name: str
    reactive: PowerCycleLoadConfig
    active: PowerCycleLoadConfig


@dataclass
class PowerCycleManagerConfig:
    name: str
    config_path: str
    systems: list


class PowerCycleSystem(PowerCycleABC):
    """
    Class to build the PowerLoad instances associated with the power
    production and consumption of a single plant system, used to
    represent that system in the time-dependent power balance of the
    Power Cycle module.

    Parameters
    ----------
    scenario:
        Scenario with a set of phases that matches every phase specified
        in the 'system_config' parameter.
    system_config:
        Dictionary that contains the necessary inputs to define
        objects of the PowerLoad class that characterize the power
        production and consumption of time-dependent power balance of
        the Power Cycle module.

    """

    def __init__(
        self,
        scenario: PowerCycleScenario,
        system_config: PowerCycleSystemConfig,
        label=None,
    ):
        super().__init__(system_config.name, label=label)
        self.scenario = scenario
        self._system_config = system_config
        self._active_config = system_config.active
        self._reactive_config = system_config.reactive

    @property
    def active_loads(self):
        """
        Dictionary of 'PhaseLoad' objects created from the 'active'
        (load type) fields of the JSON input file.
        """
        return self._make_phaseloads_from_config(self._active_config)

    @property
    def reactive_loads(self):
        """
        Dictionary of 'PhaseLoad' objects created from the 'reactive'
        (load type) fields of the JSON input file.
        """
        return self._make_phaseloads_from_config(self._reactive_config)

    @staticmethod
    def import_phaseload_inputs(module, variables_map):
        """
        Method that unpacks the 'variables_map' field of a JSON input
        file.
        """
        if module is ModuleType.EQUILIBRIA:
            phaseload_inputs = EquilibriaImporter.phaseload_inputs(
                variables_map,
            )

        elif module is ModuleType.PUMPING:
            phaseload_inputs = PumpingImporter.phaseload_inputs(
                variables_map,
            )
        else:
            phase_list = variables_map["phases"]
            normalise_list = variables_map["normalise"]

            validate_lists_to_have_same_length(
                phase_list,
                normalise_list,
            )

            unit = variables_map["unit"]
            consumption = variables_map["consumption"]
            all_efficiencies = variables_map["efficiencies"].values()

            description_list = variables_map["loads"]["description"]
            time_list = variables_map["loads"]["time"]
            data_list = variables_map["loads"]["data"]
            model_list = variables_map["loads"]["model"]

            powerload_list = []
            for n in range(
                validate_lists_to_have_same_length(
                    description_list,
                    time_list,
                    data_list,
                    model_list,
                )
            ):
                description = description_list[n]
                time = np.array(time_list[n])
                data = raw_uc(np.array(data_list[n]), unit, "W")
                
                for efficiency in all_efficiencies:
                    if consumption:
                        data /= efficiency
                    else:
                        data *= efficiency

                loaddata = LoadData(description, time, data)
                powerload = PowerLoad(description, loaddata, LoadModel[model_list[n]])

                powerload_list.append(powerload)

            phaseload_inputs = PhaseLoadConfig(
                phase_list, consumption, normalise_list, powerload_list
            )

        return phaseload_inputs

    def _build_phaseloads(self, load_name, phaseload_inputs):
        valid_phases = self.scenario.build_phase_library()

        phaseload_list = []
        for phase_label, normalisation_choice in zip(
            phaseload_inputs.phase_list, phaseload_inputs.normalise_list
        ):
            try:
                phase = valid_phases[phase_label]
            except KeyError:
                raise PowerCycleSystemError(
                    "scenario",
                    "It is not possible to build objects of the "
                    "'PhaseLoad' class for phases that are not "
                    "present in the 'scenario' attribute.",
                )

            phaseload = PhaseLoad(
                load_name,
                phase,
                phaseload_inputs.powerload_list,
                [normalisation_choice] * len(phaseload_inputs.powerload_list),
            )

            if phaseload_inputs.consumption:
                phaseload.make_consumption_explicit()

            phaseload_list.append(phaseload)

        return phaseload_list

    def _make_phaseloads_from_config(self, type_config):
        return {
            label: self._build_phaseloads(
                load_config["name"],
                self.import_phaseload_inputs(
                    load_config["module"],
                    load_config["variables_map"],
                ),
            )
            for label, load_config in type_config.items()
        }


class PowerCycleGroup(PowerCycleABC):
    """
    Class to build a collection of 'PowerCycleSystem' objects that
    represent a particular classification of power loads in a power
    planto.

    Parameters
    ----------
    name: str
        Description of the 'PowerCycleSystem' instance.
    scenario: PowerCycleScenario
        Scenario with a set of phases that matches every phase specified
        in configuration parameters for systems to be created.
    group_config: dict
        Dictionary that contains the necessary inputs to define
        'PowerCycleSystem' objects.
    """

    def __init__(
        self,
        name,
        scenario: PowerCycleScenario,
        group_config: dict,
        label=None,
    ):
        super().__init__(name, label=label)
        self.group_config = group_config
        self.system_library = {
            system_label: PowerCycleSystem(
                scenario,
                PowerCycleSystemConfig(**system_config),
                label=system_label,
            )
            for system_label, system_config in group_config.items()
        }


class PowerCycleManager:
    """
    Class to collect all inputs of the Power Cycle module and build
    net active and reactive loads to represent a the power production
    during a pulse.

    To be described:
        - Call ScenarioBuilder
        - Read load inputs JSON files (inputs for each Plant Group)
        - Build all Plant Groups
        - Merge all PulseLoad objects for active and reactive load types

    Parameters
    ----------
    scenario_config_path: str
        Path to the JSON file that defines the scenario.
    manager_config_path: str
        Path to the JSON file that lists configuration parameters to
        each Power Cycle load group.
    """

    def __init__(self, scenario_config_path: str, manager_config_path: str):
        self.scenario = ScenarioBuilder(scenario_config_path).scenario
        self.manager_configs = {
            key: PowerCycleManagerConfig(**val)
            for key, val in read_json(validate_file(manager_config_path)).items()
        }

        self.group_library = self._build_group_library()

    def _build_group_library(self):
        group_library = {}
        for group_label, group_inputs in self.manager_configs.items():
            group_config = {
                key: read_json(validate_file(group_inputs.config_path))[key]
                for key in group_inputs.systems
            }

            group = PowerCycleGroup(
                group_inputs.name,
                self.scenario,
                group_config,
                label=group_label,
            )
            group_library[group_label] = group
        return group_library

    def _build_pulseload_of_type(self, load_type):
        pulse = self.scenario.pulse_set[0]
        group_library = self.group_library

        all_phaseloads = []
        for group_label in group_library.keys():
            group = group_library[group_label]
            system_library = group.system_library

            for system_label in system_library.keys():
                system = system_library[system_label]

                loads_property = load_type.name.lower() + "_loads"
                loads_of_type = getattr(system, loads_property)
                system_phaseloads = [v for v in loads_of_type.values()]

                system_phaseloads = unnest_list(system_phaseloads)
                all_phaseloads.append(system_phaseloads)

        all_phaseloads = unnest_list(all_phaseloads)

        return PulseLoad(load_type.name.lower(), pulse, all_phaseloads)

    @property
    def net_active(self):
        """
        Net active power from the power plant, represented as a
        PulseLoad object.
        """
        return self._build_pulseload_of_type(LoadType.ACTIVE)

    @property
    def net_reactive(self):
        """
        Net reactive power from the power plant, represented as a
        PulseLoad object.
        """
        return self._build_pulseload_of_type(LoadType.REACTIVE)

    def plot(self, ax=None, n_points=None, **kwargs):
        """
        Plot a 'PulseLoad' curve for each load type and plot them in
        the same figure.

        Parameters
        ----------
        ax: Axes
            Instance of the 'matplotlib.axes.Axes' class, in which to
            plot. If 'None' is given, a new instance of axes is created.
        n_points: int
            Parameter 'n_points' passed to the 'plot' method of each
            'PulseLoad' to be plotted.
        **kwargs: dict
            Options for the 'plot' method.

        Returns
        -------
        ax: Axes
            Instance of the 'matplotlib.axes.Axes' class.
        """
        ax, active_plot_objects = self.net_active.plot(
            ax=validate_axes(ax),
            n_points=n_points,
            detailed=False,
            c="r",
            **kwargs,
        )
        ax, reactive_plot_objects = self.net_reactive.plot(
            ax=ax, n_points=n_points, detailed=False, c="b", **kwargs
        )

        return ax, (active_plot_objects, reactive_plot_objects)
