# COPYRIGHT PLACEHOLDER

"""
Classes for the calculation of net power in the Power Cycle model.
"""

from bluemira.base.constants import raw_uc
from bluemira.power_cycle.base import PowerCycleABC
from bluemira.power_cycle.errors import (
    PowerCycleGroupError,
    PowerCycleManagerError,
    PowerCycleSystemError,
)
from bluemira.power_cycle.net.importers import EquilibriaImporter, PumpingImporter
from bluemira.power_cycle.net.loads import (
    LoadData,
    PhaseLoad,
    PowerLoad,
    PowerLoadModel,
    PulseLoad,
)
from bluemira.power_cycle.time import PowerCycleScenario, ScenarioBuilder
from bluemira.power_cycle.tools import (
    convert_string_into_numeric_list,
    read_json,
    unnest_list,
    validate_dict,
    validate_file,
    validate_lists_to_have_same_length,
    validate_subdict,
)


class PowerCycleSystem(PowerCycleABC):
    """
    Class to build the PowerLoad instances associated with the power
    production and consumption of a single plant system, used to
    represent that system in the time-dependent power balance of the
    Power Cycle module.

    Parameters
    ----------
    name: str
        Description of the 'PowerCycleSystem' instance.
    scenario: PowerCycleScenario
        Scenario with a set of phases that matches every phase specified
        in the 'system_config' parameter.
    system_config: dict
        Dictionary that contains the necessary inputs to define
        objects of the PowerLoad class that characterize the power
        production and consumption of time-dependent power balance of
        the Power Cycle module.

    Properties
    ----------
    active_loads: dict
    reactive_loads: dict
    production_loads: dict

    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    _system_format = {
        "name": str,
        "production": dict,
        "reactive": dict,
        "active": dict,
    }
    _load_format = {
        "name": str,
        "module": [type(None), str],
        "variables_map": dict,
    }

    def __init__(self, scenario: PowerCycleScenario, system_config: dict, label=None):

        scenario = self._validate_scenario(scenario)

        system_format = self._system_format
        system_config = validate_dict(system_config, system_format)

        (
            name,
            production_config,
            reactive_config,
            active_config,
        ) = self._unpack_system_config(system_config)

        load_format = self._load_format
        active_config = validate_subdict(active_config, load_format)
        reactive_config = validate_subdict(reactive_config, load_format)
        production_config = validate_subdict(production_config, load_format)

        super().__init__(name, label=label)
        self.scenario = scenario
        self._system_config = system_config
        self._active_config = active_config
        self._reactive_config = reactive_config
        self._production_config = production_config

    @staticmethod
    def _validate_scenario(scenario):
        scenario_is_incorrect = type(scenario) != PowerCycleScenario
        if scenario_is_incorrect:
            raise PowerCycleSystemError("scenario")
        return scenario

    @staticmethod
    def _unpack_system_config(system_config):
        name = system_config["name"]
        production_config = system_config["production"]
        reactive_config = system_config["reactive"]
        active_config = system_config["active"]

        return name, production_config, reactive_config, active_config

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    @property
    def active_loads(self):
        active_config = self._active_config
        active_loads = self._make_phaseloads_from_config(active_config)
        return active_loads

    @property
    def reactive_loads(self):
        reactive_config = self._reactive_config
        reactive_loads = self._make_phaseloads_from_config(reactive_config)
        return reactive_loads

    @property
    def production_loads(self):
        production_config = self._production_config
        production_loads = self._make_phaseloads_from_config(production_config)
        return production_loads

    @staticmethod
    def import_phaseload_inputs(module, variables_map):
        if module is None:
            phase_list = variables_map["phases"]
            normalize_list = variables_map["normalize"]

            _ = validate_lists_to_have_same_length(
                phase_list,
                normalize_list,
            )

            unit = variables_map["unit"]
            efficiency_dict = variables_map["efficiencies"]
            all_efficiencies = efficiency_dict.values()

            description_list = variables_map["loads"]["description"]
            time_list = variables_map["loads"]["time"]
            data_list = variables_map["loads"]["data"]
            model_list = variables_map["loads"]["model"]

            n_loads = validate_lists_to_have_same_length(
                description_list,
                time_list,
                data_list,
                model_list,
            )

            powerload_list = []
            for n in range(n_loads):
                description = description_list[n]
                time = time_list[n]
                data = data_list[n]
                model = model_list[n]

                time = convert_string_into_numeric_list(time)
                data = convert_string_into_numeric_list(data)

                model = PowerLoadModel[model]

                data = raw_uc(data, unit, "W")
                for efficiency in all_efficiencies:
                    data = data / efficiency
                data = list(data)

                loaddata = LoadData(description, time, data)
                powerload = PowerLoad(description, loaddata, model)

                powerload_list.append(powerload)

            phaseload_inputs = dict()
            phaseload_inputs["phase_list"] = phase_list
            phaseload_inputs["normalize_list"] = normalize_list
            phaseload_inputs["powerload_list"] = powerload_list

        elif module == "equilibria":
            phaseload_inputs = EquilibriaImporter.phaseload_inputs(variables_map)

        elif module == "pumping":
            phaseload_inputs = PumpingImporter.phaseload_inputs(variables_map)

        else:
            raise PowerCycleSystem(
                "import",
                "Unknown routine for importing a phase load from "
                f"the {module!r} module.",
            )
        return phaseload_inputs

    def _build_phaseloads(self, load_name, phaseload_inputs):
        scenario = self.scenario
        valid_phases = scenario.build_phase_library()

        phase_list = phaseload_inputs["phase_list"]
        normalize_list = phaseload_inputs["normalize_list"]
        powerload_list = phaseload_inputs["powerload_list"]

        n_phases = len(phase_list)
        n_powerloads = len(powerload_list)

        phaseload_list = []
        for p in range(n_phases):
            phase_label = phase_list[p]
            normalization_choice = normalize_list[p]

            try:
                phase = valid_phases[phase_label]
            except KeyError:
                raise PowerCycleSystem(
                    "scenario",
                    "It is not possible to build objects of the "
                    "'PhaseLoad' class for phases that are not "
                    "present in the 'scenario' attribute.",
                )

            normalization_flags = [normalization_choice] * n_powerloads

            phaseload = PhaseLoad(
                load_name,
                phase,
                powerload_list,
                normalization_flags,
            )
            phaseload_list.append(phaseload)

        return phaseload_list

    def _make_phaseloads_from_config(self, type_config):
        system_loads = dict()
        for (label, load_config) in type_config.items():
            load_name = load_config["name"]
            module = load_config["module"]
            variables_map = load_config["variables_map"]

            phaseload_inputs = self.import_phaseload_inputs(
                module,
                variables_map,
            )
            phaseload_list = self._build_phaseloads(
                load_name,
                phaseload_inputs,
            )
            system_loads[label] = phaseload_list
        return system_loads


class PowerCycleGroup(PowerCycleABC):
    """ """

    # Build Power Cycle representations of each Plant System in the group

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    def __init__(
        self,
        name,
        scenario: PowerCycleScenario,
        group_config: dict,
        label=None,
    ):
        super().__init__(name, label=label)

        self.group_config = group_config

        system_library = self._build_system_library(scenario, group_config)
        self.system_library = system_library

    @staticmethod
    def _build_system_library(scenario, group_config):
        system_library = dict()
        for (system_label, system_config) in group_config.items():
            system = PowerCycleSystem(
                scenario,
                system_config,
                label=system_label,
            )
            system_library[system_label] = system
        return system_library

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    def _some_function(self):
        raise PowerCycleGroupError()


class PowerCycleManager:
    """ """

    # Call ScenarioBuilder
    # Read load inputs JSON files (inputs for each Plant Group)
    # Build all Plant Groups
    # Make active loads negative
    # Merge PhaseLoads and build PulseLoad

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    _manager_format = {
        "name": str,
        "config_path": str,
        "systems": list,
    }

    def __init__(self, scenario_config_path: str, manager_config_path: str):

        scenario_builder = ScenarioBuilder(scenario_config_path)
        scenario = scenario_builder.scenario
        self.scenario = scenario

        validated_path = validate_file(manager_config_path)
        json_contents = read_json(validated_path)

        manager_format = self._manager_format
        manager_config = validate_subdict(json_contents, manager_format)
        self._manager_config = manager_config

        group_library = self._build_group_library(scenario, json_contents)
        self.group_library = group_library

    @staticmethod
    def _build_group_library(scenario, manager_config):
        group_library = dict()
        for (group_label, group_inputs) in manager_config.items():
            group_name = group_inputs["name"]
            group_config_path = group_inputs["config_path"]
            group_systems = group_inputs["systems"]

            validated_path = validate_file(group_config_path)
            json_contents = read_json(validated_path)

            group_config = {key: json_contents[key] for key in group_systems}

            group = PowerCycleGroup(
                group_name,
                scenario,
                group_config,
                label=group_label,
            )
            group_library[group_label] = group
        return group_library

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    def _make_consumption_load_explicit():
        raise PowerCycleManagerError()

    def _build_pulseload(self, load_type):
        """
        load_type = 'active', 'reactive' or 'production'
        """
        group_library = self.group_library
        all_phaseloads = []
        all_group_labels = group_library.keys()
        for group_label in all_group_labels:
            group = group_library[group_label]
            system_library = group.system_library
            all_system_labels = system_library.keys()
            for system_label in all_system_labels:
                system = system_library[system_label]
                loads_property = load_type + "_loads"
                loads_of_type = getattr(system, loads_property)
                system_phaseloads = [v for v in loads_of_type.values()]
                system_phaseloads = unnest_list(system_phaseloads)
                all_phaseloads.append(system_phaseloads)
        all_phaseloads = unnest_list(all_phaseloads)

        pulseload = PulseLoad(load_type, all_phaseloads)
        return pulseload

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # ARITHMETICS
    # ------------------------------------------------------------------
