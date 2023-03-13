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
from bluemira.power_cycle.net.loads import LoadData, PowerLoad, PowerLoadModel
from bluemira.power_cycle.time import PowerCycleScenario
from bluemira.power_cycle.tools import (
    convert_string_into_numeric_list,
    read_json,
    unique_and_sorted_vector,
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
    system_config: dict
        Dictionary that contains the necessary inputsto define
        objects of the PowerLoad class that characterize the power
        production and consumption of time-dependent power balance of the
        Power Cycle module.

    Attributes
    ----------

    """

    # Build active & reactive PowerLoads for a Plant System

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

    def __init__(self, scenario: PowerCycleScenario, system_config: dict):

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

        super().__init__(name)
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
        phase_list = phaseload_inputs["phase_list"]
        normalize_list = phaseload_inputs["normalize_list"]
        powerload_list = phaseload_inputs["powerload_list"]

        n_phases = len(phase_list)
        for p in range(n_phases):
            phase = []

        phaseload_list = []
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
    # Build Power Cycle representations of each Plant System in the group
    pass


class PowerCycleManager:
    # Call ScenarioBuilder
    # Read load inputs JSON files (inputs for each Plant System)
    # Build all Plant Systems
    # Build PhaseLoads and PulseLoad

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # ARITHMETICS
    # ------------------------------------------------------------------
    def _make_consumption_load_explicit():
        pass
