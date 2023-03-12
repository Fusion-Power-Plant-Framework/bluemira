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
from bluemira.power_cycle.net.importers import EquilibriaImporter
from bluemira.power_cycle.net.loads import LoadData, PowerLoad
from bluemira.power_cycle.tools import (
    convert_string_into_numeric_list,
    read_json,
    unique_and_sorted_vector,
    validate_dict,
    validate_file,
    validate_lists_to_have_same_length,
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
        "module": [None, str],
        "variables_map": dict,
    }

    def __init__(self, system_config: dict):
        (
            name,
            production_config,
            reactive_config,
            active_config,
        ) = self._validate_config(system_config)
        self._system_config = system_config

        super().__init__(name)

        active_loads = self._build_loads_from_config(active_config)
        reactive_loads = self._build_loads_from_config(reactive_config)
        production_loads = self._build_loads_from_config(production_config)

        self.active_loads = active_loads
        self.reactive_loads = reactive_loads
        self.production_loads = production_loads

    @classmethod
    def _validate_config(cls, system_config):
        system_format = cls._system_format
        system_config = validate_dict(system_config, system_format)

        name = system_config["name"]
        production_config = system_config["production"]
        reactive_config = system_config["reactive"]
        active_config = system_config["active"]

        return name, production_config, reactive_config, active_config

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

                data = raw_uc(data, unit, "W")
                for efficiency in all_efficiencies:
                    data = data / efficiency
                data = list(data)

                loaddata = LoadData(description, time, data)
                powerload = PowerLoad(description, loaddata, model)

                powerload_list.append(powerload)

            load_inputs = dict()
            load_inputs["phase_list"] = phase_list
            load_inputs["normalize_list"] = normalize_list
            load_inputs["powerload_list"] = powerload_list

        elif module == "equilibria":
            load_inputs = EquilibriaImporter.phaseload_inputs(variables_map)

        elif module == "pumping":
            raise NotImplementedError()

        else:
            raise PowerCycleSystem(
                "import",
                "Unknown routine for importing a load from " f"the {module!r} module.",
            )
        return load_inputs

    @classmethod
    def _build_loads_from_config(cls, load_config):
        load_format = cls._load_format
        load_dict = dict()
        for (label, specs) in load_config.items():
            load_specs = validate_dict(specs, load_format)

            load_name = load_specs["name"]
            module = load_specs["module"]
            variables_map = load_specs["variables_map"]

            load_inputs = cls.import_load_inputs(module, variables_map)

        return load_name, load_inputs


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
    pass
