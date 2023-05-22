# COPYRIGHT PLACEHOLDER

"""
Classes for the calculation of net power in the Power Cycle model.
"""
import numpy as np

from bluemira.base.constants import raw_uc
from bluemira.power_cycle.base import PowerCycleABC, PowerCycleImporterABC
from bluemira.power_cycle.errors import (
    PowerCycleGroupError,
    PowerCycleManagerError,
    PowerCycleSystemError,
)
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
    FormattedDict,
    FormattedLibrary,
    Library,
    read_json,
    unnest_list,
    validate_axes,
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
    """

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    _phaseload_inputs_format = PowerCycleImporterABC._phaseload_inputs_format

    _system_format = FormattedDict.Format(
        {
            "name": str,
            "reactive": dict,
            "active": dict,
        }
    )
    _load_format = FormattedLibrary.Format(
        {
            "name": str,
            "module": [None, str],
            "variables_map": dict,
        }
    )

    def __init__(
        self,
        scenario: PowerCycleScenario,
        system_config: dict,
        label=None,
    ):
        scenario = self._validate_scenario(scenario)

        system_config = FormattedDict(
            self._system_format,
            dictionary=system_config,
        )

        (
            name,
            reactive_config,
            active_config,
        ) = self._unpack_system_config(system_config)

        active_config = FormattedLibrary(
            self._load_format,
            dictionary=active_config,
        )
        reactive_config = FormattedLibrary(
            self._load_format,
            dictionary=reactive_config,
        )

        super().__init__(name, label=label)
        self.scenario = scenario
        self._system_config = system_config
        self._active_config = active_config
        self._reactive_config = reactive_config

    @staticmethod
    def _validate_scenario(scenario):
        if type(scenario) != PowerCycleScenario:
            raise PowerCycleSystemError("scenario")
        return scenario

    @staticmethod
    def _unpack_system_config(system_config):
        name = system_config["name"]
        reactive_config = system_config["reactive"]
        active_config = system_config["active"]
        return name, reactive_config, active_config

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

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

    @classmethod
    def list_all_load_types(cls):
        """
        List with all valid load types.
        """
        load_types = list(cls._system_format.keys())
        load_types.remove("name")
        return load_types

    @classmethod
    def import_phaseload_inputs(cls, module, variables_map):
        """
        Method that unpacks the 'variables_map' field of a JSON input
        file.
        """
        if module is None:
            phase_list = variables_map["phases"]
            normalize_list = variables_map["normalize"]

            _ = validate_lists_to_have_same_length(
                phase_list,
                normalize_list,
            )

            unit = variables_map["unit"]
            consumption = variables_map["consumption"]
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

                time = np.array(time_list[n])
                data = raw_uc(np.array(data_list[n]), unit, "W")

                model = LoadModel[model]

                data = raw_uc(data, unit, "W")
                for efficiency in all_efficiencies:
                    if consumption:
                        data = data / efficiency
                    else:
                        data = data * efficiency
                data = list(data)

                loaddata = LoadData(description, time, data)
                powerload = PowerLoad(description, loaddata, model)

                powerload_list.append(powerload)

            phaseload_inputs = FormattedDict(cls._phaseload_inputs_format)
            phaseload_inputs["phase_list"] = phase_list
            phaseload_inputs["consumption"] = consumption
            phaseload_inputs["normalize_list"] = normalize_list
            phaseload_inputs["powerload_list"] = powerload_list

        elif module == "equilibria":
            phaseload_inputs = EquilibriaImporter.phaseload_inputs(
                variables_map,
            )

        elif module == "pumping":
            phaseload_inputs = PumpingImporter.phaseload_inputs(
                variables_map,
            )

        else:
            raise PowerCycleSystemError(
                "import",
                "Unknown routine for importing a phase load from "
                f"the {module!r} module.",
            )
        return phaseload_inputs

    def _build_phaseloads(self, load_name, phaseload_inputs):
        valid_phases = self.scenario.build_phase_library()

        phase_list = phaseload_inputs["phase_list"]
        consumption = phaseload_inputs["consumption"]
        normalize_list = phaseload_inputs["normalize_list"]
        powerload_list = phaseload_inputs["powerload_list"]

        phaseload_list = []
        lists_zip = zip(phase_list, normalize_list)
        for phase_label, normalization_choice in lists_zip:
            try:
                phase = valid_phases[phase_label]
            except KeyError:
                raise PowerCycleSystemError(
                    "scenario",
                    "It is not possible to build objects of the "
                    "'PhaseLoad' class for phases that are not "
                    "present in the 'scenario' attribute.",
                )

            normalization_flags = [normalization_choice] * len(powerload_list)

            phaseload = PhaseLoad(
                load_name,
                phase,
                powerload_list,
                normalization_flags,
            )

            if consumption:
                phaseload.make_consumption_explicit()

            phaseload_list.append(phaseload)

        return phaseload_list

    def _make_phaseloads_from_config(self, type_config):
        system_loads = Library(list)
        for label, load_config in type_config.items():
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
        system_library = Library(PowerCycleSystem)
        for system_label, system_config in group_config.items():
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

    def _call_error(self):
        raise PowerCycleGroupError()


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

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    _load_types = PowerCycleSystem.list_all_load_types()

    _manager_format = FormattedLibrary.Format(
        {
            "name": str,
            "config_path": str,
            "systems": list,
        }
    )

    def __init__(self, scenario_config_path: str, manager_config_path: str):
        scenario_builder = ScenarioBuilder(scenario_config_path)
        self.scenario = scenario_builder.scenario

        validated_path = validate_file(manager_config_path)
        json_contents = read_json(validated_path)

        self._manager_config = FormattedLibrary(
            self._manager_format,
            dictionary=json_contents,
        )

        self.group_library = self._build_group_library(
            self.scenario,
            json_contents,
        )

    @staticmethod
    def _build_group_library(scenario, manager_config):
        group_library = Library(PowerCycleGroup)
        for group_label, group_inputs in manager_config.items():
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

    def _build_pulseload_of_type(self, load_type):
        valid_types = self._load_types
        if load_type not in valid_types:
            raise PowerCycleManagerError(
                "load-type",
                f"The value {load_type!r} is not a valid load type.",
            )

        pulse = self.scenario.pulse_set[0]
        group_library = self.group_library

        all_phaseloads = []
        for group_label in group_library.keys():
            group = group_library[group_label]
            system_library = group.system_library

            for system_label in system_library.keys():
                system = system_library[system_label]

                loads_property = load_type + "_loads"
                loads_of_type = getattr(system, loads_property)
                system_phaseloads = [v for v in loads_of_type.values()]

                system_phaseloads = unnest_list(system_phaseloads)
                all_phaseloads.append(system_phaseloads)

        all_phaseloads = unnest_list(all_phaseloads)

        pulseload = PulseLoad(load_type, pulse, all_phaseloads)
        return pulseload

    def _build_net_loads(self):
        valid_types = self._load_types

        all_loads = Library(PulseLoad)
        for load_type in valid_types:
            all_loads[load_type] = self._build_pulseload_of_type(load_type)

        net_active = all_loads["active"]
        net_reactive = all_loads["reactive"]
        return net_active, net_reactive

    @property
    def net_active(self):
        """
        Net active power from the power plant, represented as a
        PulseLoad object.
        """
        net_active, _ = self._build_net_loads()
        return net_active

    @property
    def net_reactive(self):
        """
        Net reactive power from the power plant, represented as a
        PulseLoad object.
        """
        _, net_reactive = self._build_net_loads()
        return net_reactive

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

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
        ax = validate_axes(ax)
        net_active, net_reactive = self._build_net_loads()

        ax, active_plot_objects = net_active.plot(
            ax=ax,
            n_points=n_points,
            detailed=False,
            c="r",
            **kwargs,
        )
        ax, reactive_plot_objects = net_reactive.plot(
            ax=ax, n_points=n_points, detailed=False, c="b", **kwargs
        )

        tuple_of_plot_objects = (active_plot_objects, reactive_plot_objects)

        return ax, tuple_of_plot_objects
