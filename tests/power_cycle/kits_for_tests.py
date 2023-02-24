# COPYRIGHT PLACEHOLDER

import matplotlib.pyplot as plt
import numpy as np

import bluemira.base.constants as constants
from bluemira.power_cycle.net_loads import (
    PhaseLoad,
    PowerData,
    PowerLoad,
    PowerLoadModel,
)
from bluemira.power_cycle.time import PowerCyclePhase, PowerCyclePulse
from bluemira.power_cycle.tools import unnest_list, validate_axes


class ToolsTestKit:
    def __init__(self):
        pass

    @staticmethod
    def prepare_figure(figure_title):
        ax = validate_axes()
        plt.grid()
        plt.title(figure_title)
        return ax


class TimeTestKit:
    def __init__(self):
        pass

    def inputs_for_phase(self):
        """
        Function to create inputs for PowerCyclePhase testing.
        The lists 'input_names' and 'input_breakdowns' must have the
        same length.
        """
        input_names = [
            "Dwell",
            "Transition between dwell and flat-top",
            "Flat-top",
            "Transition between flat-top and dwell",
        ]
        input_breakdowns = [
            {
                "CS-recharge + pumping": constants.raw_uc(10, "minute", "second"),
            },
            {
                "current ramp-up": 157,
                "heating": 19,
            },
            {"plasma burn": constants.raw_uc(2, "hour", "second")},
            {
                "cooling": 123,
                "current ramp-down": 157,
            },
        ]
        assert len(input_names) == len(input_breakdowns)
        n_inputs = len(input_names)

        return (
            n_inputs,
            input_names,
            input_breakdowns,
        )

    def inputs_for_pulse(self):
        """
        Function to create inputs for PowerCyclePulse testing.
        """
        (
            n_inputs,
            input_names,
            input_breakdowns,
        ) = self.inputs_for_phase()

        input_phases = []
        for i in range(n_inputs):
            name = input_names[i]
            breakdown = input_breakdowns[i]
            phase = PowerCyclePhase(name, breakdown)
            input_phases.append(phase)

        return (
            n_inputs,
            input_phases,
        )

    def inputs_for_timeline(self):
        """
        Function to create inputs for PowerCycleTimeline testing.
        """
        (
            _,
            input_phases,
        ) = self.inputs_for_pulse()

        n_pulses = 10

        input_pulses = []
        for p in range(n_pulses):
            name = "Pulse " + str(p)
            phase = input_phases
            pulse = PowerCyclePulse(name, phase)
            input_pulses.append(pulse)

        return (
            n_pulses,
            input_pulses,
        )


class NetLoadsTestKit:
    def __init__(self):

        self.time_testkit = TimeTestKit()

        all_colors = [
            "r",
            "b",
            "g",
            "m",
            "c",
            "y",
        ]
        self.color_order_for_plotting = all_colors
        self.n_colors = len(all_colors)

        self.comparison_relative_tolerance = 1e-6
        self.comparison_absolute_tolerance = 1e-12

        manipulation_examples = [10 ** (e - 5) for e in range(10)]
        self.attribute_manipulation_examples = manipulation_examples

    @staticmethod
    def assert_is_interpolation(original_points, curve):
        """
        Confirm that curve is an interpolation with possibility of
        out-of-bounds values.

        Current simplified approach: no curve value is out of the bounds
        of the original defining interpolation points, except if it is a
        zero ('fill_value' argument of 'interp1d').

        Possibly to be substituted by `unittest.mock`.
        """
        original_max = max(original_points)
        original_min = min(original_points)
        curve_max = max(curve)
        curve_min = min(curve)
        assert (curve_max <= original_max) or (curve_max == 0)
        assert (curve_min >= original_min) or (curve_min == 0)

    def inputs_for_powerdata(self):
        """
        Function to create inputs for PowerData testing.
        The lists 'input_times' and 'input_datas' must have the same
        length.
        """

        input_times = [
            [0, 4, 7, 8],
            [2, 5, 7, 9, 10],
        ]
        input_datas = [
            [6, 9, 7, 8],
            [2, 2, 2, 4, 4],
        ]
        assert len(input_times) == len(input_datas)
        n_inputs = len(input_times)

        input_names = []
        for i in range(n_inputs):
            input_names.append("PowerData " + str(i))

        return (
            n_inputs,
            input_names,
            input_times,
            input_datas,
        )

    def inputs_for_time_interpolation(self):
        """
        Function to create inputs time interpolation testing.
        """
        (
            _,
            _,
            input_times,
            _,
        ) = self.inputs_for_powerdata()

        all_times = unnest_list(input_times)
        minimum_time = min(all_times)
        maximum_time = min(all_times)

        time_extrapolation = 2
        time_step = 0.1

        start = minimum_time - time_extrapolation
        stop = maximum_time + time_extrapolation
        time_vector = np.arange(start, stop, time_step)
        time_list = list(time_vector)
        return time_list

    def inputs_for_powerload(self):
        """
        Function to create inputs for PowerLoad testing, based on the
        function that creates inputs for PowerData testing.
        """
        (
            n_inputs,
            input_names,
            input_times,
            input_datas,
        ) = self.inputs_for_powerdata()

        all_models = [member.name for member in PowerLoadModel]
        n_models = len(all_models)

        input_models = []
        input_powerdatas = []
        for i in range(n_inputs):

            # Cycle through available models to create a model example
            model = all_models[i % n_models]
            model = PowerLoadModel[model]
            input_models.append(model)

            name = input_names[i]
            time = input_times[i]
            data = input_datas[i]

            powerdata = PowerData(name, time, data)
            input_powerdatas.append(powerdata)

        input_names = [name.replace("Data", "Load") for name in input_names]
        return (
            n_inputs,
            input_names,
            input_powerdatas,
            input_models,
        )

    def inputs_for_phaseload(self):
        """
        Function to create inputs for PhaseLoad testing, based on the
        function that creates inputs for PowerLoad testing.
        """
        (
            n_inputs,
            input_names,
            input_powerdatas,
            input_models,
        ) = self.inputs_for_powerload()

        (
            n_phases,
            all_phases,
        ) = self.time_testkit.inputs_for_pulse()

        input_phases = []
        input_powerloads = []
        input_normalflags = []
        for i in range(n_inputs):

            # Cycle through phases to pick a phase example
            phase = all_phases[i % n_phases]
            input_phases.append(phase)

            name = input_names[i]
            powerdata = input_powerdatas[i]
            model = input_models[i]
            powerload = PowerLoad(name, powerdata, model)
            input_powerloads.append(powerload)

            # Cycle through True/False to create a flag example
            normal_flag = bool(i % 2)
            input_normalflags.append(normal_flag)

        input_names = [name.replace("Power", "Phase") for name in input_names]
        return (
            n_inputs,
            input_names,
            input_phases,
            input_powerloads,
            input_normalflags,
        )

    def inputs_for_pulseload(self):
        """
        Function to create inputs for PulseLoad testing, based on the
        function that creates inputs for PhaseLoad testing.
        """
        (
            n_inputs,
            input_phases,
        ) = self.time_testkit.inputs_for_pulse()

        (
            _,
            _,
            _,
            input_powerloads,
            input_normalflags,
        ) = self.inputs_for_phaseload()

        input_names = []
        input_phaseloads = []
        for i in range(n_inputs):
            phase = input_phases[i]
            name = "PhaseLoad for " + phase.name + " (phase)"
            powerloads = input_powerloads
            normalflags = input_normalflags
            phaseload = PhaseLoad(name, phase, powerloads, normalflags)
            input_phaseloads.append(phaseload)

            name = "PulseLoad" + str(i)
            input_names.append(name)

        return (
            n_inputs,
            input_names,
            input_phaseloads,
        )
