# COPYRIGHT PLACEHOLDER

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bluemira.power_cycle.errors import PowerDataError, PowerLoadError  # PhaseLoadError,
from bluemira.power_cycle.net_loads import (
    PhaseLoad,
    PowerData,
    PowerLoad,
    PowerLoadModel,
    PulseLoad,
)
from bluemira.power_cycle.tools import adjust_2d_graph_ranges, unnest_list, validate_axes
from tests.power_cycle.test_powercycle_time import inputs_for_pulse

color_order_for_plotting = [
    "r",
    "b",
    "g",
    "m",
    "c",
    "y",
]
n_colors = len(color_order_for_plotting)


def prepare_figure(figure_title):
    ax = validate_axes()
    plt.grid()
    plt.title(figure_title)
    return ax


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


def inputs_for_powerdata():
    """
    Function to create inputs for PowerData testing.
    The lists 'input_times' and 'input_datas' must have the same length.
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


def inputs_for_time_interpolation():
    """
    Function to create inputs time interpolation testing.
    """

    (
        n_inputs,
        _,
        input_times,
        _,
    ) = inputs_for_powerdata()

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


class TestPowerData:
    def setup_method(self):
        (
            n_samples,
            sample_names,
            sample_times,
            sample_datas,
        ) = inputs_for_powerdata()

        all_samples = []
        for s in range(n_samples):
            time = sample_times[s]
            data = sample_datas[s]
            name = sample_names[s]
            sample = PowerData(name, time, data)
            all_samples.append(sample)
        self.all_samples = all_samples

    @pytest.mark.parametrize("test_attr", ["time", "data"])
    def test_is_increasing(self, test_attr):
        all_samples = self.all_samples
        n_samples = len(all_samples)

        def check_if_increasing(test_list):
            check = []
            for i in range(len(test_list) - 1):
                check.append(test_list[i] <= test_list[i + 1])
            return all(check)

        for s in range(n_samples):
            sample = all_samples[s]
            example_list = getattr(sample, test_attr)
            list_is_increasing = check_if_increasing(example_list)
            if list_is_increasing:
                assert sample._is_increasing(example_list)
            else:
                with pytest.raises(PowerDataError):
                    sample._is_increasing(example_list)

    def test_sanity(self):
        all_samples = self.all_samples
        n_samples = len(all_samples)
        for s in range(n_samples):
            sample = all_samples[s]
            name = sample.name
            list_of_given_length = sample.time
            list_shorter = list_of_given_length[:-1]
            list_longer = list_of_given_length + [10]
            all_lists = [
                list_of_given_length,
                list_shorter,
                list_longer,
            ]
            for time in all_lists:
                for data in all_lists:
                    length_time = len(time)
                    length_data = len(data)
                    if length_time == length_data:
                        test_instance = PowerData(name, time, data)
                        assert isinstance(test_instance, PowerData)
                    else:
                        with pytest.raises(PowerDataError):
                            test_instance = PowerData(name, time, data)

    @pytest.mark.parametrize("new_end_time", [10 ** (e - 5) for e in range(10)])
    def test_normalize_time(self, new_end_time):
        all_samples = self.all_samples
        for sample in all_samples:
            old_time = sample.time
            sample._normalize_time(new_end_time)
            new_time = sample.time

            norm = new_time[-1] / old_time[-1]

            old_time = [to for to in old_time if to != 0]  # filter 0s
            new_time = [tn for tn in new_time if tn != 0]  # filter 0s
            ratios = [tn / to for tn, to in zip(new_time, old_time)]
            assert all([r == norm for r in ratios])

    def test_plot(self):
        figure_title = "PowerData Plotting"
        ax = prepare_figure(figure_title)
        all_samples = self.all_samples
        n_samples = len(all_samples)
        plot_list = []
        for s in range(n_samples):

            # Cycle through available colors
            sample_color = color_order_for_plotting[s % n_colors]

            sample = all_samples[s]
            plot_list.append(sample.plot(ax=ax, c=sample_color))
        adjust_2d_graph_ranges(ax=ax)
        plt.show()  # Run with `pytest --plotting-on` to visualize


class TestPowerLoadModel:
    def test_members(self):
        all_names = [member.name for member in PowerLoadModel]
        all_values = [member.value for member in PowerLoadModel]

        for (name, value) in zip(all_names, all_values):
            assert isinstance(name, str)
            assert isinstance(value, str)


def inputs_for_powerload():
    """
    Function to create inputs for PowerLoad testing, based on the
    function that creates inputs for PowerData testing.
    """

    (
        n_inputs,
        input_names,
        input_times,
        input_datas,
    ) = inputs_for_powerdata()

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


class TestPowerLoad:
    def setup_method(self):
        (
            n_samples,
            sample_names,
            sample_powerdatas,
            sample_models,
        ) = inputs_for_powerload()

        all_samples = []
        for s in range(n_samples):
            name = sample_names[s]
            powerdata = sample_powerdatas[s]
            model = sample_models[s]
            sample = PowerLoad(name, powerdata, model)
            all_samples.append(sample)
        self.sample_powerdatas = sample_powerdatas
        self.sample_models = sample_models
        self.all_samples = all_samples

    def construct_multisample(self):
        sample_powerdatas = self.sample_powerdatas
        sample_models = self.sample_models

        multisample = PowerLoad(
            "PowerLoad with multiple powerdata & model arguments",
            sample_powerdatas,
            sample_models,
        )
        return multisample

    def test_constructor_with_multiple_arguments(self):
        multisample = self.construct_multisample()
        assert isinstance(multisample, PowerLoad)

    def test_sanity(self):
        sample_powerdatas = self.sample_powerdatas
        sample_models = self.sample_models

        max_powerdata_length = len(sample_powerdatas)
        max_powerloadmodel_length = len(sample_models)

        for powerdata in sample_powerdatas:
            base_powerdata_input = powerdata

            for powerloadmodel in sample_models:
                base_powerloadmodel_input = powerloadmodel

                for n_pd in range(max_powerdata_length):
                    data_input = sample_powerdatas[0:n_pd]
                    data_input.insert(0, base_powerdata_input)

                    for n_plm in range(max_powerloadmodel_length):
                        model_input = sample_models[0:n_plm]
                        model_input.insert(0, base_powerloadmodel_input)

                        if n_pd == n_plm:
                            load = PowerLoad(
                                "Test Load",
                                data_input,
                                model_input,
                            )
                            assert isinstance(load, PowerLoad)
                        else:
                            with pytest.raises(PowerLoadError):
                                load = PowerLoad(
                                    "Test Load",
                                    data_input,
                                    model_input,
                                )

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    def test_single_curve(self):
        sample_powerdatas = self.sample_powerdatas
        sample_models = self.sample_models

        test_time = inputs_for_time_interpolation()
        time_length = len(test_time)

        for powerdata in sample_powerdatas:
            power_points = powerdata.data

            for powerloadmodel in sample_models:
                curve = PowerLoad._single_curve(
                    powerdata,
                    powerloadmodel,
                    test_time,
                )
                curve_length = len(curve)
                assert curve_length == time_length
                assert_is_interpolation(power_points, curve)

    def test_validate_time(self):
        test_arguments = [
            None,
            1.2,
            -1.2,
            70,
            -70,
            "some string",
            [1, 2, 3, 4],
            (1, 2, 3, 4),
        ]
        test_arguments = test_arguments + self.all_samples

        for argument in test_arguments:
            if isinstance(argument, (int, float, list)):
                time = PowerLoad._validate_time(argument)
                assert isinstance(time, list)
            else:
                with pytest.raises(PowerLoadError):
                    time = PowerLoad._validate_time(argument)

    def test_curve(self):
        all_samples = self.all_samples
        test_time = inputs_for_time_interpolation()
        time_length = len(test_time)

        all_data = []
        all_sets = []
        all_models = []
        for sample in all_samples:
            curve = sample.curve(test_time)
            curve_length = len(curve)
            assert curve_length == time_length

            all_data.append(sample.data_set[0].data)
            all_sets.append(sample.data_set[0])
            all_models.append(sample.model[0])
        multiset_load = PowerLoad("Multi Load", all_sets, all_models)
        multiset_curve = multiset_load.curve(test_time)

        data_maxima = [max(data) for data in all_data]
        data_minima = [min(data) for data in all_data]
        sum_of_maxima = sum(data_maxima)
        sum_of_minima = sum(data_minima)
        extreme_points = [sum_of_minima, sum_of_maxima]
        assert_is_interpolation(extreme_points, multiset_curve)

    @pytest.mark.parametrize("new_end_time", [10 ** (e - 5) for e in range(10)])
    def test_normalize_time(self, new_end_time):
        all_samples = self.all_samples
        multisample = self.construct_multisample()
        all_samples.append(multisample)
        for sample in all_samples:
            old_data_set = sample.data_set
            sample._normalize_time(new_end_time)
            new_data_set = sample.data_set

            # Assert length of 'data_set' has not changed
            correct_number_of_powerdatas = len(old_data_set)
            assert len(new_data_set) == correct_number_of_powerdatas

            # Assert 'data' has not changed
            for p in range(correct_number_of_powerdatas):
                old_data = old_data_set[p]
                new_data = new_data_set[p]
                assert old_data == new_data

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("detailed_plot_flag", [False, True])
    def test_plot(self, detailed_plot_flag):
        figure_title = "'detailed' flag = " + str(detailed_plot_flag)
        figure_title = "PowerLoad Plotting (" + figure_title + ")"
        ax = prepare_figure(figure_title)

        list_of_plot_objects = []

        all_samples = self.all_samples
        n_samples = len(all_samples)
        for s in range(n_samples):

            # Cycle through available colors
            sample_color = color_order_for_plotting[s % n_colors]

            sample = all_samples[s]
            current_list_of_plot_objects = sample.plot(
                ax=ax, detailed=detailed_plot_flag, c=sample_color
            )
            list_of_plot_objects.append(current_list_of_plot_objects)

        adjust_2d_graph_ranges(ax=ax)
        plt.show()  # Run with `pytest --plotting-on` to visualize

    # ------------------------------------------------------------------
    # ARITHMETICS
    # ------------------------------------------------------------------

    def test_addition(self):
        figure_title = "PowerLoad Addition"
        ax = prepare_figure(figure_title)

        all_samples = self.all_samples
        sample_powerdatas = self.sample_powerdatas
        sample_models = self.sample_models

        result = sum(all_samples)
        assert isinstance(result, PowerLoad)

        result_powerdata = result.data_set
        assert result_powerdata == sample_powerdatas

        result_powerloadmodel = result.model
        assert result_powerloadmodel == sample_models

        list_of_plot_objects = result.plot(ax=ax, detailed=True, c="b")
        adjust_2d_graph_ranges(ax=ax)
        plt.show()  # Run with `pytest --plotting-on` to visualize


def inputs_for_phaseload():
    """
    Function to create inputs for PhaseLoad testing, based on the
    function that creates inputs for PowerLoad testing.
    """

    (
        n_inputs,
        input_names,
        input_powerdatas,
        input_models,
    ) = inputs_for_powerload()

    (
        n_phases,
        all_phases,
    ) = inputs_for_pulse()

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


class TestPhaseLoad:
    def setup_method(self):
        (
            n_samples,
            sample_names,
            sample_phases,
            sample_powerloads,
            sample_normalflags,
        ) = inputs_for_phaseload()

        all_samples = []
        for s in range(n_samples):
            name = sample_names[s]
            phase = sample_phases[s]
            powerload = sample_powerloads[s]
            normalflag = sample_normalflags[s]
            sample = PhaseLoad(name, phase, powerload, normalflag)
            all_samples.append(sample)
        self.sample_phases = sample_phases
        self.sample_powerloads = sample_powerloads
        self.sample_normalflags = sample_normalflags
        self.all_samples = all_samples

    def construct_multisample(self):
        sample_phases = self.sample_phases
        sample_powerloads = self.sample_powerloads
        sample_normalflags = self.sample_normalflags

        name = "PhaseLoad with multiple powerload & flag arguments"
        example_phase = sample_phases[0]
        multisample = PhaseLoad(
            name,
            example_phase,
            sample_powerloads,
            sample_normalflags,
        )
        return multisample

    def test_constructor_with_multiple_arguments(self):
        multisample = self.construct_multisample()
        assert isinstance(multisample, PhaseLoad)

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------
    def test_curve(self):
        all_samples = self.all_samples
        test_time = inputs_for_time_interpolation()
        time_length = len(test_time)

        """
        all_data = []
        all_sets = []
        all_models = []
        for sample in all_samples:
            curve = sample.curve(test_time)
            curve_length = len(curve)
            assert curve_length == time_length

            all_data.append(sample.data_set[0].data)
            all_sets.append(sample.data_set[0])
            all_models.append(sample.model[0])
        multiset_load = PowerLoad("Multi Load", all_sets, all_models)
        multiset_curve = multiset_load.curve(test_time)

        data_maxima = [max(data) for data in all_data]
        data_minima = [min(data) for data in all_data]
        sum_of_maxima = sum(data_maxima)
        sum_of_minima = sum(data_minima)
        extreme_points = [sum_of_minima, sum_of_maxima]
        assert_is_interpolation(extreme_points, multiset_curve)
        """

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("color_index", [2])
    @pytest.mark.parametrize("detailed_plot_flag", [False, True])
    def test_plot(self, color_index, detailed_plot_flag):
        sample_color = color_order_for_plotting[color_index]
        figure_title = "'detailed' flag = " + str(detailed_plot_flag)
        figure_title = "PhaseLoad Plotting (" + figure_title + ")"
        ax = prepare_figure(figure_title)

        multisample = self.construct_multisample()
        list_of_plot_objects = multisample.plot(
            ax=ax, detailed=detailed_plot_flag, c=sample_color
        )

        adjust_2d_graph_ranges(ax=ax)
        plt.show()  # Run with `pytest --plotting-on` to visualize


def inputs_for_pulseload():
    """
    Function to create inputs for PulseLoad testing, based on the
    function that creates inputs for PhaseLoad testing.
    """
    (
        n_inputs,
        input_phases,
    ) = inputs_for_pulse()

    (
        _,
        _,
        _,
        input_powerloads,
        input_normalflags,
    ) = inputs_for_phaseload()

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


class TestPulseLoad:
    def setup_method(self):
        (
            n_samples,
            sample_names,
            sample_phaseloads,
        ) = inputs_for_pulseload()

        all_samples = []
        for s in range(n_samples):
            name = sample_names[s]
            phaseload = sample_phaseloads[s]
            sample = PulseLoad(name, phaseload)
            all_samples.append(sample)
        self.sample_phaseloads = sample_phaseloads
        self.all_samples = all_samples

    def construct_multisample(self):
        sample_phaseloads = self.sample_phaseloads

        name = "PulseLoad with multiple phaseload arguments"
        multisample = PulseLoad(
            name,
            sample_phaseloads,
        )
        return multisample

    def test_constructor_with_multiple_arguments(self):
        multisample = self.construct_multisample()
        assert isinstance(multisample, PulseLoad)

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------
    def test_curve(self):
        pass

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------
    @pytest.mark.parametrize("color_index", [3])
    @pytest.mark.parametrize("detailed_plot_flag", [False, True])
    def test_plot(self, color_index, detailed_plot_flag):
        sample_color = color_order_for_plotting[color_index]
        figure_title = "'detailed' flag = " + str(detailed_plot_flag)
        figure_title = "PulseLoad Plotting (" + figure_title + ")"
        ax = prepare_figure(figure_title)

        multisample = self.construct_multisample()
        list_of_plot_objects = multisample.plot(
            ax=ax, detailed=detailed_plot_flag, c=sample_color
        )

        adjust_2d_graph_ranges(ax=ax)
        plt.show()  # Run with `pytest --plotting-on` to visualize
