# COPYRIGHT PLACEHOLDER

import copy

import matplotlib.pyplot as plt

# import numpy as np
import pytest

from bluemira.power_cycle.errors import (
    PhaseLoadError,
    PowerCycleABCError,
    PowerDataError,
    PowerLoadError,
    PulseLoadError,
)
from bluemira.power_cycle.net_loads import (
    PhaseLoad,
    PowerData,
    PowerLoad,
    PowerLoadModel,
    PulseLoad,
)
from bluemira.power_cycle.time import PowerCyclePhase, PowerCyclePulse
from bluemira.power_cycle.tools import adjust_2d_graph_ranges
from tests.power_cycle.kits_for_tests import NetLoadsTestKit, TimeTestKit, ToolsTestKit

tools_testkit = ToolsTestKit()
time_testkit = TimeTestKit()
netloads_testkit = NetLoadsTestKit()


class TestPowerData:
    def setup_method(self):
        (
            n_samples,
            sample_names,
            sample_times,
            sample_datas,
        ) = netloads_testkit.inputs_for_powerdata()

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

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------
    def test_normalize_time(self):
        time_examples = netloads_testkit.attribute_manipulation_examples
        rel_tol = None
        abs_tol = None

        list_of_new_end_times = time_examples

        all_samples = self.all_samples
        for sample in all_samples:
            sample_copy = copy.deepcopy(sample)

            for new_end_time in list_of_new_end_times:
                old_time = sample_copy.time
                sample_copy._normalize_time(new_end_time)
                new_time = sample_copy.time

                norm = new_time[-1] / old_time[-1]

                old_time_without_0 = [to for to in old_time if to != 0]
                new_time_without_0 = [tn for tn in new_time if tn != 0]

                time_zip = zip(new_time_without_0, old_time_without_0)
                ratios = [tn / to for (tn, to) in time_zip]
                norm_list = [norm] * len(ratios)

                each_ratio_is_norm = []
                for (r, n) in zip(ratios, norm_list):
                    check = r == pytest.approx(n, rel=rel_tol, abs=abs_tol)
                    each_ratio_is_norm.append(check)

                all_ratios_are_norm = all(each_ratio_is_norm)
                assert all_ratios_are_norm

            all_norms = sample_copy._norm
            assert len(all_norms) == len(list_of_new_end_times)

    def test_shift_time(self):
        time_examples = netloads_testkit.attribute_manipulation_examples
        rel_tol = None
        abs_tol = None

        list_of_time_shifts = time_examples

        all_samples = self.all_samples
        for sample in all_samples:
            sample_copy = copy.deepcopy(sample)

            for time_shift in list_of_time_shifts:
                old_time = sample_copy.time
                sample_copy._shift_time(time_shift)
                new_time = sample_copy.time

                time_zip = zip(old_time, new_time)
                differences = [(tn - to) for (to, tn) in time_zip]

                ts = time_shift
                each_difference_is_ts = []

                for d in differences:
                    check = d == pytest.approx(ts, rel=rel_tol, abs=abs_tol)
                    each_difference_is_ts.append(check)

                all_differences_are_timeshift = all(each_difference_is_ts)
                assert all_differences_are_timeshift

            all_shifts = sample_copy._shift
            assert len(all_shifts) == len(list_of_time_shifts)

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    def test_intrinsic_time(self):
        """
        No new functionality to be tested.
        """
        pass

    def test_plot(self):
        figure_title = "PowerData Plotting"
        ax = tools_testkit.prepare_figure(figure_title)

        colors = netloads_testkit.color_order_for_plotting
        colors = iter(colors)

        all_samples = self.all_samples
        list_of_plot_objects = []
        for sample in all_samples:
            sample_color = next(colors)
            ax, plot_list = sample.plot(ax=ax, c=sample_color)
            list_of_plot_objects.append(plot_list)
        adjust_2d_graph_ranges(ax=ax)
        plt.show()


class TestPowerLoadModel:
    def test_members(self):
        all_names = [member.name for member in PowerLoadModel]
        all_values = [member.value for member in PowerLoadModel]

        for (name, value) in zip(all_names, all_values):
            assert isinstance(name, str)
            assert isinstance(value, str)


class TestPowerLoad:
    def setup_method(self):
        (
            n_samples,
            sample_names,
            sample_powerdatas,
            sample_models,
        ) = netloads_testkit.inputs_for_powerload()

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

    def assert_constructor_fails(
        self,
        test_name,
        test_powerdata,
        test_model,
    ):
        constructor_exceptions = (PowerCycleABCError, PowerLoadError)
        with pytest.raises(constructor_exceptions):
            wrong_sample = PowerLoad(
                test_name,
                test_powerdata,
                test_model,
            )

    def test_validate_powerdata_set(self):
        sample_powerdatas = self.sample_powerdatas
        sample_models = self.sample_models

        right_name = [
            "PowerLoad with a non-PowerData",
            "element in its 'powerdata_set' list argument",
        ]
        right_name = " ".join(right_name)
        right_powerdatas = sample_powerdatas
        right_models = sample_models

        wrong_powerdatas = copy.deepcopy(right_powerdatas)
        wrong_powerdatas[0] = "non-PowerData"
        self.assert_constructor_fails(
            right_name,
            wrong_powerdatas,
            right_models,
        )

    def test_validate_model(self):
        sample_powerdatas = self.sample_powerdatas
        sample_models = self.sample_models

        right_name = [
            "PowerLoad with a non-PowerLoadModel",
            "element in its 'model' list argument",
        ]
        right_name = " ".join(right_name)
        right_powerdatas = sample_powerdatas
        right_models = sample_models

        wrong_models = copy.deepcopy(right_models)
        wrong_models[0] = "non-PowerLoadModel"
        self.assert_constructor_fails(
            right_name,
            right_powerdatas,
            wrong_models,
        )

    def test_sanity(self):
        sample_powerdatas = self.sample_powerdatas
        sample_models = self.sample_models

        right_name = [
            "PowerLoad with different lengths",
            "of 'powerdata_set' & 'model' arguments",
        ]
        right_name = " ".join(right_name)
        right_powerdatas = sample_powerdatas
        right_models = sample_models

        wrong_powerdatas = copy.deepcopy(right_powerdatas)
        wrong_powerdatas.pop()
        self.assert_constructor_fails(
            right_name,
            wrong_powerdatas,
            right_models,
        )

        wrong_models = copy.deepcopy(right_models)
        wrong_models.pop()
        self.assert_constructor_fails(
            right_name,
            right_powerdatas,
            wrong_models,
        )

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    def test_single_curve(self):
        sample_powerdatas = self.sample_powerdatas
        sample_models = self.sample_models

        test_time = netloads_testkit.inputs_for_time_interpolation()
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
                netloads_testkit.assert_is_interpolation(power_points, curve)

    def test_validate_curve_input(self):
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
                time = PowerLoad._validate_curve_input(argument)
                assert isinstance(time, list)
            else:
                with pytest.raises(PowerLoadError):
                    time = PowerLoad._validate_curve_input(argument)

    def test_curve(self):
        all_samples = self.all_samples
        test_time = netloads_testkit.inputs_for_time_interpolation()
        time_length = len(test_time)

        all_data = []
        all_sets = []
        all_models = []
        for sample in all_samples:
            curve = sample.curve(test_time)
            curve_length = len(curve)
            assert curve_length == time_length

            all_data.append(sample.powerdata_set[0].data)
            all_sets.append(sample.powerdata_set[0])
            all_models.append(sample.model[0])
        multiset_load = PowerLoad("Multi Load", all_sets, all_models)
        multiset_curve = multiset_load.curve(test_time)

        data_maxima = [max(data) for data in all_data]
        data_minima = [min(data) for data in all_data]
        sum_of_maxima = sum(data_maxima)
        sum_of_minima = sum(data_minima)
        extreme_points = [sum_of_minima, sum_of_maxima]
        netloads_testkit.assert_is_interpolation(extreme_points, multiset_curve)

    @pytest.mark.parametrize(
        "new_end_time",
        netloads_testkit.attribute_manipulation_examples,
    )
    def test_normalize_time(self, new_end_time):
        all_samples = self.all_samples
        multisample = self.construct_multisample()
        all_samples.append(multisample)
        for sample in all_samples:
            old_powerdata_set = sample.powerdata_set
            sample._normalize_time(new_end_time)
            new_powerdata_set = sample.powerdata_set

            # Assert length of 'powerdata_set' has not changed
            correct_number_of_powerdatas = len(old_powerdata_set)
            assert len(new_powerdata_set) == correct_number_of_powerdatas

            # Assert 'data' has not changed
            for p in range(correct_number_of_powerdatas):
                old_data = old_powerdata_set[p]
                new_data = new_powerdata_set[p]
                assert old_data == new_data

    @pytest.mark.parametrize(
        "time_shift",
        netloads_testkit.attribute_manipulation_examples,
    )
    def test_shift_time(self, time_shift):
        all_samples = self.all_samples
        multisample = self.construct_multisample()
        all_samples.append(multisample)
        for sample in all_samples:
            old_powerdata_set = sample.powerdata_set
            sample._shift_time(time_shift)
            new_powerdata_set = sample.powerdata_set

            # Assert length of 'powerdata_set' has not changed
            correct_number_of_powerdatas = len(old_powerdata_set)
            assert len(new_powerdata_set) == correct_number_of_powerdatas

            # Assert 'data' has not changed
            for p in range(correct_number_of_powerdatas):
                old_data = old_powerdata_set[p]
                new_data = new_powerdata_set[p]
                assert old_data == new_data

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    def test_intrinsic_time(self):
        multisample = self.construct_multisample()
        powerdata_set = multisample.powerdata_set

        intrinsic_time = multisample.intrinsic_time
        for powerdata in powerdata_set:
            powerdata_time = powerdata.intrinsic_time
            for t in powerdata_time:
                assert t in intrinsic_time

        setter_errors = (AttributeError, PowerLoadError)
        with pytest.raises(setter_errors):
            multisample.intrinsic_time = 0

    @pytest.mark.parametrize("detailed_plot_flag", [False, True])
    def test_plot(self, detailed_plot_flag):
        figure_title = "'detailed' flag = " + str(detailed_plot_flag)
        figure_title = "PowerLoad Plotting (" + figure_title + ")"
        ax = tools_testkit.prepare_figure(figure_title)

        colors = netloads_testkit.color_order_for_plotting
        colors = iter(colors)

        all_samples = self.all_samples
        list_of_plot_objects = []
        for sample in all_samples:
            sample_color = next(colors)
            ax, plot_list = sample.plot(
                ax=ax,
                detailed=detailed_plot_flag,
                c=sample_color,
            )
            list_of_plot_objects.append(plot_list)
        adjust_2d_graph_ranges(ax=ax)
        plt.show()

    # ------------------------------------------------------------------
    # ARITHMETICS
    # ------------------------------------------------------------------

    def test_addition(self):
        """
        Tests both '__add__' and '__radd__'.
        """
        figure_title = "PowerLoad Addition"
        ax = tools_testkit.prepare_figure(figure_title)

        all_samples = self.all_samples
        sample_powerdatas = self.sample_powerdatas
        sample_models = self.sample_models

        result = sum(all_samples)  # requires both __add__ and __radd__
        assert isinstance(result, PowerLoad)

        result_powerdata = result.powerdata_set
        assert result_powerdata == sample_powerdatas

        result_powerloadmodel = result.model
        assert result_powerloadmodel == sample_models

        ax, list_of_plot_objects = result.plot(
            ax=ax,
            detailed=True,
            c="b",
        )
        adjust_2d_graph_ranges(ax=ax)
        plt.show()

    @pytest.mark.parametrize("number", [2, 5])
    def test_multiplication(self, number):
        """
        Tests both '__mul__' and '__truediv__'.
        """
        rel_tol = None
        abs_tol = None

        num_spec = "x" + str(number) + " & /" + str(number)
        figure_title = "PowerLoad Multiplication (" + num_spec + ")"
        ax = tools_testkit.prepare_figure(figure_title)

        colors = netloads_testkit.color_order_for_plotting
        colors = iter(colors)

        multisample = self.construct_multisample()
        test_time = multisample.time
        curve = multisample.curve(test_time)

        lesser_multisample = multisample / number
        lesser_multisample.name = "Divided sample"
        lesser_curve = lesser_multisample.curve(test_time)

        greater_multisample = multisample * number
        greater_multisample.name = "Multiplied sample"
        greater_curve = greater_multisample.curve(test_time)

        length_time = len(test_time)
        for t in range(length_time):
            curve_point = curve[t]
            lesser_point = lesser_curve[t]
            greater_point = greater_curve[t]

            n = number
            cp = curve_point
            lp = lesser_point
            gp = greater_point

            assert lp == pytest.approx(cp / n, rel=rel_tol, abs=abs_tol)
            assert gp == pytest.approx(cp * n, rel=rel_tol, abs=abs_tol)

        samples_to_plot = [
            multisample,
            lesser_multisample,
            greater_multisample,
        ]
        n_samples = len(samples_to_plot)
        list_of_plot_objects = []
        for s in range(n_samples):
            sample = samples_to_plot[s]
            assert isinstance(sample, PowerLoad)

            sample_color = next(colors)
            ax, sample_plot_list = sample.plot(
                ax=ax,
                detailed=False,
                c=sample_color,
            )
            list_of_plot_objects.append(sample_plot_list)
        adjust_2d_graph_ranges(ax=ax)
        plt.show()


class TestPhaseLoad:
    def setup_method(self):
        (
            n_samples,
            sample_names,
            sample_phases,
            sample_powerloads,
            sample_normalflags,
        ) = netloads_testkit.inputs_for_phaseload()

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

        multisample_phase = multisample.phase
        assert isinstance(multisample_phase, PowerCyclePhase)

    def assert_constructor_fails(
        self,
        test_name,
        test_phase,
        test_powerloads,
        test_normalflags,
    ):
        constructor_exceptions = (PowerCycleABCError, PhaseLoadError)
        with pytest.raises(constructor_exceptions):
            wrong_sample = PhaseLoad(
                test_name,
                test_phase,
                test_powerloads,
                test_normalflags,
            )

    def test_validate_phase(self):
        sample_phases = self.sample_phases
        sample_powerloads = self.sample_powerloads
        sample_normalflags = self.sample_normalflags

        right_name = [
            "PhaseLoad with a non-PowerCyclePhase",
            "element in its 'phase' argument",
        ]
        right_name = " ".join(right_name)
        right_phase = sample_phases[0]
        right_phaseloads = sample_powerloads
        right_normalflags = sample_normalflags

        wrong_phase = "non-PowerCyclePhase"
        self.assert_constructor_fails(
            right_name,
            wrong_phase,
            right_phaseloads,
            right_normalflags,
        )

    def test_validate_powerload_set(self):
        sample_phases = self.sample_phases
        sample_powerloads = self.sample_powerloads
        sample_normalflags = self.sample_normalflags

        right_name = [
            "PhaseLoad with a non-PowerLoad",
            "element in its 'powerload_set' list argument",
        ]
        right_name = " ".join(right_name)
        example_phase = sample_phases[0]
        right_phaseloads = sample_powerloads
        right_normalflags = sample_normalflags

        wrong_phaseloads = copy.deepcopy(right_phaseloads)
        wrong_phaseloads[0] = "non-PowerLoad"
        self.assert_constructor_fails(
            right_name,
            example_phase,
            wrong_phaseloads,
            right_normalflags,
        )

    def test_validate_normalize(self):
        sample_phases = self.sample_phases
        sample_powerloads = self.sample_powerloads
        sample_normalflags = self.sample_normalflags

        right_name = [
            "PhaseLoad with a non-boolean",
            "element in its 'normalize' list argument",
        ]
        right_name = " ".join(right_name)
        example_phase = sample_phases[0]
        right_phaseloads = sample_powerloads
        right_normalflags = sample_normalflags

        wrong_normalflags = copy.deepcopy(right_normalflags)
        wrong_normalflags[0] = "non-boolean"
        self.assert_constructor_fails(
            right_name,
            example_phase,
            right_phaseloads,
            wrong_normalflags,
        )

    def test_sanity(self):
        sample_phases = self.sample_phases
        sample_powerloads = self.sample_powerloads
        sample_normalflags = self.sample_normalflags

        right_name = [
            "PhaseLoad with different lengths",
            "of 'powerload_set' & 'normalize' arguments",
        ]
        right_name = " ".join(right_name)
        example_phase = sample_phases[0]
        right_phaseloads = sample_powerloads
        right_normalflags = sample_normalflags

        wrong_phaseloads = copy.deepcopy(right_phaseloads)
        wrong_phaseloads.pop()
        self.assert_constructor_fails(
            right_name,
            example_phase,
            wrong_phaseloads,
            right_normalflags,
        )

        wrong_normalflags = copy.deepcopy(right_normalflags)
        wrong_normalflags.pop()
        self.assert_constructor_fails(
            right_name,
            example_phase,
            right_phaseloads,
            wrong_normalflags,
        )

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    def test_normalized_set(self):
        multisample = self.construct_multisample()
        normalized_set = multisample._normalized_set

        normalize = multisample.normalize
        n_normalize = len(normalize)

        for n in range(n_normalize):
            normalization_flag = normalize[n]
            powerload = normalized_set[n]
            powerdata_set = powerload.powerdata_set
            for powerdata in powerdata_set:
                norm = powerdata._norm
                if normalization_flag:
                    assert len(norm) != 0
                else:
                    assert len(norm) == 0

        powerload_set = multisample.powerload_set
        setter_errors = (AttributeError, PhaseLoadError)
        with pytest.raises(setter_errors):
            multisample._normalized_set = powerload_set

    def test_private_curve(self):
        """
        No new functionality to be tested.
        """
        pass

    def test_public_curve(self):
        """
        No new functionality to be tested.
        """
        pass

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------
    def test_intrinsic_time(self):
        multisample = self.construct_multisample()
        powerload_set = multisample.powerload_set

        intrinsic_time = multisample.intrinsic_time
        for powerload in powerload_set:
            powerload_time = powerload.intrinsic_time
            for t in powerload_time:
                assert t in intrinsic_time

        setter_errors = (AttributeError, PhaseLoadError)
        with pytest.raises(setter_errors):
            multisample.intrinsic_time = 0

    def test_normalized_time(self):
        multisample = self.construct_multisample()
        normalized_set = multisample._normalized_set

        normalized_time = multisample.normalized_time
        for normal_load in normalized_set:
            normal_load_time = normal_load.intrinsic_time
            for t in normal_load_time:
                assert t in normalized_time

        setter_errors = (AttributeError, PhaseLoadError)
        with pytest.raises(setter_errors):
            multisample.normalized_time = 0

    def test_private_plot(self):
        """
        Tested in 'test_public_plot'.
        """
        pass

    @pytest.mark.parametrize("color_index", [2])
    @pytest.mark.parametrize("detailed_plot_flag", [False, True])
    def test_public_plot(self, color_index, detailed_plot_flag):

        figure_title = "'detailed' flag = " + str(detailed_plot_flag)
        figure_title = "PhaseLoad Plotting (" + figure_title + ")"
        ax = tools_testkit.prepare_figure(figure_title)

        colors = netloads_testkit.color_order_for_plotting
        sample_color = colors[color_index]

        multisample = self.construct_multisample()
        ax, list_of_plot_objects = multisample.plot(
            ax=ax,
            detailed=detailed_plot_flag,
            c=sample_color,
        )
        adjust_2d_graph_ranges(ax=ax)
        plt.show()


class TestPulseLoad:
    def setup_method(self):
        (
            n_samples,
            sample_names,
            sample_phaseloads,
        ) = netloads_testkit.inputs_for_pulseload()

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

    def assert_constructor_fails(
        self,
        test_name,
        test_phaseloads,
    ):
        constructor_exceptions = PowerCycleABCError
        with pytest.raises(constructor_exceptions):
            wrong_sample = PulseLoad(
                test_name,
                test_phaseloads,
            )

    def test_validate_phaseload_set(self):
        sample_phaseloads = self.sample_phaseloads

        right_name = [
            "PulseLoad with a non-PhaseLoad",
            "element in its 'phaseload_set' list argument",
        ]
        right_name = " ".join(right_name)
        right_phaseloads = sample_phaseloads

        wrong_phaseloads = copy.deepcopy(right_phaseloads)
        wrong_phaseloads[0] = "non-PhaseLoad"
        self.assert_constructor_fails(
            right_name,
            wrong_phaseloads,
        )

    def test_build_pulse(self):
        sample_phaseloads = self.sample_phaseloads
        n_phaseloads = len(sample_phaseloads)

        multisample = self.construct_multisample()
        built_pulse = multisample.pulse
        assert isinstance(built_pulse, PowerCyclePulse)

        pulse_phases = built_pulse.phase_set
        n_phases = len(pulse_phases)

        assert n_phaseloads == n_phases
        for i in range(n_phases):
            current_phaseload = sample_phaseloads[i]
            current_phase = pulse_phases[i]
            assert current_phaseload.phase == current_phase

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    def test_shifted_set(self):
        rel_tol = None
        abs_tol = None

        multisample = self.construct_multisample()
        shifted_set = multisample._shifted_set

        time_memory = []
        for shifted_load in shifted_set:
            current_phase = shifted_load.phase
            current_phase_duration = current_phase.duration

            powerload_set = shifted_load.powerload_set
            current_time = shifted_load._build_time_from_power_set(powerload_set)
            time_memory.append(current_time)

            first_time = current_time[0]
            last_time = current_time[-1]
            time_span = last_time - first_time

            spn = time_span
            dur = current_phase_duration
            assert spn == pytest.approx(dur, rel=rel_tol, abs=abs_tol)

        normalized_set = multisample.phaseload_set[0]._normalized_set
        setter_errors = (AttributeError, PulseLoadError)
        with pytest.raises(setter_errors):
            multisample._shifted_set = normalized_set

    def test_curve(self):
        rel_tol = None
        abs_tol = None

        multisample = self.construct_multisample()
        shifted_time = multisample.shifted_time
        epsilon = multisample.epsilon

        rel_tol = epsilon

        original_time = shifted_time
        modified_time, curve = multisample.curve(original_time)
        assert len(modified_time) >= len(original_time)

        n_modified = len(modified_time)
        presence_list = []
        for i in range(n_modified):

            mt = modified_time[i]
            present_in_original = mt in modified_time
            presence_list.append(present_in_original)

            if not present_in_original:
                next_mt = mt = modified_time[i + 1]
                assert mt == pytest.approx(next_mt, rel=rel_tol, abs=abs_tol)

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    def test_intrinsic_time(self):
        multisample = self.construct_multisample()
        phaseload_set = multisample.phaseload_set

        intrinsic_time = multisample.intrinsic_time
        for phaseload in phaseload_set:
            phaseload_time = phaseload.intrinsic_time
            for t in phaseload_time:
                assert t in intrinsic_time

        setter_errors = (AttributeError, PulseLoadError)
        with pytest.raises(setter_errors):
            multisample.intrinsic_time = 0

    def test_shifted_time(self):
        multisample = self.construct_multisample()
        shifted_set = multisample._shifted_set

        shifted_time = multisample.shifted_time
        for shifted_load in shifted_set:
            shifted_load_time = shifted_load.intrinsic_time
            for t in shifted_load_time:
                assert t in shifted_time

        setter_errors = (AttributeError, PulseLoadError)
        with pytest.raises(setter_errors):
            multisample.intrinsic_time = 0

    def test_plot_phase_delimiters(self):
        """
        Tested in 'test_plot'.
        """
        pass

    @pytest.mark.parametrize("color_index", [3])
    @pytest.mark.parametrize("detailed_plot_flag", [False, True])
    def test_plot(self, color_index, detailed_plot_flag):
        figure_title = "'detailed' flag = " + str(detailed_plot_flag)
        figure_title = "PulseLoad Plotting (" + figure_title + ")"
        ax = tools_testkit.prepare_figure(figure_title)

        colors = netloads_testkit.color_order_for_plotting
        sample_color = colors[color_index]

        multisample = self.construct_multisample()
        list_of_plot_objects = multisample.plot(
            ax=ax, detailed=detailed_plot_flag, c=sample_color
        )

        adjust_2d_graph_ranges(ax=ax)
        plt.show()


class TestScenarioLoad:
    pass
