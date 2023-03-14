# COPYRIGHT PLACEHOLDER

import copy

import matplotlib.pyplot as plt
import pytest

from bluemira.power_cycle.base import PowerCycleLoadABC
from bluemira.power_cycle.errors import (
    LoadDataError,
    PhaseLoadError,
    PowerCycleABCError,
    PowerCycleLoadABCError,
    PowerLoadError,
    PowerLoadModelError,
    PulseLoadError,
)
from bluemira.power_cycle.net.loads import (
    LoadData,
    PhaseLoad,
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


class TestLoadData:
    tested_class_super = PowerCycleLoadABC
    tested_class_super_error = PowerCycleLoadABCError
    tested_class = LoadData
    tested_class_error = LoadDataError

    def setup_method(self):
        tested_class = self.tested_class

        (
            n_samples,
            sample_names,
            sample_times,
            sample_datas,
        ) = netloads_testkit.inputs_for_loaddata()

        all_samples = []
        for s in range(n_samples):
            time = sample_times[s]
            data = sample_datas[s]
            name = sample_names[s]
            sample = tested_class(name, time, data)
            all_samples.append(sample)
        self.all_samples = all_samples

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("test_attr", ["time", "data"])
    def test_is_increasing(self, test_attr):
        tested_class_error = self.tested_class_error

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
                with pytest.raises(tested_class_error):
                    sample._is_increasing(example_list)

    def test_sanity(self):
        tested_class = self.tested_class
        tested_class_error = self.tested_class_error

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
                        test_instance = tested_class(name, time, data)
                        assert isinstance(test_instance, tested_class)
                    else:
                        with pytest.raises(tested_class_error):
                            test_instance = tested_class(name, time, data)

    def test_null_constructor(self):
        tested_class = self.tested_class

        null_instance = tested_class.null()

        null_time = [0, 1]
        assert null_instance.time == null_time

        null_data = [0, 0]
        assert null_instance.data == null_data

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
        figure_title = "LoadData Plotting"
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
    tested_class = PowerLoadModel
    tested_class_error = PowerLoadModelError

    def test_members(self):
        tested_class = self.tested_class

        all_names = [member.name for member in tested_class]
        all_values = [member.value for member in tested_class]

        for (name, value) in zip(all_names, all_values):
            assert isinstance(name, str)
            assert isinstance(value, str)


class TestPowerLoad:
    tested_class_super = PowerCycleLoadABC
    tested_class_super_error = PowerCycleLoadABCError
    tested_class = PowerLoad
    tested_class_error = PowerLoadError

    def setup_method(self):
        tested_class = self.tested_class

        (
            n_samples,
            sample_names,
            sample_loaddatas,
            sample_models,
        ) = netloads_testkit.inputs_for_powerload()

        all_samples = []
        for s in range(n_samples):
            name = sample_names[s]
            loaddata = sample_loaddatas[s]
            model = sample_models[s]
            sample = tested_class(name, loaddata, model)
            all_samples.append(sample)
        self.sample_loaddatas = sample_loaddatas
        self.sample_models = sample_models
        self.all_samples = all_samples

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    def construct_multisample(self):
        sample_loaddatas = self.sample_loaddatas
        sample_models = self.sample_models

        multisample = PowerLoad(
            "PowerLoad with multiple loaddata & model arguments",
            sample_loaddatas,
            sample_models,
        )
        return multisample

    def test_constructor_with_multiple_arguments(self):
        multisample = self.construct_multisample()
        assert isinstance(multisample, PowerLoad)

    def assert_constructor_fails(
        self,
        test_name,
        test_loaddata,
        test_model,
    ):
        tested_class = self.tested_class
        tested_class_error = self.tested_class_error

        constructor_exceptions = (PowerCycleABCError, tested_class_error)
        with pytest.raises(constructor_exceptions):
            wrong_sample = tested_class(
                test_name,
                test_loaddata,
                test_model,
            )

    def test_validate_loaddata_set(self):
        sample_loaddatas = self.sample_loaddatas
        sample_models = self.sample_models

        right_name = [
            "PowerLoad with a non-LoadData",
            "element in its 'loaddata_set' list argument",
        ]
        right_name = " ".join(right_name)
        right_loaddatas = sample_loaddatas
        right_models = sample_models

        wrong_loaddatas = copy.deepcopy(right_loaddatas)
        wrong_loaddatas[0] = "non-LoadData"
        self.assert_constructor_fails(
            right_name,
            wrong_loaddatas,
            right_models,
        )

    def test_validate_model(self):
        sample_loaddatas = self.sample_loaddatas
        sample_models = self.sample_models

        right_name = [
            "PowerLoad with a non-PowerLoadModel",
            "element in its 'model' list argument",
        ]
        right_name = " ".join(right_name)
        right_loaddatas = sample_loaddatas
        right_models = sample_models

        wrong_models = copy.deepcopy(right_models)
        wrong_models[0] = "non-PowerLoadModel"
        self.assert_constructor_fails(
            right_name,
            right_loaddatas,
            wrong_models,
        )

    def test_sanity(self):
        sample_loaddatas = self.sample_loaddatas
        sample_models = self.sample_models

        right_name = [
            "PowerLoad with different lengths",
            "of 'loaddata_set' & 'model' arguments",
        ]
        right_name = " ".join(right_name)
        right_loaddatas = sample_loaddatas
        right_models = sample_models

        wrong_loaddatas = copy.deepcopy(right_loaddatas)
        wrong_loaddatas.pop()
        self.assert_constructor_fails(
            right_name,
            wrong_loaddatas,
            right_models,
        )

        wrong_models = copy.deepcopy(right_models)
        wrong_models.pop()
        self.assert_constructor_fails(
            right_name,
            right_loaddatas,
            wrong_models,
        )

    def test_null_constructor(self):
        tested_class = self.tested_class

        null_instance = tested_class.null()

        null_loaddata = LoadData.null()
        loaddata_set = null_instance.loaddata_set
        for loaddata in loaddata_set:
            assert loaddata.time == null_loaddata.time
            assert loaddata.data == null_loaddata.data

        null_model = PowerLoadModel["RAMP"]
        model = null_instance.model
        for element in model:
            assert element == null_model

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    def test_single_curve(self):
        tested_class = self.tested_class

        sample_loaddatas = self.sample_loaddatas
        sample_models = self.sample_models

        test_time = netloads_testkit.inputs_for_time_interpolation()
        time_length = len(test_time)

        for loaddata in sample_loaddatas:
            power_points = loaddata.data

            for powerloadmodel in sample_models:
                curve = tested_class._single_curve(
                    loaddata,
                    powerloadmodel,
                    test_time,
                )
                curve_length = len(curve)
                assert curve_length == time_length
                netloads_testkit.assert_is_interpolation(power_points, curve)

    def test_validate_curve_input(self):
        tested_class = self.tested_class
        tested_class_error = self.tested_class_error

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
                time = tested_class._validate_curve_input(argument)
                assert isinstance(time, list)
            else:
                with pytest.raises(tested_class_error):
                    time = tested_class._validate_curve_input(argument)

    def test_curve(self):
        tested_class = self.tested_class

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

            all_data.append(sample.loaddata_set[0].data)
            all_sets.append(sample.loaddata_set[0])
            all_models.append(sample.model[0])
        multiset_load = tested_class("Multi Load", all_sets, all_models)
        multiset_curve = multiset_load.curve(test_time)

        data_maxima = [max(data) for data in all_data]
        data_minima = [min(data) for data in all_data]
        sum_of_maxima = sum(data_maxima)
        sum_of_minima = sum(data_minima)
        extreme_points = [sum_of_minima, sum_of_maxima]
        netloads_testkit.assert_is_interpolation(extreme_points, multiset_curve)

    @staticmethod
    def assert_length_and_data_have_not_changed(old_set, new_set):
        correct_number_of_elements = len(old_set)
        wanted_lenght = correct_number_of_elements
        new_set_lenght = len(new_set)
        length_has_not_changed = new_set_lenght == wanted_lenght
        assert length_has_not_changed

        for p in range(correct_number_of_elements):
            old_data = old_set[p]
            new_data = new_set[p]
            data_has_not_changed = old_data == new_data
            assert data_has_not_changed

    @pytest.mark.parametrize(
        "new_end_time",
        netloads_testkit.attribute_manipulation_examples,
    )
    def test_normalize_time(self, new_end_time):
        all_samples = self.all_samples
        multisample = self.construct_multisample()
        all_samples.append(multisample)
        for sample in all_samples:
            old_loaddata_set = sample.loaddata_set
            sample._normalize_time(new_end_time)
            new_loaddata_set = sample.loaddata_set

            self.assert_length_and_data_have_not_changed(
                old_loaddata_set,
                new_loaddata_set,
            )

    @pytest.mark.parametrize(
        "time_shift",
        netloads_testkit.attribute_manipulation_examples,
    )
    def test_shift_time(self, time_shift):
        all_samples = self.all_samples
        multisample = self.construct_multisample()
        all_samples.append(multisample)
        for sample in all_samples:
            old_loaddata_set = sample.loaddata_set
            sample._shift_time(time_shift)
            new_loaddata_set = sample.loaddata_set

            self.assert_length_and_data_have_not_changed(
                old_loaddata_set,
                new_loaddata_set,
            )

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    def test_intrinsic_time(self):
        tested_class_error = self.tested_class_error

        multisample = self.construct_multisample()
        loaddata_set = multisample.loaddata_set

        intrinsic_time = multisample.intrinsic_time
        for loaddata in loaddata_set:
            loaddata_time = loaddata.intrinsic_time
            for t in loaddata_time:
                assert t in intrinsic_time

        setter_errors = (AttributeError, tested_class_error)
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
        tested_class = self.tested_class

        figure_title = "PowerLoad Addition"
        ax = tools_testkit.prepare_figure(figure_title)

        all_samples = self.all_samples
        sample_loaddatas = self.sample_loaddatas
        sample_models = self.sample_models

        result = sum(all_samples)  # requires both __add__ and __radd__
        assert isinstance(result, tested_class)

        result_loaddata = result.loaddata_set
        assert result_loaddata == sample_loaddatas

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
        tested_class = self.tested_class

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
            assert isinstance(sample, tested_class)

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
    tested_class_super = PowerCycleLoadABC
    tested_class_super_error = PowerCycleLoadABCError
    tested_class = PhaseLoad
    tested_class_error = PhaseLoadError

    def setup_method(self):
        tested_class = self.tested_class

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
            sample = tested_class(name, phase, powerload, normalflag)
            all_samples.append(sample)
        self.sample_phases = sample_phases
        self.sample_powerloads = sample_powerloads
        self.sample_normalflags = sample_normalflags
        self.all_samples = all_samples

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    def construct_multisample(self):
        tested_class = self.tested_class

        sample_phases = self.sample_phases
        sample_powerloads = self.sample_powerloads
        sample_normalflags = self.sample_normalflags

        name = "PhaseLoad with multiple powerload & flag arguments"
        example_phase = sample_phases[0]
        multisample = tested_class(
            name,
            example_phase,
            sample_powerloads,
            sample_normalflags,
        )
        return multisample

    def test_constructor_with_multiple_arguments(self):
        tested_class = self.tested_class

        multisample = self.construct_multisample()
        assert isinstance(multisample, tested_class)

        multisample_phase = multisample.phase
        assert isinstance(multisample_phase, PowerCyclePhase)

    def assert_constructor_fails(
        self,
        test_name,
        test_phase,
        test_powerloads,
        test_normalflags,
    ):
        tested_class = self.tested_class
        tested_class_error = self.tested_class_error

        constructor_exceptions = (PowerCycleABCError, tested_class_error)
        with pytest.raises(constructor_exceptions):
            wrong_sample = tested_class(
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

    def test_null_constructor(self):
        tested_class = self.tested_class

        sample_phases = self.sample_phases

        null_powerload = PowerLoad.null()
        for phase in sample_phases:

            null_instance = tested_class.null(phase)

            powerload_set = null_instance.powerload_set
            for powerload in powerload_set:
                loaddata_set = powerload.loaddata_set
                assert len(loaddata_set) == 1

                loaddata = loaddata_set[0]
                null_loaddata = null_powerload.loaddata_set[0]
                assert loaddata.time == null_loaddata.time
                assert loaddata.data == null_loaddata.data

            assert null_instance.normalize == [True]

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

    def test_normalized_set(self):
        tested_class_error = self.tested_class_error

        multisample = self.construct_multisample()
        normalized_set = multisample._normalized_set

        normalize = multisample.normalize
        n_normalize = len(normalize)

        for n in range(n_normalize):
            normalization_flag = normalize[n]
            powerload = normalized_set[n]
            loaddata_set = powerload.loaddata_set
            for loaddata in loaddata_set:
                norm = loaddata._norm
                if normalization_flag:
                    assert len(norm) != 0
                else:
                    assert len(norm) == 0

        powerload_set = multisample.powerload_set
        setter_errors = (AttributeError, tested_class_error)
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
        tested_class_error = self.tested_class_error

        multisample = self.construct_multisample()
        powerload_set = multisample.powerload_set

        intrinsic_time = multisample.intrinsic_time
        for powerload in powerload_set:
            powerload_time = powerload.intrinsic_time
            for t in powerload_time:
                assert t in intrinsic_time

        setter_errors = (AttributeError, tested_class_error)
        with pytest.raises(setter_errors):
            multisample.intrinsic_time = 0

    def test_normalized_time(self):
        tested_class_error = self.tested_class_error

        multisample = self.construct_multisample()
        normalized_set = multisample._normalized_set

        normalized_time = multisample.normalized_time
        for normal_load in normalized_set:
            normal_load_time = normal_load.intrinsic_time
            for t in normal_load_time:
                assert t in normalized_time

        setter_errors = (AttributeError, tested_class_error)
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

    # ------------------------------------------------------------------
    # ARITHMETICS
    # ------------------------------------------------------------------
    def test_addition(self):
        """
        Tests both '__add__' and '__radd__'.
        """
        tested_class = self.tested_class
        tested_class_error = self.tested_class_error

        figure_title = "PhaseLoad Addition"
        ax = tools_testkit.prepare_figure(figure_title)

        all_samples = self.all_samples
        colors = netloads_testkit.color_order_for_plotting
        colors = iter(colors)

        count = 0
        all_results = []
        list_of_plot_objects = []
        for sample in all_samples:

            ax, sample_plot_objects = sample.plot(
                ax=ax,
                detailed=False,
                c="k",
            )
            list_of_plot_objects.append(sample_plot_objects)

            this_phase = sample.phase
            this_set = sample.powerload_set
            this_normalize = sample.normalize

            result = sample + sample
            assert isinstance(result, tested_class)

            result_phase = result.phase
            assert result_phase == this_phase

            result_set = result.powerload_set
            assert result_set == this_set + this_set

            result_normalize = result.normalize
            assert result_normalize == this_normalize + this_normalize

            result.name = "2x " + sample.name + " (added to itself)"
            result_color = next(colors)
            ax, result_plot_objects = result.plot(
                ax=ax,
                detailed=False,
                c=result_color,
            )

            all_results.append(result)
            list_of_plot_objects.append(result_plot_objects)

        adjust_2d_graph_ranges(ax=ax)
        plt.show()

        with pytest.raises(tested_class_error):
            adding_phaseloads_with_different_phases = sum(all_results)


class TestPulseLoad:
    tested_class_super = PowerCycleLoadABC
    tested_class_super_error = PowerCycleLoadABCError
    tested_class = PulseLoad
    tested_class_error = PulseLoadError

    def setup_method(self):
        tested_class = self.tested_class

        (
            n_samples,
            sample_names,
            sample_phaseloads,
        ) = netloads_testkit.inputs_for_pulseload()

        phase_set = []
        for phaseload in sample_phaseloads:
            phase = phaseload.phase
            phase_set.append(phase)
        pulse = PowerCyclePulse("example pulse", phase_set)

        all_samples = []
        for s in range(n_samples):
            name = sample_names[s]
            phaseload = sample_phaseloads[s]
            sample = tested_class(name, pulse, phaseload)
            all_samples.append(sample)

        self.example_pulse = pulse
        self.sample_phaseloads = sample_phaseloads
        self.all_samples = all_samples

    # ------------------------------------------------------------------
    # CLASS ATTRIBUTES & CONSTRUCTOR
    # ------------------------------------------------------------------

    def construct_multisample(self):
        tested_class = self.tested_class

        sample_phaseloads = self.sample_phaseloads
        pulse = self.example_pulse

        name = "PulseLoad with multiple phaseload arguments"
        multisample = tested_class(
            name,
            pulse,
            sample_phaseloads,
        )
        return multisample

    def test_constructor_with_multiple_arguments(self):
        tested_class = self.tested_class

        multisample = self.construct_multisample()
        assert isinstance(multisample, tested_class)

    def assert_constructor_fails(
        self,
        test_name,
        test_phaseloads,
    ):
        tested_class = self.tested_class

        constructor_exceptions = PowerCycleABCError
        with pytest.raises(constructor_exceptions):
            wrong_sample = tested_class(
                test_name,
                test_phaseloads,
            )

    def test_validate_phaseload_set(self):
        """
        import pprint

        tested_class = self.tested_class

        sample_phaseloads = self.sample_phaseloads

        pulse = self.example_pulse
        phase_library = pulse.build_phase_library()
        phase_library_keys = list(phase_library.keys())
        phase_library_values = list(phase_library.values())

        phaseload_set = tested_class._validate_phaseload_set(
            sample_phaseloads,
            pulse,
        )
        n_phaseloads = len(phaseload_set)
        assert n_phaseloads == len(phase_library)

        assert 0

        for p in range(n_phaseloads):
            phaseload = phaseload_set[p]

            phase_label = phase_library_keys[p]
            phase_in_library = phase_library_values[p]

            check = phaseload.phase.is_equivalent(phase_in_library)
            if check:
                phaseload_is_sum_of_phaseloads_for_phase = True
                assert phaseload_is_sum_of_phaseloads_for_phase
            else:
                phaseload_is_null = True
                assert phaseload_is_null

        """

        """
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
        """

        pass

    def test_null_constructor(self):
        tested_class = self.tested_class

        pulse = self.example_pulse
        phase_set = pulse.phase_set
        n_phases = len(phase_set)

        null_instance = tested_class.null(pulse)
        assert null_instance.pulse == pulse

        phaseload_set = null_instance.phaseload_set
        n_phaseloads = len(phaseload_set)
        assert n_phaseloads == n_phases

    # ------------------------------------------------------------------
    # OPERATIONS
    # ------------------------------------------------------------------

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

    def test_shifted_set(self):
        tested_class_error = self.tested_class_error

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
        setter_errors = (AttributeError, tested_class_error)
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
        tested_class_error = self.tested_class_error

        multisample = self.construct_multisample()
        phaseload_set = multisample.phaseload_set

        intrinsic_time = multisample.intrinsic_time
        for phaseload in phaseload_set:
            phaseload_time = phaseload.intrinsic_time
            for t in phaseload_time:
                assert t in intrinsic_time

        setter_errors = (AttributeError, tested_class_error)
        with pytest.raises(setter_errors):
            multisample.intrinsic_time = 0

    def test_shifted_time(self):
        tested_class_error = self.tested_class_error

        multisample = self.construct_multisample()
        shifted_set = multisample._shifted_set

        shifted_time = multisample.shifted_time
        for shifted_load in shifted_set:
            shifted_load_time = shifted_load.intrinsic_time
            for t in shifted_load_time:
                assert t in shifted_time

        setter_errors = (AttributeError, tested_class_error)
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
