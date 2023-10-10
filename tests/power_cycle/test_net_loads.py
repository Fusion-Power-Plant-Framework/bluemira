# COPYRIGHT PLACEHOLDER

import copy

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bluemira.power_cycle.errors import (
    PowerCycleError,
    PowerLoadError,
)
from bluemira.power_cycle.net.loads import (
    LoadData,
    PhaseLoad,
    PulseLoad,
)
from bluemira.power_cycle.tools import adjust_2d_graph_ranges
from tests.power_cycle.kits_for_tests import NetLoadsTestKit, TimeTestKit, ToolsTestKit

tools_testkit = ToolsTestKit()
time_testkit = TimeTestKit()
netloads_testkit = NetLoadsTestKit()


class TestLoadData:
    def setup_method(self):
        (
            n_samples,
            sample_names,
            sample_times,
            sample_datas,
        ) = netloads_testkit.inputs_for_loaddata()

        self.all_samples = [
            LoadData(sample_names[s], sample_times[s], sample_datas[s])
            for s in range(n_samples)
        ]

    def test_null_constructor(self):
        null_instance = LoadData.null()

        assert np.allclose(null_instance.time, [0, 1])
        assert np.allclose(null_instance.data, [0, 0])

    def test_normalise_time(self):
        time_examples = netloads_testkit.attribute_manipulation_examples
        rel_tol = None
        abs_tol = None

        list_of_new_end_times = time_examples

        all_samples = self.all_samples
        for sample in all_samples:
            sample_copy = copy.deepcopy(sample)

            for new_end_time in list_of_new_end_times:
                old_time = sample_copy.time
                sample_copy._normalise_time(new_end_time)
                new_time = sample_copy.time

                norm = new_time[-1] / old_time[-1]

                old_time_without_0 = [to for to in old_time if to != 0]
                new_time_without_0 = [tn for tn in new_time if tn != 0]

                ratios = [
                    tn / to for tn, to in zip(new_time_without_0, old_time_without_0)
                ]
                norm_list = [norm] * len(ratios)

                each_ratio_is_norm = []
                for r, n in zip(ratios, norm_list):
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

                differences = [(tn - to) for (to, tn) in zip(old_time, new_time)]

                ts = time_shift
                each_difference_is_ts = []

                for d in differences:
                    check = d == pytest.approx(ts, rel=rel_tol, abs=abs_tol)
                    each_difference_is_ts.append(check)

                all_differences_are_timeshift = all(each_difference_is_ts)
                assert all_differences_are_timeshift

            all_shifts = sample_copy._shift
            assert len(all_shifts) == len(list_of_time_shifts)

    def test_make_consumption_explicit(self):
        all_samples = self.all_samples
        for sample in all_samples:
            sample.make_consumption_explicit()
            data = sample.data
            data_is_nonpositive = [v <= 0 for v in data]
            assert all(data_is_nonpositive)

    def test_intrinsic_time(self):
        """
        No new functionality to be tested.
        """

    def test_plot(self):
        ax = tools_testkit.prepare_figure("LoadData Plotting")

        colors = iter(netloads_testkit.color_order_for_plotting)

        list_of_plot_objects = []
        for sample in self.all_samples:
            sample_color = next(colors)
            ax, plot_list = sample.plot(ax=ax, color=sample_color)
            list_of_plot_objects.append(plot_list)
        adjust_2d_graph_ranges(ax=ax)
        plt.show()

    def test_addition(self):
        with pytest.raises(PowerCycleError):
            _ = sum(self.all_samples)


class TestPowerLoad:
    def setup_method(self):
        (
            n_samples,
            sample_names,
            sample_loaddatas,
            sample_loadmodels,
        ) = netloads_testkit.inputs_for_powerload()

        all_samples = []
        for s in range(n_samples):
            name = sample_names[s]
            loaddata = sample_loaddatas[s]
            loadmodel = sample_loadmodels[s]
            sample = PowerLoad(name, loaddata, loadmodel)
            all_samples.append(sample)
        self.sample_loaddatas = sample_loaddatas
        self.sample_loadmodels = sample_loadmodels
        self.all_samples = all_samples

    def construct_multisample(self):
        sample_loaddatas = self.sample_loaddatas
        sample_loadmodels = self.sample_loadmodels

        return PowerLoad(
            "PowerLoad with multiple loaddata & loadmodel arguments",
            sample_loaddatas,
            sample_loadmodels,
        )

    def test_constructor_with_multiple_arguments(self):
        multisample = self.construct_multisample()
        assert isinstance(multisample, PowerLoad)

    def test_validate_loaddata_set(self):
        sample_loaddatas = self.sample_loaddatas
        sample_loadmodels = self.sample_loadmodels

        right_name = [
            "PowerLoad with a non-LoadData",
            "element in its 'loaddata_set' list argument",
        ]
        right_name = " ".join(right_name)
        right_loaddatas = sample_loaddatas
        right_loadmodels = sample_loadmodels

        wrong_loaddatas = copy.deepcopy(right_loaddatas)
        wrong_loaddatas[0] = "non-LoadData"
        with pytest.raises(TypeError):
            PowerLoad(right_name, wrong_loaddatas, right_loadmodels)

    def test_validate_loadmodel_set(self):
        sample_loaddatas = self.sample_loaddatas
        sample_loadmodels = self.sample_loadmodels

        right_name = [
            "PowerLoad with a non-LoadModel",
            "element in its 'loadmodel_set' list argument",
        ]
        right_name = " ".join(right_name)
        right_loaddatas = sample_loaddatas
        right_loadmodels = sample_loadmodels

        wrong_loadmodels = copy.deepcopy(right_loadmodels)
        wrong_loadmodels[0] = "non-LoadModel"
        with pytest.raises(KeyError):
            PowerLoad(right_name, right_loaddatas, wrong_loadmodels)

    def test_sanity(self):
        sample_loaddatas = self.sample_loaddatas
        sample_loadmodels = self.sample_loadmodels

        right_name = [
            "PowerLoad with different lengths",
            "of 'loaddata_set' & 'loadmodel' arguments",
        ]
        right_name = " ".join(right_name)
        right_loaddatas = sample_loaddatas
        right_loadmodels = sample_loadmodels

        wrong_loaddatas = copy.deepcopy(right_loaddatas)
        wrong_loaddatas.pop()
        with pytest.raises(PowerLoadError):
            PowerLoad(right_name, wrong_loaddatas, right_loadmodels)

        wrong_loadmodels = copy.deepcopy(right_loadmodels)
        wrong_loadmodels.pop()
        with pytest.raises(PowerLoadError):
            PowerLoad(right_name, right_loaddatas, wrong_loadmodels)

    def test_null_constructor(self):
        null_instance = PowerLoad.null()
        null_loaddata = LoadData.null()

        for loaddata in null_instance.loaddata_set:
            assert np.allclose(loaddata.time, null_loaddata.time)
            assert np.allclose(loaddata.data, null_loaddata.data)

        for element in null_instance.loadmodel_set:
            assert element == LoadModel["RAMP"]

    def test_single_curve(self):
        test_time = netloads_testkit.inputs_for_time_interpolation()
        time_length = len(test_time)

        for loaddata in self.sample_loaddatas:
            power_points = loaddata.data

            for loadmodel in self.sample_loadmodels:
                curve = PowerLoad._single_curve(
                    loaddata,
                    loadmodel,
                    test_time,
                )
                assert len(curve) == time_length
                netloads_testkit.assert_is_interpolation(power_points, curve)

    def test_curve(self):
        all_samples = self.all_samples
        test_time = netloads_testkit.inputs_for_time_interpolation()
        time_length = len(test_time)

        all_data = []
        all_sets = []
        all_models = []
        for sample in all_samples:
            assert len(sample.curve(test_time)) == time_length

            all_data.append(sample.loaddata_set[0].data)
            all_sets.append(sample.loaddata_set[0])
            all_models.append(sample.loadmodel_set[0])

        netloads_testkit.assert_is_interpolation(
            [
                sum([min(data) for data in all_data]),
                sum([max(data) for data in all_data]),
            ],
            PowerLoad("Multi Load", all_sets, all_models).curve(test_time),
        )

    @staticmethod
    def assert_length_and_data_have_not_changed(old_set, new_set):
        assert len(new_set) == len(old_set)
        for p in range(len(old_set)):
            assert old_set[p] == new_set[p]

    @pytest.mark.parametrize(
        "new_end_time",
        netloads_testkit.attribute_manipulation_examples,
    )
    def test_normalise_time(self, new_end_time):
        self.all_samples.append(self.construct_multisample())
        for sample in self.all_samples:
            old_loaddata_set = sample.loaddata_set
            sample._normalise_time(new_end_time)
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

    def test_make_consumption_explicit(self):
        all_samples = self.all_samples
        for sample in all_samples:
            sample.make_consumption_explicit()

            loaddata_set = sample.loaddata_set
            for loaddata in loaddata_set:
                data = loaddata.data
                data_is_nonpositive = [v <= 0 for v in data]
                assert all(data_is_nonpositive)

    def test_intrinsic_time(self):
        multisample = self.construct_multisample()
        intrinsic_time = multisample.intrinsic_time
        for loaddata in multisample.loaddata_set:
            for t in loaddata.intrinsic_time:
                assert t in intrinsic_time

    @pytest.mark.parametrize("detailed_plot_flag", [False, True])
    def test_plot(self, detailed_plot_flag):
        ax = tools_testkit.prepare_figure(
            f"PowerLoad Plotting ('detailed' flag = {detailed_plot_flag})"
        )
        colors = iter(netloads_testkit.color_order_for_plotting)
        list_of_plot_objects = []
        for sample in self.all_samples:
            sample_color = next(colors)
            ax, plot_list = sample.plot(
                ax=ax,
                detailed=detailed_plot_flag,
                color=sample_color,
            )
            list_of_plot_objects.append(plot_list)
        adjust_2d_graph_ranges(ax=ax)
        plt.show()

    def test_addition(self):
        """
        Tests both '__add__' and '__radd__'.
        """

        figure_title = "PowerLoad Addition"
        ax = tools_testkit.prepare_figure(figure_title)

        result = sum(self.all_samples)  # requires both __add__ and __radd__
        assert isinstance(result, PowerLoad)
        assert result.loaddata_set == self.sample_loaddatas
        assert result.loadmodel_set == self.sample_loadmodels

        ax = result.plot(
            ax=ax,
            detailed=True,
            color="b",
        )
        adjust_2d_graph_ranges(ax=ax)
        plt.show()

    @pytest.mark.parametrize("number", [2, 5])
    def test_multiplication(self, number):
        """
        Tests both '__mul__' and '__truediv__'.
        """
        ax = tools_testkit.prepare_figure(
            f"PowerLoad Multiplication (x{number} & /{number})"
        )
        colors = iter(netloads_testkit.color_order_for_plotting)

        multisample = self.construct_multisample()
        test_time = multisample.intrinsic_time
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

            assert lp == pytest.approx(cp / n)
            assert gp == pytest.approx(cp * n)

        list_of_plot_objects = []
        for sample in [multisample, lesser_multisample, greater_multisample]:
            assert isinstance(sample, PowerLoad)

            sample_color = next(colors)
            ax, sample_plot_list = sample.plot(
                ax=ax,
                detailed=False,
                color=sample_color,
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
        return PhaseLoad(
            "PhaseLoad with multiple powerload & flag arguments",
            self.sample_phases[0],
            self.sample_powerloads,
            self.sample_normalflags,
        )

    def test_constructor_with_multiple_arguments(self):
        multisample = self.construct_multisample()
        assert isinstance(multisample, PhaseLoad)
        assert isinstance(multisample.phase, PowerCyclePhase)

    def assert_constructor_fails(
        self,
        test_name,
        test_phase,
        test_powerloads,
        test_normalflags,
    ):
        with pytest.raises(PhaseLoadError):
            PhaseLoad(test_name, test_phase, test_powerloads, test_normalflags)

    def test_validate_phase(self):
        right_name = (
            "PhaseLoad with a non-PowerCyclePhase element in its 'phase' argument"
        )
        with pytest.raises(TypeError):
            PhaseLoad(
                right_name,
                "non-PowerCyclePhase",
                self.sample_powerloads,
                self.sample_normalflags,
            )

    def test_validate_powerload_set(self):
        right_name = (
            "PhaseLoad with a non-PowerLoad element in its 'powerload_set' list argument"
        )
        wrong_phaseloads = copy.deepcopy(self.sample_powerloads)
        wrong_phaseloads[0] = "non-PowerLoad"
        with pytest.raises(TypeError):
            PhaseLoad(
                right_name,
                self.sample_phases[0],
                wrong_phaseloads,
                self.sample_normalflags,
            )

    def test_validate_normalise(self):
        right_name = (
            "PhaseLoad with a non-boolean element in its 'normalise' list argument"
        )

        wrong_normalflags = copy.deepcopy(self.sample_normalflags)
        wrong_normalflags[0] = "non-boolean"
        with pytest.raises(ValueError):
            PhaseLoad(
                right_name,
                self.sample_phases[0],
                self.sample_powerloads,
                wrong_normalflags,
            )

    def test_sanity(self):
        right_name = (
            "PhaseLoad with different lengths of 'powerload_set' & 'normalise' arguments"
        )

        wrong_phaseloads = copy.deepcopy(self.sample_powerloads)
        wrong_phaseloads.pop()
        self.assert_constructor_fails(
            right_name,
            self.sample_phases[0],
            wrong_phaseloads,
            self.sample_normalflags,
        )

        wrong_normalflags = copy.deepcopy(self.sample_normalflags)
        wrong_normalflags.pop()
        self.assert_constructor_fails(
            right_name,
            self.sample_phases[0],
            self.sample_powerloads,
            wrong_normalflags,
        )

    def test_null_constructor(self):
        null_powerload = PowerLoad.null()
        for phase in self.sample_phases:
            null_instance = PhaseLoad.null(phase)

            powerload_set = null_instance.powerload_set
            for powerload in powerload_set:
                loaddata_set = powerload.loaddata_set
                assert len(loaddata_set) == 1

                loaddata = loaddata_set[0]
                null_loaddata = null_powerload.loaddata_set[0]
                assert np.allclose(loaddata.time, null_loaddata.time)
                assert np.allclose(loaddata.data, null_loaddata.data)

            assert null_instance.normalise == [True]

    def test_normalised_set(self):
        multisample = self.construct_multisample()
        normalised_set = multisample._normalised_set
        normalise = multisample.normalise

        for n in range(len(normalise)):
            normalization_flag = normalise[n]
            powerload = normalised_set[n]
            loaddata_set = powerload.loaddata_set
            for loaddata in loaddata_set:
                norm = loaddata._norm
                if normalization_flag:
                    assert len(norm) != 0
                else:
                    assert len(norm) == 0

        with pytest.raises((AttributeError, PhaseLoadError)):
            multisample._normalised_set = multisample.powerload_set

    def test_intrinsic_time(self):
        multisample = self.construct_multisample()
        intrinsic_time = multisample.intrinsic_time
        for powerload in multisample.powerload_set:
            powerload_time = powerload.intrinsic_time
            for t in powerload_time:
                assert t in intrinsic_time

        with pytest.raises((AttributeError, PhaseLoadError)):
            multisample.intrinsic_time = 0

    def test_normalised_time(self):
        multisample = self.construct_multisample()
        normalised_time = multisample.normalised_time
        for normal_load in multisample._normalised_set:
            normal_load_time = normal_load.intrinsic_time
            for t in normal_load_time:
                assert t in normalised_time

        with pytest.raises((AttributeError, PhaseLoadError)):
            multisample.normalised_time = 0

    @pytest.mark.parametrize("color_index", [2])
    @pytest.mark.parametrize("detailed_plot_flag", [False, True])
    def test_public_plot(self, color_index, detailed_plot_flag):
        ax = tools_testkit.prepare_figure(
            f"PhaseLoad Plotting ('detailed' flag = {detailed_plot_flag})"
        )

        multisample = self.construct_multisample()
        ax, list_of_plot_objects = multisample.plot(
            ax=ax,
            detailed=detailed_plot_flag,
            color=netloads_testkit.color_order_for_plotting[color_index],
        )
        adjust_2d_graph_ranges(ax=ax)
        plt.show()

    def test_addition(self):
        """
        Tests both '__add__' and '__radd__'.
        """
        ax = tools_testkit.prepare_figure("PhaseLoad Addition")

        colors = iter(netloads_testkit.color_order_for_plotting)

        all_results = []
        list_of_plot_objects = []
        for sample in self.all_samples:
            ax, sample_plot_objects = sample.plot(
                ax=ax,
                detailed=False,
                color="k",
            )
            list_of_plot_objects.append(sample_plot_objects)

            this_phase = sample.phase
            this_set = sample.powerload_set
            this_normalise = sample.normalise

            result = sample + sample
            assert isinstance(result, PhaseLoad)

            assert result.phase == this_phase
            assert result.powerload_set == this_set + this_set
            assert all(
                result.normalise == np.concatenate([this_normalise, this_normalise])
            )

            result.name = "2x " + sample.name + " (added to itself)"
            result_color = next(colors)
            a = result.plot(
                ax=ax,
                detailed=False,
                color=result_color,
            )

            all_results.append(result)
            list_of_plot_objects.append(result_plot_objects)

        adjust_2d_graph_ranges(ax=ax)
        plt.show()

        # adding_phaseloads_with_different_phases
        with pytest.raises(PhaseLoadError):
            _ = sum(all_results)


class TestPulseLoad:
    def setup_method(self):
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
            sample = PulseLoad(name, pulse, phaseload)
            all_samples.append(sample)

        self.example_pulse = pulse
        self.sample_phaseloads = sample_phaseloads
        self.all_samples = all_samples

    def construct_multisample(self):
        return PulseLoad(
            "PulseLoad with multiple phaseload arguments",
            self.example_pulse,
            self.sample_phaseloads,
        )

    def test_validate_phaseload_set(self):
        phase_library = self.example_pulse.build_phase_library()

        phaseload_set = PulseLoad._validate_phaseload_set(
            self.sample_phaseloads,
            self.example_pulse,
        )
        n_phaseloads = len(phaseload_set)
        assert n_phaseloads == len(phase_library)

        phase_library_values = list(phase_library.values())
        for p in range(n_phaseloads):
            phaseload_in_set = phaseload_set[p]

            phase_in_pulse = phase_library_values[p]

            all_phaseloads_for_phase = []
            for phaseload in self.sample_phaseloads:
                if phaseload.phase == phase_in_pulse:
                    all_phaseloads_for_phase.append(phaseload)

            if len(all_phaseloads_for_phase) == 0:
                final_phaseload_made_for_phase = PhaseLoad.null(phase_in_pulse)
            else:
                final_phaseload_made_for_phase = sum(all_phaseloads_for_phase)

            assert phaseload_in_set == final_phaseload_made_for_phase

    def test_null_constructor(self):
        null_instance = PulseLoad.null(self.example_pulse)
        assert null_instance.pulse == self.example_pulse
        assert len(null_instance.phaseload_set) == len(self.example_pulse.phase_set)

    def test_build_pulse(self):
        built_pulse = self.construct_multisample().pulse
        assert isinstance(built_pulse, PowerCyclePulse)

        pulse_phases = built_pulse.phase_set
        n_phases = len(pulse_phases)

        assert len(self.sample_phaseloads) == n_phases
        for i in range(n_phases):
            current_phaseload = self.sample_phaseloads[i]
            current_phase = pulse_phases[i]
            assert current_phaseload.phase == current_phase

    def test_shifted_set(self):
        rel_tol = None
        abs_tol = None

        multisample = self.construct_multisample()
        shifted_set = multisample._shifted_set

        time_memory = []
        for shifted_load in shifted_set:
            powerload_set = shifted_load.powerload_set
            current_time = shifted_load._build_time_from_load_set(powerload_set)
            time_memory.append(current_time)

            first_time = current_time[0]
            last_time = current_time[-1]
            time_span = last_time - first_time

            assert time_span == pytest.approx(
                shifted_load.phase.duration, rel=rel_tol, abs=abs_tol
            )

        with pytest.raises((AttributeError, PulseLoadError)):
            multisample._shifted_set = multisample.phaseload_set[0]._normalised_set

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

    def test_intrinsic_time(self):
        multisample = self.construct_multisample()
        phaseload_set = multisample.phaseload_set

        intrinsic_time = multisample.intrinsic_time
        for phaseload in phaseload_set:
            phaseload_time = phaseload.intrinsic_time
            for t in phaseload_time:
                assert t in intrinsic_time

        with pytest.raises((AttributeError, PulseLoadError)):
            multisample.intrinsic_time = 0

    def test_shifted_time(self):
        multisample = self.construct_multisample()
        shifted_set = multisample._shifted_set

        shifted_time = multisample.shifted_time
        for shifted_load in shifted_set:
            shifted_load_time = shifted_load.intrinsic_time
            for t in shifted_load_time:
                assert t in shifted_time

        with pytest.raises((AttributeError, PulseLoadError)):
            multisample.intrinsic_time = 0

    @pytest.mark.parametrize("color_index", [3])
    @pytest.mark.parametrize("detailed_plot_flag", [False, True])
    def test_plot(self, color_index, detailed_plot_flag):
        ax = tools_testkit.prepare_figure(
            f"PulseLoad Plotting 'detailed' flag = {detailed_plot_flag}"
        )
        self.construct_multisample().plot(
            ax=ax,
            detailed=detailed_plot_flag,
            color=netloads_testkit.color_order_for_plotting[color_index],
        )

        adjust_2d_graph_ranges(ax=ax)
        plt.show()

    @pytest.mark.parametrize("humor", [False, True])
    def test_addition(self, humor):
        """
        Tests both '__add__' and '__radd__'.
        """
        ax = tools_testkit.prepare_figure("PulseLoad Addition", humor)

        original = self.construct_multisample()
        result = original + original

        assert np.allclose(result.shifted_time, original.shifted_time)
        assert result.pulse == original.pulse

        for original_phaseload, result_phaseload in zip(
            original.phaseload_set, result.phaseload_set
        ):
            sum_of_originals = original_phaseload + original_phaseload
            assert result_phaseload == sum_of_originals

            original_intrinsic_time = original_phaseload.intrinsic_time
            result_intrinsic_time = result_phaseload.intrinsic_time
            assert np.allclose(result_intrinsic_time, original_intrinsic_time)

        negative_original = copy.deepcopy(original)
        negative_original.make_consumption_explicit()
        result_minus_original = result + negative_original

        ax = result.plot(ax=ax, detailed=False, color="y")
        ax = original.plot(ax=ax, detailed=False, color="k")
        ax = result_minus_original.plot(ax=ax, detailed=False, color="c", linestyle="--")

        adjust_2d_graph_ranges(ax=ax)
        plt.show()
