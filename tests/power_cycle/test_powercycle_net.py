from pprint import pformat

import matplotlib.pyplot as plt
import numpy as np
import pytest

from bluemira.base.look_and_feel import bluemira_debug
from bluemira.power_cycle.net import (
    NetPowerABC,
    NetPowerABCError,
    PowerData,
    PowerDataError,
    PowerLoad,
    PowerLoadError,
    PowerLoadModel,
)
from bluemira.power_cycle.tools import adjust_2d_graph_ranges, validate_axes


def script_title():
    return "Test Power Cycle 'net'"


def test_NetPowerABCError():
    with pytest.raises(NetPowerABCError):
        raise NetPowerABCError(
            None,
            "Some error in the 'NetPowerABC' class.",
        )


class TestNetPowerABC:
    class SampleConcreteClass(NetPowerABC):
        """
        Inner class that is a dummy concrete class for testing the main
        abstract class of the test.
        """

        pass

    def setup_method(self):
        sample = self.SampleConcreteClass("A sample instance name")
        another_sample = self.SampleConcreteClass("Another name")

        test_arguments = [
            None,
            1.2,
            70,
            "some string",
            [1, 2, 3, 4],
            (1, 2, 3, 4),
            sample,
            another_sample,
        ]

        self.sample = sample
        self.another_sample = another_sample
        self.test_arguments = test_arguments

    def test_validate_n_points(self):
        sample = self.sample
        all_arguments = self.test_arguments
        for argument in all_arguments:
            bluemira_debug(
                f"""
                {script_title()} (NetPowerABC._validate_n_points)

                Argument:
                {pformat(argument)}
                """
            )
            if not argument:
                default_n_points = sample._n_points
                validated_arg = sample._validate_n_points(argument)
                assert validated_arg == default_n_points

            elif (type(argument) is int) or (type(argument) is float):
                validated_arg = sample._validate_n_points(argument)
                assert isinstance(validated_arg, int)

            else:
                with pytest.raises(NetPowerABCError):
                    validated_arg = sample._validate_n_points(argument)

    def test_make_secondary_in_plot(self):
        one_sample = self.sample
        another_sample = self.another_sample

        another_sample._make_secondary_in_plot()
        attr_to_compare = [
            "_text_index",
            "_plot_kwargs",
        ]

        for attribute in attr_to_compare:
            one_attr = getattr(one_sample, attribute)
            another_attr = getattr(another_sample, attribute)
            assert one_attr != another_attr


def test_PowerDataError():
    with pytest.raises(PowerDataError):
        raise PowerDataError(
            None,
            "Some error in the 'PowerData' class.",
        )


class TestPowerData:
    def setup_method(self):
        sample_name = "Sample PowerData Instance"
        sample_time = [0, 4, 7, 8]
        sample_data = [6, 9, 7, 8]
        sample = PowerData(sample_name, sample_time, sample_data)
        self.sample = sample

    def test_is_increasing(self):
        sample = self.sample
        increasing_list_example = sample.time
        non_increasing_list_example = sample.data
        bluemira_debug(
            f"""
            {script_title()} (PowerData._is_increasing)

            Example of increasing list:
            {pformat(increasing_list_example)}

            Example of non-increasing list:
            {pformat(non_increasing_list_example)}
            """
        )
        assert sample._is_increasing(increasing_list_example)
        with pytest.raises(PowerDataError):
            sample._is_increasing(non_increasing_list_example)

    def test_sanity(self):
        sample = self.sample
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
                bluemira_debug(
                    f"""
                    {script_title()} (PowerData._sanity)

                    List used as time vector:
                    {pformat(time)}

                    List used as data vector:
                    {pformat(data)}
                    """
                )

                if length_time == length_data:
                    test_instance = PowerData(name, time, data)
                    assert isinstance(test_instance, PowerData)

                else:
                    with pytest.raises(PowerDataError):
                        test_instance = PowerData(name, time, data)

    def test_plot(self):
        fig = plt.figure()
        plt.grid()
        sample = self.sample
        all_axes = sample.plot()
        bluemira_debug(
            f"""
            {script_title()} (PowerData._plot)

            All plotted objects:
            {pformat(all_axes)}
            """
        )
        adjust_2d_graph_ranges()
        plt.show()  # Run with `pytest --plotting-on` to visualize


class TestPowerLoadModel:
    def test_members(self):

        all_names = [member.name for member in PowerLoadModel]
        all_values = [member.value for member in PowerLoadModel]
        bluemira_debug(
            f"""
            {script_title()} (PowerLoadModel)

            All member names:
            {pformat(all_names)}

            All member values:
            {pformat(all_values)}
            """
        )


def test_PowerLoadError():
    with pytest.raises(PowerLoadError):
        raise PowerLoadError(
            None,
            "Some error in the 'PowerLoad' class.",
        )


class TestPowerLoad:
    def setup_method(self):

        time_1 = [0, 4, 7, 8]
        data_1 = [6, 9, 7, 8]
        model_1 = PowerLoadModel.RAMP
        time_2 = [2, 5, 7, 9, 10]
        data_2 = [2, 2, 2, 4, 4]
        model_2 = PowerLoadModel.STEP

        data_set_1 = PowerData("Data 1", time_1, data_1)
        data_set_2 = PowerData("Data 2", time_2, data_2)
        load_1 = PowerLoad("Load 1", data_set_1, model_1)
        load_2 = PowerLoad("Load 2", data_set_2, model_2)

        self.load_1 = load_1
        self.load_2 = load_2

    def extract_input_samples(self):
        load_1 = self.load_1
        load_2 = self.load_2
        powerdata_1 = load_1.data_set[0]
        powerdata_2 = load_2.data_set[0]
        powerloadmodel_1 = load_1.model[0]
        powerloadmodel_2 = load_2.model[0]

        powerdata_samples = [
            powerdata_1,
            powerdata_2,
        ]
        powerloadmodel_samples = [
            powerloadmodel_1,
            powerloadmodel_2,
        ]
        return powerdata_samples, powerloadmodel_samples

    def test_constructor(self):

        out1, out2 = self.extract_input_samples()
        powerdata_samples, powerloadmodel_samples = out1, out2

        multi_load = PowerLoad(
            "Load with multiple data sets & models",
            powerdata_samples,
            powerloadmodel_samples,
        )
        assert isinstance(multi_load, PowerLoad)

    def test_sanity(self):

        out1, out2 = self.extract_input_samples()
        powerdata_samples, powerloadmodel_samples = out1, out2

        max_powerdata_length = len(powerdata_samples)
        max_powerloadmodel_length = len(powerloadmodel_samples)

        for powerdata in powerdata_samples:
            base_powerdata_input = powerdata

            for powerloadmodel in powerloadmodel_samples:
                base_powerloadmodel_input = powerloadmodel

                for n_pd in range(max_powerdata_length):
                    data_input = powerdata_samples[0:n_pd]
                    data_input.insert(0, base_powerdata_input)

                    for n_plm in range(max_powerloadmodel_length):
                        model_input = powerloadmodel_samples[0:n_plm]
                        model_input.insert(0, base_powerloadmodel_input)

                        bluemira_debug(
                            f"""
                            {script_title()} (PowerLoadModel._sanity)

                            Current PowerData input:
                            {pformat(data_input)}

                            Current PowerLoadModel input:
                            {pformat(model_input)}
                            """
                        )

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

    @staticmethod
    def make_time_list_for_interpolation():
        start = -2
        stop = 12
        step = 0.1
        time_vector = np.arange(start, stop, step)
        time_list = list(time_vector)
        return time_list

    @staticmethod
    def check_interpolation(original_points, curve):
        """
        Confirm that curve is an interpolation.

        Current simplified approach: no curve value is out of the bounds
        of the original defining interpolation points, except if it is a
        zero ('fill_value' argument of 'interp1d').
        """
        original_max = max(original_points)
        original_min = min(original_points)
        curve_max = max(curve)
        curve_min = min(curve)
        assert (curve_max <= original_max) or (curve_max == 0)
        assert (curve_min >= original_min) or (curve_min == 0)

    def test_single_curve(self):

        out1, out2 = self.extract_input_samples()
        powerdata_samples, powerloadmodel_samples = out1, out2

        test_time = self.make_time_list_for_interpolation()
        time_length = len(test_time)

        for powerdata in powerdata_samples:
            power_points = powerdata.data

            for model in powerloadmodel_samples:
                curve = PowerLoad._single_curve(
                    powerdata,
                    model,
                    test_time,
                )
                assert len(curve) == time_length
                self.check_interpolation(power_points, curve)

    def test_validate_time(self):
        load = self.load_1
        test_arguments = [
            None,
            1.2,
            -1.2,
            70,
            -70,
            "some string",
            [1, 2, 3, 4],
            (1, 2, 3, 4),
            load,
        ]

        for argument in test_arguments:
            bluemira_debug(
                f"""
                {script_title()} (PowerLoadModel._validate_time)

                Argument currently being tested:
                {pformat(argument)}
                """
            )
            if isinstance(argument, (int, float, list)):
                time = PowerLoad._validate_time(argument)
                assert isinstance(time, list)
            else:
                with pytest.raises(PowerLoadError):
                    time = PowerLoad._validate_time(argument)

    def test_curve(self):

        load_samples = [
            self.load_1,
            self.load_2,
        ]
        test_time = self.make_time_list_for_interpolation()
        time_length = len(test_time)

        for sample in load_samples:
            curve = sample.curve(test_time)
            curve_length = len(curve)
            assert curve_length == time_length

            # How to test interpolation of a super-imposed data set?

    # ------------------------------------------------------------------
    # VISUALIZATION
    # ------------------------------------------------------------------

    def test_refine_vector(self):

        powerdata_samples, _ = self.extract_input_samples()

        max_number_of_refinements = 10
        all_refinement_orders = range(max_number_of_refinements)
        for sample in powerdata_samples:
            time = sample.time

            for refinement_order in all_refinement_orders:
                refined_time = PowerLoad._refine_vector(
                    time,
                    refinement_order,
                )

                set_from_time = set(time)
                set_from_refined_time = set(refined_time)
                check = set_from_time.issubset(set_from_refined_time)
                time_is_subset_of_refined_time = check
                assert time_is_subset_of_refined_time

    @staticmethod
    def prepare_figure():
        fig = plt.figure()
        plt.grid()

    def run_plot(self, detailed_plot_flag):
        self.prepare_figure()
        all_axes = validate_axes()
        list_of_plot_objects = []

        load_samples_and_associated_colors = {
            "r": self.load_1,
            "b": self.load_2,
        }
        for sample_color in load_samples_and_associated_colors:
            sample = load_samples_and_associated_colors[sample_color]
            current_list_of_plot_objects = sample.plot(
                ax=all_axes, detailed=detailed_plot_flag, c=sample_color
            )
            list_of_plot_objects.append(current_list_of_plot_objects)
            bluemira_debug(
                f"""
                {script_title()} (PowerData._plot)

                All plotted objects:
                {pformat(list_of_plot_objects)}
                """
            )

        adjust_2d_graph_ranges()
        plt.show()  # Run with `pytest --plotting-on` to visualize

    def test_simple_plot(self):
        detailed_flag = False
        self.run_plot(detailed_flag)

    def test_detailed_plot(self):
        detailed_flag = True
        self.run_plot(detailed_flag)

    # ------------------------------------------------------------------
    # ARITHMETICS
    # ------------------------------------------------------------------
    def test_add(self):
        self.prepare_figure()

        load_1 = self.load_1
        load_2 = self.load_2
        powerdata_1 = load_1.data_set
        powerdata_2 = load_2.data_set
        powerloadmodel_1 = load_1.model
        powerloadmodel_2 = load_2.model

        result = load_1 + load_2
        powerdata_r = result.data_set
        powerloadmodel_r = result.model

        bluemira_debug(
            f"""
            {script_title()} (PowerLoadModel.__add__)

            Load data in data sets:
            {pformat([p.data for p in powerdata_1])}
            {pformat([p.data for p in powerdata_2])}
            {pformat([p.data for p in powerdata_r])}

            Load models:
            {pformat(powerloadmodel_1)}
            {pformat(powerloadmodel_2)}
            {pformat(powerloadmodel_r)}
            """
        )
        assert isinstance(result, PowerLoad)
        assert powerdata_r == powerdata_1 + powerdata_2
        assert powerloadmodel_r == powerloadmodel_1 + powerloadmodel_2
        list_of_plot_objects = result.plot(detailed=True, c="b")
        adjust_2d_graph_ranges()
        plt.show()  # Run with `pytest --plotting-on` to visualize
