from pprint import pformat

import matplotlib.pyplot as plt
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


def script_title():
    return "Test Power Cycle 'net'"


def test_NetPowerABCError():
    with pytest.raises(NetPowerABCError):
        raise NetPowerABCError(
            None,
            "Some error in the 'NetPowerABC' class.",
        )


class TestNetPowerABC:
    class SampleConcreteClass(NetPowerABC):  # Inner Class
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
        plt.show()  # Run with `pytest --plotting-on` to visualize


class TestPowerLoadModel:
    def test_members(self):

        all_values = [member.value for member in PowerLoadModel]
        all_names = [member.name for member in PowerLoadModel]
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

    def test_add(self):
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
