"""
Full power cycle model object, used to visualize results.
"""
# Import
# import sys
# import json
# import os
# import numpy as np

'''
class GenericPowerLoad:
    """

    Attributes
    ----------
    load: PowerData
        Original `time` and `data` data used to define instance.

    """

    # Generate & store PowerCurve using model specification
    # self.curve = generate_curve(self)

    @staticmethod
    def __validate_input(input):
        """
        Validate an input for class instance creation to be a string,
        and remove list-indicator characters.
        """
        if isinstance(input, str):
            input = input.replace("[", "")
            input = input.replace("]", "")
            return input
        else:
            print(
                """
                The inputs used to create an instance of the
                'PowerLoad' class must all be strings.
                """
            )
            raise TypeError()

    @staticmethod
    def __validate_vector(vector):
        """
        Validate vector inputs and convert them to numeric lists.
        """
        # Split string into elements and convert them into floats
        vector = vector.replace(" ", "")
        vector = vector.split(",")
        vector = [float(i) for i in vector]
        return vector



    # ---------------------------------------------------------------- #
    # CURVE GENERATION
    # ---------------------------------------------------------------- #
    def generate_curve(self, n_points=0):
        """
        Select which load curve model to apply, and generate power load
        curve with an extra `n_points` values in each curve segment.

        Parameters
        ----------
        n_points: `int`
            Number of points interpolated in each curve segment.
            The default value is 0, which would return the same curve
            stored in the instance attribute `input`. Instead, it
            forces the method to retrieve the default number of
            interpolation points defined in the class attribute of
            same name.
        """
        input = self.input
        model = self.model

        # If necessary, retrieve default number of interpolation points
        if n_points == 0:
            n_points = self.n_points

        # Retrieve load and time to build curve
        load = input.load
        time = input.time

        # Select model to be applied
        if model in self.valid_models:
            method_name = "generate_" + model
            generate_segment = getattr(self, method_name)
        else:
            print(
                f"""'
                Unknown 'model' for {self.__class__.__name__} class.
                """
            )
            raise ValueError()

        # Preallocate outputs
        expanded_time = []
        expanded_load = []

        # Number of curve segments
        n_segments = len(self.load) - 1

        # For each curve segment (pair of points)
        for s in range(n_segments):
            first = (time[s], load[s])
            last = (time[s + 1], load[s + 1])
            time_s, load_s = generate_segment(first, last, n_points)
            expanded_time = expanded_time + time_s
            expanded_load = expanded_load + load_s

        # Store & return curve
        curve = PowerData(expanded_load, expanded_time)
        return curve

    @staticmethod
    def interpolate_load(old_PowerCurve, new_time):
        """
        Applies the `interpolate` method of the package `scipy` to
        derive a new load vector for a desired time vector, based on an
        instance of the PowerCurve class for the interpolation.

        Parameters
        ----------
        old_PowerCurve: instance of the PowerCurve class
        new_time: float

        Returns
        -------
        new_load: float
        """

        # Create general interpolating function
        x = old_PowerCurve.time
        y = old_PowerCurve.load
        k = 'linear'
        f = float("nan")
        old_lookup = imported_interp1d(x, y, kind=k, fill_value=f)

        # First and last elements in `old_PowerCurve` time
        old_first = x[0]
        old_last = x[-1]

        # First and last elements in x
        # x_first = x[0]
        # x_last = x[-1]

        # Preallocate `new_load`
        n_new = len(new_time)
        new_load = [0] * n_new

        # Look-up values for `new_time` if necessary
        index_range = range(n_new)
        for i in index_range:

            # Current time
            t = new_time[i]

            # Check if t is out-of-bounds of `old_time`
            t_below_first = t < old_first
            t_above_last = t > old_last
            t_outbound = t_below_first or t_above_last

            # Only interpolate if t is in-bounds
            if not t_outbound:
                load_value = old_lookup(t)
                new_load[i] = load_value.tolist()

        return new_load

    @staticmethod
    def generate_ramp(first, last, n_points):
        """
        Generate curve segment using 'ramp' model.

        Parameters
        ----------
        first: tuple
            First point in curve segment ([s],[W])
        last: tuple
            Last point in curve segment ([s],[W])
        n_points: int
            Number of points in curve segment
        """
        x1, y1 = first
        x2, y2 = last
        a, b = np.polyfit([x1, x2], [y1, y2], 1)
        expanded_time = np.linspace(x1, x2, n_points).tolist()
        expanded_load = [a * t + b for t in expanded_time]
        return expanded_time, expanded_load

    @staticmethod
    def generate_step(first, last, n_points):
        """
        Generate curve segment using 'ramp' model.

        Parameters
        ----------
        first: tuple
            First point in curve segment ([s],[W])
        last: tuple
            Last point in curve segment ([s],[W])
        n_points: int
            Number of points in curve segment
        """
    # ---------------------------------------------------------------- #
    # ------------------------- VISUALIZATION ------------------------ #
    # ---------------------------------------------------------------- #
    """
    def plot(self, **kwargs):
        """
        Parameters
        ----------
        n_points: `int`
            Number of points interpolated in each curve segment.
            The default value is 0, which would return the same curve
            stored in the instance attribute `input`. Instead, it
            forces the method to retrieve the default number of
            interpolation points defined in the class attribute of
            same name.
        """

        # Retrieve default parameters
        default_kwargs = self.plot_defaults
        default_parameters = default_kwargs.keys()
        # n_parameters = len(default_parameters)

        # Preallocate extended kwargs
        extended_kwargs = kwargs

        # For each default parameter
        for parameter in default_parameters:

            # Check if parameter is specified by user
            if parameter not in kwargs:

                # Default value
                value = default_kwargs[parameter]

                # Add parameter to kwargs
                extended_kwargs[parameter] = value

        # Retrieve number of points to plot per curve segment
        n_points = self.n_points

        # Retrieve curve characteristics
        time = self.time
        load = self.load

        # Number of curve points
        n_segments

        plt.scatter(this_time, this_load, color="b", s=m_size)
    """
'''


'''
# #################################################################### #
# ######################## POWER LOAD MANAGER ######################## #
# #################################################################### #


class PowerLoadManager:
    """
    Manager for building/visualizing multiple power loads.

    Takes a JSON input file with the description of different power
    loads, organized by category and system, builds their evolution
    curves by applying the relevant model (specified in each load type)
    and builds a single equivalent power load.

    Parameters
    ----------
    file: str
        Path to input file (JSON) with power load specifications
    phases: dict
        List of time durations of each pulse phase, written as a
        dictionary with the following keys:
            d2f: (dwell-2-flat duration)    [s]
            ftt: (flat-top duration)        [s]
            f2d: (flat-2-dwell duration)    [s]
            dwl: (dwell duration)           [s]
    """

    # Valid keys for phases input
    valid_phases = ["d2f", "ftt", "f2d", "dwl"]

    # Valid load types
    valid_load_types = ["active", "reactive"]

    # ---------------------------------------------------------------- #
    # -------------------------- CONSTRUCTOR ------------------------- #
    # ---------------------------------------------------------------- #
    def __init__(self, file, phases):

        # Validate file
        self.data = self.__validate_file(file)

        # Count categories and systems
        self.n_category = self.__count_level_1(self.data)
        self.n_system = self.__count_level_2(self.data)

        # Validate phases
        self.phases = self.__validate_phases(phases)

    @staticmethod
    def __validate_file(string):
        """
        Validate a string during class instance creation to ensure it
        references a valid JSON file, and read its contents as a
        dictionary.
        """
        if not isinstance(string, str):
            print(
                """
                The first input used to create an instance of the
                'PowerLoadManager' class must be a string that
                identifies a JSON file path.
                """
            )
            raise TypeError()

        # Validate file existence
        if os.path.exists(string):
            file = open(string)
            data = json.load(file)
            file.close()
            return data
        else:
            print(
                """
                No JSON file exists in the specified path.
                """
            )
            raise TypeError()

    @staticmethod
    def __count_level_1(data):
        n_keys = len(data)
        return n_keys

    @staticmethod
    def __count_level_2(data):
        n_keys = []
        for key_level_1 in data:
            n_keys_level_2 = len(data[key_level_1])
            n_keys.append(n_keys_level_2)
        return n_keys

    @staticmethod
    def __validate_phases(dictionary):
        if not isinstance(dictionary, dict):
            print(
                """
                The second input used to create an instance of the
                'PowerLoadManager' class must be a dictionary with 4
                key:value pairs that identify the duration of each
                pulse phase, in seconds.
                """
            )
            raise TypeError()
        else:
            for key in PowerLoadManager.valid_phases:
                if key not in dictionary:
                    print(
                        f"""
                        The key '{key}' is not present in the phases
                        dictionary provided during instance creation.
                        """
                    )
                    raise ValueError()
                return dictionary

    # ---------------------------------------------------------------- #
    # ------------------------- MANAGE CURVES ------------------------ #
    # ---------------------------------------------------------------- #

    """

        def superimpose(self, other):
        """
        Create a new instance of the `PowerLoad` class by superimposing
        another instance onto this.

        This method applies the `curve` method to create new `load`
        points for any data point that does not exist in both instances.

        Parameters
        ----------
        other: `PowerLoad`
            Other instace to be sumper-imposed onto this instance.

        Returns
        -------
        another: `float`
            List of power values [W]
        """

        # Validate `this` and `other`
        this = self._validate_PowerCurve(self)
        other = self._validate_PowerCurve(other)

        # Retrieve time for both `this` and `other`
        this_time = this.time
        other_time = other.time

        # Create `another_time` (sort and unique of joined times)
        another_time = this_time + other_time
        another_time = list(set(another_time))

        # Call `interpolate_load` to interpolate loads for `another`
        new_time = another_time
        new_this_load = self.interpolate_load(this, new_time)
        new_other_load = self.interpolate_load(other, new_time)

        # Sum vectors (element by element)
        new_this_load = np.array(new_this_load)
        new_other_load = np.array(new_other_load)
        another_load = new_this_load + new_other_load
        another_load = another_load.tolist()

        # Build & output new PowerCurve
        another = PowerCurve(another_load, another_time)
        return another

    def _build_phase_load(self, load_data, phase):

        # Retrieve phase data
        phase_data = load_data[phase]

        # Retrieve model class
        model = phase_data["model_class"]

        # Create model class
        if model == "None":
            # Create generic builder with trivial inputs
            builder_class = globals()["GenericPowerLoad"]
            curve_builder = builder_class("ramp", "[0,0]", "[0,1]")
        else:
            # Retrieve arguments
            arguments = phase_data["variables_map"]
            arguments = arguments.values()
            # Create specified builder
            builder_class = globals()[model]
            curve_builder = builder_class(*arguments)

        # Build and output curve
        curve = curve_builder.generate_curve()
        return curve

    def _build_system_curves(self, system_data):

        # Pre-allocate output as dictionary
        system_name = system_data["name"]
        system_curves = {"system": system_name}

        # For each load type
        for load_type in self.valid_load_types:

            # Retrieve load data
            load_key = load_type + "_load"
            time_key = load_type + "_time"
            load_data = system_data[load_key]

            # Retrieve wall-plug efficiency and fixed load
            efficiency = load_data["wallplug_efficiency"]
            fixed_load = load_data["fixed"]

            # Pre-allocate vectors of complete evolution
            pulse_load = []
            pulse_time = []

            # For each phase
            for phase in self.valid_phases:

                # Call curve builder to compute variable load
                curve = self._build_phase_load(load_data, phase)
                variable_load = curve["power"]
                variable_time = curve["times"]

                # Add fixed load
                total_load = [val + fixed_load for val in variable_load]

                # Apply wall-plug efficiency
                real_load = np.divide(total_load, efficiency).tolist()

                # Re-scale curve with phase duration
                duration = self.phases[phase]
                last_time = variable_time[-1]
                normalization = np.divide(duration, last_time)
                real_time = np.multiply(variable_time, normalization)
                real_time = real_time.tolist()

                # Append phase load & time to pulse load & time
                pulse_load = pulse_load + real_load
                if pulse_time:
                    last_time = pulse_time[-1]
                else:
                    last_time = 0
                real_time = [val + last_time for val in real_time]
                pulse_time = pulse_time + real_time

            # Store load and time
            system_curves[load_key] = pulse_load
            system_curves[time_key] = pulse_time

        # Output curves
        return system_curves

    """
    def _build_category_curve(self, category):

        # Retrieve category data
        category_data = self.data[category]

        # List systems in category
        all_systems = list(category_data.keys())

        # Pre-allocate output as dictionary
        category_curves = {"category": category, "systems": all_systems}

        # Pre-allocate vectors of complete evolution
        pulse_load = []
        pulse_time = []

        # Memory for each system
        breakdown = {}

        # For each system in category
        for system in all_systems:

            # Retrieve system data
            system_data = category_data[system]

            # Build system curves
            system_curves = self._build_system_curves(system_data)

            # Store system curves
            breakdown.append(system_curves)

            # For each load type
            # for load_type in self.valid_load_types:

            # Super-impose loads

            # Store system curves
            # category_curves[load_key] = pulse_load
            # system_curves[time_key] = pulse_time



# #################################################################### #
# ######################### POWER CYCLE MODEL ######################## #
# #################################################################### #
# class PowerCycleModel:

'''
