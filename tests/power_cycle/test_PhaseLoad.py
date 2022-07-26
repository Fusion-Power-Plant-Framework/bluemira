# Import
from pprint import pprint

import matplotlib.pyplot as plt
from _TIAGO_FILES_.Tools import Tools as imported_tools

import bluemira.base.constants as constants
from bluemira.power_cycle.base import PowerCycleUtilities as imported_utilities
from bluemira.power_cycle.loads import PhaseLoad as imported_class_3
from bluemira.power_cycle.loads import PowerData as imported_class_1
from bluemira.power_cycle.loads import PowerLoad as imported_class_2
from bluemira.power_cycle.timeline import PowerCyclePhase as imported_class_0

# Header
imported_tools.print_header("Test PhaseLoad")

# Phase
ftt = imported_class_0(
    "Dwell",
    "dwl",
    "ss",
    constants.raw_uc(10, "minute", "second"),
)

# PowerLoad 1
data_11 = imported_class_1(
    "Load 1 - Fixed Consumption",
    [0, 1],
    [2, 2],
)
data_12 = imported_class_1(
    "Load 1 - Variable Consumption",
    [0, 4, 7, 8],
    [6, 9, 7, 8],
)
instance_1 = imported_class_2(
    "Load 1",
    [data_11, data_12],
    ["ramp", "ramp"],
)

# PowerLoad 2
data_21 = imported_class_1(
    "Load 2 - Fixed Consumption",
    [0, 200],
    [4, 4],
)
data_22 = imported_class_1(
    "Load 2 - Variable Consumption",
    [200, 500, 700, 900, 1000],
    [0, 2, 3, 5, 8],
)
instance_2 = imported_class_2(
    "Load 2",
    [data_21, data_22],
    ["step", "ramp"],
)

# Create instance of PhaseLoad
test_name = "Phase Load during Dwell"
test_set = [instance_1, instance_2]
test_normalize = [
    True,  # Normalize PowerLoad 1
    False,  # Don't normalize PowerLoad 2, and cut at phase end
]
test_instance = imported_class_3(test_name, ftt, test_set, test_normalize)
pprint(vars(test_instance))

# Test validation method
check_instance = imported_class_3._validate(test_instance)
"check_instance = imported_class_3._validate(test_name)"
pprint("No errors raised on validation!")

# Test `_normal_set` attribute
load_data = test_instance.display_data(option="load")
pprint(load_data)
normal_data = test_instance.display_data(option="normal")
pprint(normal_data)

# Test visualization method
plt.figure()
plt.grid()
test_instance.plot(n_points=1000, detailed=True, c="r")
imported_utilities.adjust_2d_graph_ranges()

# Show plots
plt.show()
