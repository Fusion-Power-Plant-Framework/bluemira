# Import general packages
from pprint import pprint

import matplotlib.pyplot as plt

# Import BLUEMIRA packages
import bluemira.base.constants as constants

# Import Power Cycle packages
from bluemira.power_cycle.loads import PhaseLoad, PowerData, PowerLoad
from bluemira.power_cycle.timeline import PowerCyclePhase
from bluemira.power_cycle.utilities import adjust_2d_graph_ranges, print_header

# Header
print_header("Test PhaseLoad")

# Phase
ftt = PowerCyclePhase(
    "Dwell",
    "dwl",
    "ss",
    constants.raw_uc(10, "minute", "second"),
)

# PowerLoad 1
data_11 = PowerData(
    "Load 1 - Fixed Consumption",
    [0, 1],
    [2, 2],
)
data_12 = PowerData(
    "Load 1 - Variable Consumption",
    [0, 4, 7, 8],
    [6, 9, 7, 8],
)
instance_1 = PowerLoad(
    "Load 1",
    [data_11, data_12],
    ["ramp", "ramp"],
)

# PowerLoad 2
data_21 = PowerData(
    "Load 2 - Fixed Consumption",
    [0, 200],
    [4, 4],
)
data_22 = PowerData(
    "Load 2 - Variable Consumption",
    [200, 500, 700, 900, 1000],
    [0, 2, 3, 5, 8],
)
instance_2 = PowerLoad(
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
test_instance = PhaseLoad(test_name, ftt, test_set, test_normalize)
pprint(vars(test_instance))

# Test validation method
check_instance = PhaseLoad._validate(test_instance)
"check_instance = PhaseLoad._validate(test_name)"
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
adjust_2d_graph_ranges()

# Show plots
plt.show()
