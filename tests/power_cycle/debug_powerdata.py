from pprint import pprint

import matplotlib.pyplot as plt

from bluemira.base.look_and_feel import bluemira_print
from bluemira.power_cycle.base import PowerCycleABCError
from bluemira.power_cycle.NET_loads import PowerData
from bluemira.power_cycle.tools import adjust_2d_graph_ranges

# Header
bluemira_print("Debug PowerData")

# Test data
test_name = "test_powerdata"
test_time = [0, 4, 7, 8]
test_data = [6, 9, 7, 8]

# Create instance of PowerData
test_instance = PowerData(test_name, test_time, test_data)

# Print instance attributes
pprint(test_instance.time)
pprint(test_instance.data)

# Test validation method
try:
    test_instance = PowerData.validate_class(test_instance)
    print("No errors raised on validation!")
except (PowerCycleABCError):
    print("PowerDataError wrongly raised on validation!")

try:
    PowerData.validate_class(test_data)
except (PowerCycleABCError):
    print("PowerDataError correctly raised on validation!")


# Test visualization method
plt.figure()
plt.grid()
plot_list = test_instance.plot(c="r")
pprint(plot_list, indent=4)
adjust_2d_graph_ranges()

# Show plots
plt.show()
