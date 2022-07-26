# Import
from pprint import pprint

import matplotlib.pyplot as plt
from _TIAGO_FILES_.Tools import Tools as imported_tools

from bluemira.power_cycle.base import PowerCycleUtilities as imported_utilities
from bluemira.power_cycle.loads import PowerData as imported_class

# Header
imported_tools.print_header("Test PowerData")

# Test data
test_name = "test_powerdata"
test_time = [0, 4, 7, 8]
test_data = [6, 9, 7, 8]

# Create instance of PowerData
test_instance = imported_class(test_name, test_time, test_data)

# Print instance attributes
pprint(test_instance.time)
pprint(test_instance.data)

# Test validation method
test_instance = imported_class._validate(test_instance)
"imported_class._validate(test_data)"
pprint("No errors raised on validation!")

# Test visualization method
plt.figure()
plt.grid()
plot_list = test_instance.plot()
pprint(plot_list)
imported_utilities.adjust_2d_graph_ranges()

# Show plots
plt.show()
