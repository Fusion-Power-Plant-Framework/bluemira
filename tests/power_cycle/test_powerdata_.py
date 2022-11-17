# Import general packages
from pprint import pprint

import matplotlib.pyplot as plt

# Import Power Cycle packages
from bluemira.power_cycle.loads import PowerData
from bluemira.power_cycle.utilities import adjust_2d_graph_ranges, print_header

# Header
print_header("Test PowerData")

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
test_instance = PowerData._validate(test_instance)
"PowerData._validate(test_data)"
pprint("No errors raised on validation!")

# Test visualization method
plt.figure()
plt.grid()
plot_list = test_instance.plot()
pprint(plot_list)
adjust_2d_graph_ranges()

# Show plots
plt.show()
