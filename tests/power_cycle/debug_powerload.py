# Import general packages
# import numpy as np
from pprint import pprint

import matplotlib.pyplot as plt

# Import Power Cycle packages
from bluemira.base.look_and_feel import bluemira_print
from bluemira.power_cycle.net import PowerData, PowerLoad, PowerLoadModel
from bluemira.power_cycle.tools import adjust_2d_graph_ranges

# from scipy.interpolate import interp1d as imported_interp1d


# Header
bluemira_print("Debug PowerLoad")

# Test data
name_1 = "test_1"
# time_1 = [0, 4, 7, 8]
# data_1 = [6, 9, 7, 8]
time_1 = [0, 4, 7]
data_1 = [6, 9, 7]
load_1 = PowerData(name_1, time_1, data_1)
model_1 = PowerLoadModel.RAMP
name_2 = "test_2"
time_2 = [2, 5, 7, 9, 10]
data_2 = [2, 2, 2, 4, 4]
load_2 = PowerData(name_2, time_2, data_2)
model_2 = PowerLoadModel.STEP

# Create instances of PowerLoad
instance_1 = PowerLoad("Test 1", load_1, model_1)
instance_2 = PowerLoad("Test 2", load_2, model_2)

# Test `_refine_vector` method
refined_time_1 = PowerLoad._refine_vector(time_1, 3)
print(refined_time_1)
refined_time_2 = PowerLoad._refine_vector(time_2, 0)
print(refined_time_2)

# Test visualization method
plt.figure()
plt.grid()
plot_list_1 = instance_1.plot(c="r")
plot_list_2 = instance_2.plot(c="b")
adjust_2d_graph_ranges()
pprint(plot_list_1)
pprint(plot_list_2)

"""
# Test addition method
plt.figure()
plt.grid()
instance_3 = instance_1 + instance_2
plot_list = instance_3.plot(detailed=True)
pprint(plot_list)
"""

# Show plots
plt.show()
