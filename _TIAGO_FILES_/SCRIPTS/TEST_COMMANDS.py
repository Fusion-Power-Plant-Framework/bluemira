# Import
import json
import sys

import matplotlib.pyplot as plt

# import numpy as np
from scipy.interpolate import interp1d as INTERP1D

# Idealized validation functionality
print("\n")

# Create test vector
test_vector = "[0, 2, 4, 6, 8]"
print(test_vector)

# Remove white spaces
test_vector = test_vector.replace(" ", "")
print(test_vector)

# Remove starting and ending list indicator
test_vector = test_vector.replace("[", "")
test_vector = test_vector.replace("]", "")
print(test_vector)

# Split string into elements
test_vector = test_vector.split(",")
print(test_vector)

# Convert elements into floats
test_vector = [float(i) for i in test_vector]
print(test_vector)

# Test tuples
print("\n")
test_tuple = (test_vector, test_vector)
x1, y1 = test_tuple
print(x1)
print(y1)
print(test_tuple)

# Test foor loop
print("\n")
n_segments = 10
test_list = []
for s in range(n_segments):
    test_list.append(s)

print(test_list)

# Test joining lists
print("\n")
test_join = test_vector + test_vector
print(test_join)

# Test opening JSON
file_path = "data/power_cycle/D17-iHCPB-con.json"
file = open(file_path)
data = json.load(file)
file.close()
print(json.dumps(data, indent=4))

# Test reading JSON
data_length = len(data)
print(data_length)

# Test accessing all subkeys in a value
category = "Heating & Current Drive"
all_systems = data[category]
print(list(all_systems.keys()))

# Test interp1d
eps = sys.float_info.epsilon
this_load = [6, 9, 7, 8, 0, 0]
this_time = [0, 4, 7, 8, 8 + eps, 10]
other_load = [0, 0, 2, 2, 2, 4, 4]
other_time = [0, 2 - eps, 2, 5, 7, 9, 10]

this_lookup = INTERP1D(this_time, this_load)
other_lookup = INTERP1D(other_time, other_load)

fig = plt.figure()
m_size = 100
plt.grid()
plt.scatter(this_time, this_load, color="b", s=m_size)
plt.scatter(other_time, other_load, color="r", s=m_size)

# this_load = other_lookup(this_time)
# other_load = this_lookup(other_time)
# plt.scatter(this_time, this_load, color="r", s=m_size)
# plt.scatter(other_time, other_load, color="r", s=m_size)

plt.show()
