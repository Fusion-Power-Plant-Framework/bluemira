# Import
import json

import matplotlib.pyplot as plt

from bluemira.power_cycle.cycle import PowerLoadManager as imported_plm

# Test PowerLoadManager class
print("-------------------------------------------------------------\n")
print("-------------------------------------------------------------\n")

# Create PowerLoadManager instance
file_path = "data/power_cycle/D17-iHCPB-con.json"
phase_durations = {"d2f": 20, "ftt": 100, "f2d": 20, "dwl": 30}
test_PLM = imported_plm(file_path, phase_durations)
print(test_PLM.data)
print(test_PLM.n_category)
print(test_PLM.n_system)
print(test_PLM.phases)

# Test creating NBI load during pulse
category = "Heating & Current Drive"
system = "NBI"
curves_nbi = test_PLM._build_system_curves(category, system)
print(curves_nbi)
json_nbi = json.dumps(curves_nbi, indent=4)
print(json_nbi)

# Test plotting NBI load during pulse
fig = plt.figure()
ax = fig.add_subplot(111)

active_load = curves_nbi["active_load"]
active_time = curves_nbi["active_time"]
reactive_load = curves_nbi["reactive_load"]
reactive_time = curves_nbi["reactive_time"]
ax.scatter(active_time, active_load, label="active_load")
ax.scatter(reactive_time, reactive_load, label="reactive_load")
ax.plot(active_time, active_load, label="active_load")
ax.plot(reactive_time, reactive_load, label="reactive_load")
leg1 = ax.legend(loc="lower left")
plt.title("Example NBI Curve")
plt.xlabel("time (s)")
plt.ylabel("power (W)")
plt.grid()
plt.show()

# category_data = test_PLM.data[category]
# system_data = category_data[system]
# phase_data = system_data['active_load']['d2f']

# print(category_data)
# print(system_data)
# print(phase_data)
