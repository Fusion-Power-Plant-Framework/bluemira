"""
An example of how to create a DisplayConverter
"""
# %%
import os

import bluemira.power_cycle.display as dsp

# %%[markdown]
# # Create DisplayConverter instance

# %%
# Start script
os.system("reset")
print("\n")

# Create an instance of the DisplayConverter class
converter = dsp.DisplayConverter()
# print(vars(converter))
print(converter.default_display_units)
print(converter.desired_display_units)
