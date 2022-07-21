"""
Created on Thu April 07 17:56:00 2022

@author: tplobo
"""

# Import packages
import json
import os

# Import bluemira classes
import bluemira.base.file as blm
import bluemira.power_cycle.display as dsp

# Start script
os.system("reset")
print("\n")

# Root path of bluemira project
project_path = blm.get_bluemira_root()
print(project_path)
print("\n")

# Path of current script
this_path = os.path.abspath(__file__)
print(this_path)
print("\n")

# Path of current directory
dir_path = os.path.dirname(this_path)
print(dir_path)
print("\n")

# Path of `display_units.json` file in current directory
json_path = os.path.join(dir_path, "display_units.json")
print(json_path)
print("\n")

# Read `display_units.json` file
with open(json_path) as json_file:
    display_units = json.load(json_file)
    print(display_units)
print("\n")

# Display JSON data
print(json.dumps(display_units, indent=4))
print("\n")

# Read data as dictionary
print(display_units["Use"])
print("\n")

print(display_units["Display Units"])
print("\n")

print(display_units["Display Units"]["Tables"])
print("\n")

print(display_units["Display Units"]["Tables"]["mass-flow"])
print("\n")

# Create an instance of the DisplayConverter class
converter = dsp.DisplayConverter()
# print(vars(converter))
print(converter.default_display_units)
print(converter.desired_display_units)
