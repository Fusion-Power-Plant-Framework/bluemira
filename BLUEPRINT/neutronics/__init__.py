# flake8: noqa
from bluemira.base.look_and_feel import bluemira_warn
import os
import matplotlib

if os.sys.platform != "linux":
    bluemira_warn("Cannot run radiation transport codes on Windows or Mac..")
else:
    try:
        import openmc
    except ModuleNotFoundError:
        bluemira_warn("OpenMC not installed.")

    try:
        rcParamsDefault = matplotlib.rcParamsDefault.copy()
        import openmoc

        # OpenMOC overwrites rcParamsDefault via their plotter.py... bad news :(
        matplotlib.rcParamsDefault = rcParamsDefault
    except ModuleNotFoundError:
        bluemira_warn("OpenMOC not installed.")
