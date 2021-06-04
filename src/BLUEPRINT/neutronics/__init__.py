# flake8: noqa
from BLUEPRINT.base.lookandfeel import bpwarn
import os
import matplotlib

if os.sys.platform != "linux":
    bpwarn("Cannot run radiation transport codes on Windows or Mac..")
else:
    try:
        import openmc
    except ModuleNotFoundError:
        bpwarn("OpenMC not installed.")

    try:
        rcParamsDefault = matplotlib.rcParamsDefault.copy()
        import openmoc

        # OpenMOC overwrites rcParamsDefault via their plotter.py... bad news :(
        matplotlib.rcParamsDefault = rcParamsDefault
    except ModuleNotFoundError:
        bpwarn("OpenMOC not installed.")
