# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
FreeCAD configuration
"""

import enum
import importlib
import os
import sys
from pathlib import Path


class _FreeCADPathContext:
    """Context manager for FreeCAD Imports

    If FreeCAD is installed using apt we need to add some elements to sys path
    for imports to function correctly
    """

    apt_install = False

    def __init__(self):
        # TODO(je-cook) is it possible to use the flatpak
        # /app/org.freecadweb.FreeCAD/current/active/files/freecad/lib"
        base_path = Path("/usr/lib/freecad/")
        subfolders = (
            "",
            "Ext",
            "Mod",
            "Mod/Part",
            "Mod/Draft",
            "Mod/Arch",
            "Mod/OpenSCAD",
            "Mod/Import",
        )
        self.paths = [
            "/usr/lib/freecad-python3/lib",
            *(Path(base_path, sub).as_posix() for sub in subfolders),
            "/usr/lib/python3/dist-packages",
        ]

    def __enter__(self):
        try:
            freecad_message_removal()
            import freecad  # noqa: F401, PLC0415
        except (AttributeError, ImportError):
            type(self).apt_install = True
        if self.apt_install:
            for pth in self.paths:
                sys.path.append(pth)
            freecad_message_removal()

    def __exit__(self, _exc_type, _exc_value, _exc_traceback):
        if self.apt_install:
            for pth in self.paths:
                sys.path.pop(sys.path.index(pth))


def get_freecad_modules(*mod):
    imps = []
    with _FreeCADPathContext():
        for m in mod:
            if isinstance(m, tuple):
                imp = __import__(m[0], fromlist=[*m[1:]])
                imps.extend(getattr(imp, _m) for _m in m[1:])
            else:
                imps.append(__import__(m))

    return imps[0] if len(imps) == 1 else tuple(imps)


def freecad_message_removal():
    """
    Remove annoying message about freecad libdir not being set
    """
    if "PATH_TO_FREECAD_LIBDIR" in os.environ:
        return os.environ["PATH_TO_FREECAD_LIBDIR"]
    freecad_default_path = None
    with open(importlib.util.find_spec("freecad").origin) as rr:
        for line in rr:
            if '_path_to_freecad_libdir = "' in line:
                freecad_default_path = line.split('"')[1]
                break
    if freecad_default_path is not None:
        os.environ["PATH_TO_FREECAD_LIBDIR"] = freecad_default_path

    return freecad_default_path


class _Unit(enum.IntEnum):
    """Available units in FreeCAD"""

    MM = 0  # mmKS
    SI = 1  # MKS
    US = 2  # in/lb
    IMP_DEC = 3  # imperial_decimal
    BUILD_EURO = 4  # cm/m2/m3
    BUILD_US = 5  # ft-in/sqft/cft
    CNC = 6  # mm, mm/min
    IMP_CIV = 7  # ft, ft/sec
    FEM = 8  # mm/N/s


class _StpFileScheme(enum.Enum):
    """Available STEP file schemes in FreeCAD"""

    AP203 = enum.auto()
    AP214CD = enum.auto()
    AP214DIS = enum.auto()
    AP214IS = enum.auto()
    AP242DIS = enum.auto()


def _freecad_save_config(
    unit: str = "SI",
    no_dp: int = 5,
    author: str = "Bluemira",
    stp_file_scheme: str = "AP242DIS",
):
    """
    Attempts to configure FreeCAD with units file schemes and attributions

    This must be run before Part is imported for legacy exporters
    """
    unit_prefs = FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Units")
    # Seems to have little effect on anything but its an option to set
    # does effect the GUI be apparently not the base unit of the built part...
    unit_prefs.SetInt("UserSchema", _Unit[unit].value)
    unit_prefs.SetInt("Decimals", no_dp)  # 100th mm

    part_step_prefs = FreeCAD.ParamGet(
        "User parameter:BaseApp/Preferences/Mod/Part/STEP"
    )
    part_step_prefs.SetString("Scheme", _StpFileScheme[stp_file_scheme].name)
    part_step_prefs.SetString("Author", author)
    part_step_prefs.SetString("Company", "Bluemira")
    # Seems to have little effect on anything but its an option to set
    part_step_prefs.SetInt("Unit", _Unit[unit].value)

    part_gen_prefs = FreeCAD.ParamGet(
        "User parameter:BaseApp/Preferences/Mod/Part/General"
    )
    part_gen_prefs.SetInt("WriteSurfaceCurveMode", 1)

    import_prefs = FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Mod/Import")
    import_prefs.SetInt("ImportMode", 0)
    import_prefs.SetBool("ExportLegacy", False)


freecad, FreeCAD = get_freecad_modules("freecad", "FreeCAD")

_freecad_save_config()
