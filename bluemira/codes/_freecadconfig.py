# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
FreeCAD configuration
"""

import enum  # noqa: I001
import subprocess  # noqa: S404
from pathlib import Path

import freecad  # noqa: F401
import FreeCAD


class FCUnit(enum.IntEnum):
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
    unit_prefs.SetInt("UserSchema", FCUnit[unit].value)
    unit_prefs.SetInt("Decimals", no_dp)  # 100th mm

    part_step_prefs = FreeCAD.ParamGet(
        "User parameter:BaseApp/Preferences/Mod/Part/STEP"
    )
    part_step_prefs.SetString("Scheme", _StpFileScheme[stp_file_scheme].name)
    part_step_prefs.SetString("Author", author)
    part_step_prefs.SetString("Company", "Bluemira")
    # Seems to have little effect on anything but its an option to set
    part_step_prefs.SetInt("Unit", FCUnit[unit].value)

    part_step_2_prefs = FreeCAD.ParamGet(
        "User parameter:BaseApp/Preferences/Mod/Import/hSTEP"
    )
    part_step_2_prefs.SetBool("ReadShapeCompoundMode", False)  # noqa: FBT003

    part_iges_prefs = FreeCAD.ParamGet(
        "User parameter:BaseApp/Preferences/Mod/Part/IGES"
    )
    part_iges_prefs.SetBool("BrepMode", True)  # noqa: FBT003
    part_iges_prefs.SetString("Author", author)
    part_iges_prefs.SetString("Company", "Bluemira")

    part_gen_prefs = FreeCAD.ParamGet(
        "User parameter:BaseApp/Preferences/Mod/Part/General"
    )
    part_gen_prefs.SetInt("WriteSurfaceCurveMode", 1)

    import_prefs = FreeCAD.ParamGet("User parameter:BaseApp/Preferences/Mod/Import")
    import_prefs.SetInt("ImportMode", 0)
    import_prefs.SetBool("ExportLegacy", False)  # noqa: FBT003
    import_prefs.SetBool("ExpandCompound", True)  # noqa: FBT003


def _patch_pivy():
    import pivy  # noqa: PLC0415

    if pivy.__version__ == "0.6.9":
        subprocess.run(  # noqa: S602
            [
                "/usr/bin/patch",
                "-l",
                "-p2",
                "-N",
                "-d",
                f"{Path(pivy.__file__).parent}",
                "<",
                "pivy.patch",
            ],
            cwd=Path(__file__).parent,
            shell=True,
            check=False,
        )

    from pivy import coin, quarter  # noqa: PLC0415

    return coin, quarter
