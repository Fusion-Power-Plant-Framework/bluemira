# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
FreeCAD API
"""


def freecad_message_removal():
    """
    Remove annoying message about freecad libdir not being set

    Returns
    -------
    :
        The default freecad path
    """
    import importlib  # noqa: PLC0415
    import os  # noqa: PLC0415

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


freecad_default_path = freecad_message_removal()

from bluemira.codes.cadapi._freecad.config import _freecad_save_config

_freecad_save_config()
