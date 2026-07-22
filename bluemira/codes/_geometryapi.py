# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Geometry backend dispatcher.

Selects the active CAD backend at import time based on the environment variable
``BLUEMIRA_GEOMETRY_BACKEND``.  Valid values: ``freecad`` (default), ``cadquery``.

Usage::

    from bluemira.codes import _geometryapi as cadapi

This replaces direct imports of ``_freecad.api`` in geometry modules so that the
rest of the codebase is backend-agnostic.
"""

from __future__ import annotations

import os

_BACKEND = os.environ.get("BLUEMIRA_GEOMETRY_BACKEND", "freecad").lower()

if _BACKEND == "cadquery":
    import bluemira.codes.cadapi._cadquery as _impl
    from bluemira.codes.cadapi._cadquery import *  # noqa: F403
elif _BACKEND == "freecad":
    import bluemira.codes.cadapi._freecad.api as _impl  # type: ignore[no-redef]
    from bluemira.codes.cadapi._freecad.api import *  # noqa: F403
else:
    msg = (
        f"Unknown BLUEMIRA_GEOMETRY_BACKEND value: {_BACKEND!r}. "
        "Valid options are 'freecad' and 'cadquery'."
    )
    raise ValueError(msg)


# Delegate private-name attribute access (e.g. cadapi._wire_is_planar) to the
# backend module.  Python calls __getattr__ only when normal lookup fails.
def __getattr__(name: str):
    return getattr(_impl, name)
