# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
CadQuery backend for bluemira.

Implements the same public interface as ``_freecadapi.py`` using CadQuery's
free-function / direct Shape API (no Workplane state). ``show_cad`` delegates
to polyscope; placements go through the :class:`_placement._CQPlacement`
adapter (a drop-in for FreeCAD's ``Base.Placement``).

This package re-exports the public API from its submodules. The actual
implementation lives in:

* ``_aliases``    — type aliases (``apiVertex``, ``apiEdge``, …) and tolerances
* ``_placement``  — Vector / Placement / Plane adapters and constructors
* ``_core``       — geometry creation, transforms, booleans, I/O, …
* ``_patches``    — side-effect monkey-patching of CadQuery shape classes
"""

from __future__ import annotations

# Side-effect: patches cq.Wire / cq.Face / cq.Solid / cq.Shell at import time.
# Must come AFTER `_core` so `_cq_area_prop` is available.
from bluemira.codes._cadqueryapi import _patches
from bluemira.codes._cadqueryapi._aliases import *
from bluemira.codes._cadqueryapi._aliases import __all__ as _aliases_all
from bluemira.codes._cadqueryapi._core import *
from bluemira.codes._cadqueryapi._core import __all__ as _core_all
from bluemira.codes._cadqueryapi._placement import *
from bluemira.codes._cadqueryapi._placement import __all__ as _placement_all

# Re-export error types under the same names ``_freecadapi`` exposes them, so
# ``cadapi.FreeCADError`` keeps resolving for callers that catch backend errors
# without knowing which backend is active. The ``as X`` form is the PEP 484
# explicit-re-export pattern (recognised by pyright/mypy as public API).
from bluemira.codes.error import FreeCADError as FreeCADError
from bluemira.codes.error import InvalidCADInputsError as InvalidCADInputsError
from bluemira.geometry.error import GeometryError as GeometryError

# PLE0604: ruff can't statically verify the *spread sources are str-lists, but
# every submodule defines its own ``__all__`` as a list of literal strings.
__all__ = [  # noqa: PLE0604
    *_aliases_all,
    *_placement_all,
    *_core_all,
    "FreeCADError",
    "GeometryError",
    "InvalidCADInputsError",
]


def __getattr__(name: str):
    # Let Python handle dunder attributes normally (e.g. __path__, __spec__)
    if name.startswith("__") and name.endswith("__"):
        raise AttributeError(name)
    # Backwards-compat: when this was a single ``_cadqueryapi.py`` file every
    # module-level name (including underscore-prefixed helpers) was reachable
    # via ``getattr``. The dispatcher in ``_geometryapi`` and a handful of
    # tests rely on this. Fall through to the submodules before declaring the
    # name unimplemented.
    from bluemira.codes._cadqueryapi import _core, _placement  # noqa: PLC0415

    for _mod in (_core, _placement):
        if hasattr(_mod, name):
            return getattr(_mod, name)
    raise NotImplementedError(
        f"_cadqueryapi: '{name}' is not yet implemented in the CadQuery backend. "
        f"Add it to the appropriate submodule under bluemira/codes/_cadqueryapi/."
    )
