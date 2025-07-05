# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Initialization functions to register magnet classes.

These functions import necessary modules to trigger class registration
(via metaclasses) without polluting the importing namespace.
"""


def register_strands():
    """
    Import and register all known Strand classes.

    This triggers their metaclass registration into the STRAND_REGISTRY.
    Importing here avoids polluting the top-level namespace.

    Classes registered
    -------------------
    - Strand
    - SuperconductingStrand
    """


def register_cables():
    """
    Import and register all known Cable classes.

    This triggers their metaclass registration into the CABLE_REGISTRY.
    Importing here avoids polluting the top-level namespace.

    Classes registered
    -------------------
    - Cable
    - Specialized cable types (e.g., TwistedCables)
    """


def register_conductors():
    """
    Import and register all known Conductor classes.

    This triggers their metaclass registration into the CONDUCTOR_REGISTRY.
    Importing here avoids polluting the top-level namespace.

    Classes registered
    -------------------
    - Conductor
    - Specialized conductors
    """


def register_all_magnets():
    """
    Import and register all known magnet-related classes.

    Calls `register_strands()`, `register_cables()`, and `register_conductors()`
    to fully populate all internal registries.

    Use this function at initialization if you want all classes to be available.
    """
    register_strands()
    register_cables()
    register_conductors()
