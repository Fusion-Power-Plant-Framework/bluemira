# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Initialise the bluemira package.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("bluemira")
except PackageNotFoundError:
    from setuptools_scm import get_version

    __version__ = get_version()
