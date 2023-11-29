# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Test that the version can be retrieved as expected
"""

import bluemira


def test_version():
    """
    Test that we can get the version from bluemira.
    """
    assert bluemira.__version__
