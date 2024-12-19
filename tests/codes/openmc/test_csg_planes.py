# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from bluemira.codes.openmc.make_csg import (
    OpenMCEnvironment,
)


def test_ztorus():
    OpenMCEnvironment


def test_zplane(): ...


def test_zcone(): ...


def test_error_when_not_sharing_neighbouring_planes():
    OpenMCEnvironment
