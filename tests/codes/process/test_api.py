# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from unittest.mock import patch

import numpy as np
import pytest

from bluemira.codes.process import api

PROCESS_OBS_VAR = {
    "ni": "ni wang",
    "ni wang": "ni peng",
    "garden": "shrubbery",
}


@patch("bluemira.codes.process.api.OBS_VARS", PROCESS_OBS_VAR)
def test_update_obsolete_vars():
    str1 = api.update_obsolete_vars("ni")
    str2 = api.update_obsolete_vars("garden")
    assert str1 == "ni peng"
    assert str2 == "shrubbery"


@pytest.mark.skipif(not api.ENABLED, reason="PROCESS is not installed on the system.")
@patch.object(api, "imp_data")
def test_impurities(imp_data_mock):
    imp_data_mock.__file__ = "./__init__.py"
    assert api.Impurities["H"] == api.Impurities.H
    assert api.Impurities(1) == api.Impurities.H
    assert api.Impurities(1).id() == "fimp(01)"
    assert api.Impurities(10).id() == "fimp(10)"
    assert api.Impurities(1).files()["lz"].parts[-1] == "H__lz_tau.dat"
    assert api.Impurities(1).files()["z"].parts[-1] == "H__z_tau.dat"
    assert api.Impurities(1).files()["z2"].parts[-1] == "H__z2_tau.dat"
    assert api.Impurities(10).files()["lz"].parts[-1] == "Fe_lz_tau.dat"


@pytest.mark.skipif(not api.ENABLED, reason="PROCESS is not installed on the system.")
def test_impurity_datafiles():
    confinement_time_ms = 0.1
    lz, z = api.Impurities["H"].read_impurity_files(("lz", "z"))
    z2, lz2 = api.Impurities["H"].read_impurity_files(("z", "lz"))

    lz_ref = filter(lambda _lz: f"{confinement_time_ms:.1f}" in _lz.content, lz)
    lz2_ref = filter(lambda _lz: f"{confinement_time_ms:.1f}" in _lz.content, lz2)

    z_ref = filter(lambda _z: f"{confinement_time_ms:.1f}" in _z.content, z)
    z2_ref = filter(lambda _z: f"{confinement_time_ms:.1f}" in _z.content, z2)

    np.testing.assert_allclose(
        np.array(next(lz_ref).data, dtype=float),
        np.array(next(lz2_ref).data, dtype=float),
    )
    np.testing.assert_allclose(
        np.array(next(z_ref).data, dtype=float), np.array(next(z2_ref).data, dtype=float)
    )


def test_INVariable_works_with_floats():
    inv = api._INVariable("name", 5, "a 'type'", "a 'group'", "a comment")

    assert inv.name == "name"
    assert inv._value == inv.get_value == 5
    assert inv.value == "5"

    inv.value = 6

    assert inv._value == inv.get_value == 6
    assert inv.value == "6"

    assert inv.v_type == "a 'type'"
    assert inv.parameter_group == "a 'group'"
    assert inv.comment == "a comment"


def test_INVariable_works_with_list_and_dict():
    inv = api._INVariable("name", [5], "a 'type'", "a 'group'", "a comment")

    assert inv.name == "name"
    assert inv._value == inv.value == inv.get_value == [5]

    inv.value = {"a": 6}

    assert inv._value == inv.value == inv.get_value == {"a": 6}

    assert inv.v_type == "a 'type'"
    assert inv.parameter_group == "a 'group'"
    assert inv.comment == "a comment"
