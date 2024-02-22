# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import pytest

from bluemira.equilibria.coils.coil_type import CoilType


def test_coil_type_enum():
    # Test valid enum members
    assert CoilType("PF") == CoilType.PF
    assert CoilType("CS") == CoilType.CS
    assert CoilType("DUM") == CoilType.DUM
    assert CoilType("NONE") == CoilType.NONE

    # Test case-insensitive lookup
    assert CoilType("pf") == CoilType.PF
    assert CoilType("cs") == CoilType.CS
    assert CoilType("dum") == CoilType.DUM
    assert CoilType("none") == CoilType.NONE

    # Test invalid enum member
    with pytest.raises(
        ValueError,
        match="INVALID is not a valid CoilType. Choose from: PF, CS, DUM, NONE",
    ):
        CoilType("INVALID")

    # Test invalid input type
    with pytest.raises(TypeError, match="Input must be a string."):
        CoilType(42)

    # Test invalid input type as string
    with pytest.raises(
        ValueError, match="42 is not a valid CoilType. Choose from: PF, CS, DUM, NONE"
    ):
        CoilType("42")
