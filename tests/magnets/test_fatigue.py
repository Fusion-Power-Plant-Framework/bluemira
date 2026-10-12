# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from bluemira.magnets.fatigue import (
    EllipticalEmbeddedCrack,
    QuarterEllipticalCornerCrack,
    SemiEllipticalSurfaceCrack,
)


class TestFatigueCracks:
    @pytest.mark.parametrize(
        ("crack_cls", "area", "aspect_ratio"),
        [
            (SemiEllipticalSurfaceCrack, 1, 3),
            (SemiEllipticalSurfaceCrack, 2e-6, 2),
            (QuarterEllipticalCornerCrack, 1, 3),
            (QuarterEllipticalCornerCrack, 2e-6, 2),
            (EllipticalEmbeddedCrack, 1, 3),
            (EllipticalEmbeddedCrack, 2e-6, 2),
        ],
    )
    def test_init_by_area(self, crack_cls, area, aspect_ratio):
        crack = crack_cls.from_area(area, aspect_ratio)
        assert np.isclose(crack.area, area, atol=1e-9, rtol=0.0)
