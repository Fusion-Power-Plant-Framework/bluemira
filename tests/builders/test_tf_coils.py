# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from bluemira.base.parameter_frame import make_parameter_frame
from bluemira.builders.tf_coils import (
    EquispacedSelector,
    ExtremaSelector,
    FixedSelector,
    MaximiseSelector,
    RippleConstrainedLengthGOP,
    RippleConstrainedLengthGOPParams,
)
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.geometry.tools import make_polygon


class TestRippleConstrainedLengthGOP:
    princeton = PrincetonD(
        {
            "x1": {"value": 4, "fixed": True},
            "dz": {"value": 0, "fixed": True},
            "x2": {"value": 15, "lower_bound": 13, "upper_bound": 20},
        }
    )
    lcfs = make_polygon({"x": [6, 12, 6], "z": [-4, 0, 4]}, closed=True)
    wp_xs = make_polygon(
        {"x": [3.5, 4.5, 4.5, 3.5], "y": [-0.5, -0.5, 0.5, 0.5], "z": 0}, closed=True
    )
    params = make_parameter_frame(
        {
            "n_TF": {"value": 16, "unit": "", "source": "test"},
            "R_0": {"value": 9, "unit": "m", "source": "test"},
            "z_0": {"value": 0, "unit": "m", "source": "test"},
            "B_0": {"value": 6, "unit": "T", "source": "test"},
            "TF_ripple_limit": {"value": 0.6, "unit": "%", "source": "test"},
        },
        RippleConstrainedLengthGOPParams,
    )

    def test_default_setup(self):
        problem = RippleConstrainedLengthGOP(
            self.princeton,
            "SLSQP",
            {"max_eval": 100, "ftol_rel": 1e-6},
            {},
            self.params,
            self.wp_xs,
            self.lcfs,
            keep_out_zone=None,
            rip_con_tol=1e-3,
            n_rip_points=3,
        )
        problem.optimise()
        problem.plot()
        assert np.isclose(
            max(problem.ripple_values),
            self.params.TF_ripple_limit.value,
            rtol=0,
            atol=1e-3,
        )

    @pytest.mark.parametrize(
        "selector",
        [
            EquispacedSelector(3),
            EquispacedSelector(3, x_frac=0.5),
            EquispacedSelector(3, x_frac=1.0),
            ExtremaSelector(),
            FixedSelector(Coordinates({"x": [12, 6, 6], "z": [0, -4, 4]})),
            MaximiseSelector(),
        ],
    )
    def test_selector_setup(self, selector):
        problem = RippleConstrainedLengthGOP(
            self.princeton,
            "SLSQP",
            {"max_eval": 100, "ftol_rel": 1e-6},
            {},
            self.params,
            self.wp_xs,
            self.lcfs,
            keep_out_zone=None,
            rip_con_tol=1e-3,
            ripple_selector=selector,
        )
        problem.optimise()
        problem.plot()
        assert np.isclose(
            max(problem.ripple_values),
            self.params.TF_ripple_limit.value,
            rtol=0,
            atol=1e-3,
        )
