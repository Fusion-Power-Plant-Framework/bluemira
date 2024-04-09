# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np

from bluemira.equilibria.coils import Coil, CoilSet, SymmetricCircuit
from bluemira.equilibria.optimisation.problem import CoilsetPositionCOP
from bluemira.geometry.tools import make_polygon
from bluemira.utilities.positioning import PositionMapper, RegionInterpolator


class TestCoilsetOptimiser:
    @classmethod
    def setup_class(cls):
        circuit = SymmetricCircuit(
            Coil(
                x=1.5,
                z=6.0,
                current=1e6,
                dx=0.25,
                dz=0.5,
                j_max=1e-5,
                b_max=100,
                ctype="PF",
                name="PF_2",
            ),
            Coil(
                x=1.5,
                z=-6.0,
                current=1e6,
                dx=0.25,
                dz=0.5,
                j_max=1e-5,
                b_max=100,
                ctype="PF",
                name="PF_4",
            ),
        )

        coil2 = Coil(
            x=4.0,
            z=10.0,
            current=2e6,
            dx=0.5,
            dz=0.33,
            j_max=5.0e-6,
            b_max=50.0,
            name="PF_1",
        )

        coil3 = Coil(
            x=4.0,
            z=20.0,
            current=7e6,
            dx=0.5,
            dz=0.33,
            j_max=np.nan,
            b_max=50.0,
            name="PF_3",
        )
        cls.coilset = CoilSet(circuit, coil2, coil3)

        max_coil_shifts = {
            "x_shifts_lower": -2.0,
            "x_shifts_upper": 1.0,
            "z_shifts_lower": -1.0,
            "z_shifts_upper": 5.0,
        }

        cls.pfregions = {}

        xup = cls.coilset.x[cls.coilset._control_ind] + max_coil_shifts["x_shifts_upper"]
        xlo = cls.coilset.x[cls.coilset._control_ind] + max_coil_shifts["x_shifts_lower"]
        zup = cls.coilset.z[cls.coilset._control_ind] + max_coil_shifts["z_shifts_upper"]
        zlo = cls.coilset.z[cls.coilset._control_ind] + max_coil_shifts["z_shifts_lower"]

        for name, xl, xu, zl, zu in zip(
            cls.coilset.name, xup, xlo, zup, zlo, strict=False
        ):
            cls.pfregions[name] = RegionInterpolator(
                make_polygon({"x": [xl, xu, xu, xl, xl], "z": [zl, zl, zu, zu, zl]})
            )

        cls.optimiser = CoilsetPositionCOP(
            cls.coilset, None, None, PositionMapper(cls.pfregions)
        )

    def test_modify_coilset(self):
        # Read
        coilset_opt_state = self.optimiser.coilset.get_optimisation_state(
            current_scale=self.optimiser.scale
        )
        # Modify vectors
        x, z, currents = (
            coilset_opt_state.xs,
            coilset_opt_state.zs,
            coilset_opt_state.currents,
        )
        x += 1.1
        z += 0.6
        currents += 0.99
        # Update
        self.optimiser.coilset.set_optimisation_state(
            currents,
            coil_position_map={
                "PF_2": [x[0], z[0]],
                "PF_1": [x[1], z[1]],
                "PF_3": [x[2], z[2]],
            },
            current_scale=self.optimiser.scale,
        )
        post_coilset_opt_state = self.optimiser.coilset.get_optimisation_state()
        assert np.allclose(post_coilset_opt_state.xs, x)
        assert np.allclose(post_coilset_opt_state.zs, z)
        assert np.allclose(
            post_coilset_opt_state.currents, currents * self.optimiser.scale
        )

    def test_current_bounds(self):
        n_control_currents = len(self.coilset.current[self.coilset._control_ind])
        user_max_current = 2.0e9
        user_current_limits = (
            user_max_current * np.ones(n_control_currents) / self.optimiser.scale
        )
        coilset_current_limits = self.optimiser.coilset.get_max_current()

        control_current_limits = np.minimum(user_current_limits, coilset_current_limits)
        bounds = (-control_current_limits, control_current_limits)

        assert n_control_currents == len(user_current_limits)
        assert n_control_currents == len(coilset_current_limits)

        optimiser_current_bounds = self.optimiser.get_current_bounds(
            self.optimiser.coilset, user_max_current, self.optimiser.scale
        )
        assert np.allclose(
            bounds[0],
            self.coilset._opt_currents_expand_mat @ optimiser_current_bounds[0],
        )
        assert np.allclose(
            bounds[1],
            self.coilset._opt_currents_expand_mat @ optimiser_current_bounds[1],
        )

        # print(self.optimiser.coilset.get_max_currents(0.0))
        # print(self.optimiser.get_current_bounds(10.0) / self.optimiser.scale)

        # self.optimiser.get_current_bounds()

        # optimiser_maxima = 0.9
        # i_max = self.coilset.get_max_currents(max_currents)
