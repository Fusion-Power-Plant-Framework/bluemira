# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from copy import deepcopy
from pathlib import Path

import numpy as np
from eqdsk.cocos import Sign

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.coils import (
    Coil,
    CoilSet,
    SymmetricCircuit,
)
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.constraints import (
    CoilFieldConstraints,
    CoilForceConstraints,
)
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
            ctype="PF",
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
            ctype="PF",
        )
        cls.coilset_none = CoilSet(coil2, coil3)
        cls.coilset_partial = CoilSet(circuit, coil2, coil3)
        cls.coilset_sym = CoilSet(circuit)

        max_coil_shifts = {
            "x_shifts_lower": -2.0,
            "x_shifts_upper": 1.0,
            "z_shifts_lower": -1.0,
            "z_shifts_upper": 5.0,
        }

        cls.pfregions = {}

        xup = (
            cls.coilset_partial.x[cls.coilset_partial._control_ind]
            + max_coil_shifts["x_shifts_upper"]
        )
        xlo = (
            cls.coilset_partial.x[cls.coilset_partial._control_ind]
            + max_coil_shifts["x_shifts_lower"]
        )
        zup = (
            cls.coilset_partial.z[cls.coilset_partial._control_ind]
            + max_coil_shifts["z_shifts_upper"]
        )
        zlo = (
            cls.coilset_partial.z[cls.coilset_partial._control_ind]
            + max_coil_shifts["z_shifts_lower"]
        )

        for name, xl, xu, zl, zu in zip(
            cls.coilset_partial.name, xup, xlo, zup, zlo, strict=False
        ):
            cls.pfregions[name] = RegionInterpolator(
                make_polygon({"x": [xl, xu, xu, xl, xl], "z": [zl, zl, zu, zu, zl]})
            )

        force_c_none = CoilForceConstraints(
            cls.coilset_none,
            PF_Fz_max=450e6,
            CS_Fz_sum_max=300e6,
            CS_Fz_sep_max=250e6,
            tolerance=0.000001,
        )
        field_c_none = CoilFieldConstraints(cls.coilset_none, 20, tolerance=0.000001)
        force_c_partial = CoilForceConstraints(
            cls.coilset_partial,
            PF_Fz_max=450e6,
            CS_Fz_sum_max=300e6,
            CS_Fz_sep_max=250e6,
            tolerance=0.000001,
        )
        field_c_partial = CoilFieldConstraints(
            cls.coilset_partial, 20, tolerance=0.000001
        )
        force_c_sym = CoilForceConstraints(
            cls.coilset_sym,
            PF_Fz_max=450e6,
            CS_Fz_sum_max=300e6,
            CS_Fz_sep_max=250e6,
            tolerance=0.000001,
        )
        field_c_sym = CoilFieldConstraints(cls.coilset_sym, 20, tolerance=0.000001)

        _dummy_eq_none = Equilibrium.from_eqdsk(
            Path(
                get_bluemira_path("equilibria/test_data", subfolder="tests"),
                "DN-DEMO_eqref.json",
            ).as_posix(),
            from_cocos=3,
            qpsi_sign=Sign.NEGATIVE,
        )
        _dummy_eq_partial = deepcopy(_dummy_eq_none)
        _dummy_eq_sym = deepcopy(_dummy_eq_partial)

        _dummy_eq_partial.coilset = cls.coilset_partial
        _dummy_eq_none.coilset = cls.coilset_none
        _dummy_eq_sym.coilset = cls.coilset_sym

        cls.optimiser_none = CoilsetPositionCOP(
            cls.coilset_none,
            _dummy_eq_none,
            None,
            PositionMapper(cls.pfregions),
            constraints=[force_c_none, field_c_none],
        )

        cls.optimiser_partial = CoilsetPositionCOP(
            cls.coilset_partial,
            _dummy_eq_partial,
            None,
            PositionMapper(cls.pfregions),
            constraints=[force_c_partial, field_c_partial],
        )
        cls.optimiser_sym = CoilsetPositionCOP(
            cls.coilset_sym,
            _dummy_eq_sym,
            None,
            PositionMapper(cls.pfregions),
            constraints=[force_c_sym, field_c_sym],
        )

    def test_modify_coilset(self):
        # Read
        coilset_opt_state = self.optimiser_partial.coilset.get_optimisation_state(
            current_scale=self.optimiser_partial.scale
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
        self.optimiser_partial.coilset.set_optimisation_state(
            currents,
            coil_position_map={
                "PF_2": [x[0], z[0]],
                "PF_1": [x[1], z[1]],
                "PF_3": [x[2], z[2]],
            },
            current_scale=self.optimiser_partial.scale,
        )
        post_coilset_opt_state = self.optimiser_partial.coilset.get_optimisation_state()
        assert np.allclose(post_coilset_opt_state.xs, x)
        assert np.allclose(post_coilset_opt_state.zs, z)
        assert np.allclose(
            post_coilset_opt_state.currents, currents * self.optimiser_partial.scale
        )

    def test_current_bounds(self):
        n_control_currents = len(
            self.coilset_partial.current[self.coilset_partial._control_ind]
        )
        user_max_current = 2.0e9
        user_current_limits = (
            user_max_current * np.ones(n_control_currents) / self.optimiser_partial.scale
        )
        coilset_current_limits = self.optimiser_partial.coilset.get_max_current()

        control_current_limits = np.minimum(user_current_limits, coilset_current_limits)
        bounds = (-control_current_limits, control_current_limits)

        assert n_control_currents == len(user_current_limits)
        assert n_control_currents == len(coilset_current_limits)

        optimiser_current_bounds = self.optimiser_partial.get_current_bounds(
            self.optimiser_partial.coilset,
            user_max_current,
            self.optimiser_partial.scale,
        )
        assert np.allclose(
            bounds[0],
            self.coilset_partial._opt_currents_expand_mat @ optimiser_current_bounds[0],
        )
        assert np.allclose(
            bounds[1],
            self.coilset_partial._opt_currents_expand_mat @ optimiser_current_bounds[1],
        )

    def test_coilset_symmetry_status(self):
        assert self.coilset_none._contains_circuits is False
        assert self.coilset_partial._contains_circuits is True
        assert self.coilset_sym._contains_circuits is True

    def test_numerical_constraints(self):
        self.optimiser_none.update_magnetic_constraints(I_not_dI=True, fixed_coils=False)
        self.optimiser_partial.update_magnetic_constraints(
            I_not_dI=True, fixed_coils=False
        )
        self.optimiser_sym.update_magnetic_constraints(I_not_dI=True, fixed_coils=False)

        eq_constraints_none, ineq_constraints_none = (
            self.optimiser_none._make_numerical_constraints(self.optimiser_none.coilset)
        )
        eq_constraints_part, ineq_constraints_part = (
            self.optimiser_partial._make_numerical_constraints(
                self.optimiser_partial.coilset
            )
        )
        eq_constraints_sym, ineq_constraints_sym = (
            self.optimiser_sym._make_numerical_constraints(self.optimiser_sym.coilset)
        )

        # make sure things make sense

        assert len(eq_constraints_none) == 0
        assert len(ineq_constraints_none) == 2
        dfs_c_none = [c.get("df_constraint") for c in ineq_constraints_none]
        assert all(c is not None for c in dfs_c_none)

        assert len(eq_constraints_part) == 0
        assert len(ineq_constraints_part) == 2
        dfs_c_part = [c.get("df_constraint") for c in ineq_constraints_sym]
        assert all(c is not None for c in dfs_c_part)

        assert len(eq_constraints_sym) == 0
        assert len(ineq_constraints_sym) == 2
        dfs_c_sym = [c.get("df_constraint") for c in ineq_constraints_sym]
        assert all(c is not None for c in dfs_c_sym)

        f_c_none = ineq_constraints_none[0]["f_constraint"]
        f_c_part = ineq_constraints_part[0]["f_constraint"]
        f_c_sym = ineq_constraints_sym[0]["f_constraint"]

        f_c_none_res = f_c_none(self.optimiser_none.coilset._opt_currents)
        f_c_part_res = f_c_part(self.optimiser_partial.coilset._opt_currents)
        f_c_sym_res = f_c_sym(self.optimiser_sym.coilset._opt_currents)

        df_c_none = ineq_constraints_none[0]["df_constraint"]
        df_c_part = ineq_constraints_part[0]["df_constraint"]
        df_c_sym = ineq_constraints_sym[0]["df_constraint"]

        df_c_none_res = df_c_none(self.optimiser_none.coilset._opt_currents)
        df_c_part_res = df_c_part(self.optimiser_partial.coilset._opt_currents)
        df_c_sym_res = df_c_sym(self.optimiser_sym.coilset._opt_currents)

        # none sym. should have an f_c res shape of 2 (as there are two coils)
        # and a df_c res shape of 2x2 (2 forces x 2 coils)
        assert f_c_none_res.shape == (2,)
        assert df_c_none_res.shape == (2, 2)

        # partial sym. should have an f_c res shape of 4 (as there are four coils)
        # and can't test the df_c res shape as it will be numerical approximated
        # but would have a shape of 4x3 (4 forces per coil, and there are 3 coils,
        # the two coils and the primary one from the SymmetricCircuit
        assert f_c_part_res.shape == (4,)
        assert df_c_part_res.shape == (4, 3)

        # fully sym. should have an f_c res shape of 2 (as there are two coils)
        # but df_c  a shape of 2x1 (2 forces x 1 coil (the primary one))
        assert f_c_sym_res.shape == (2,)
        assert df_c_sym_res.shape == (2, 1)
