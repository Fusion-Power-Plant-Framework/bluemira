# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from copy import deepcopy
from dataclasses import fields
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from bluemira.base.constants import EPS
from bluemira.equilibria.coils import Coil, CoilSet
from bluemira.equilibria.run import (
    BreakdownCOPConfig,
    EQConfig,
    OptimisedPulsedCoilsetDesign,
    PulsedCoilsetDesign,
    PulsedCoilsetDesignFrame,
    Snapshot,
)


class TestSnapshot:
    class Emptyclass:  # noqa: D106
        pass

    def test_Snapshot_copy(self):
        classes = [self.Emptyclass() for _ in range(7)]
        snapshot = Snapshot(*classes)

        for cl, sn in zip(classes, fields(type(snapshot)), strict=False):
            if sn.name not in {"constraints", "tfcoil"}:
                assert cl != getattr(snapshot, sn.name)
            else:
                assert cl == getattr(snapshot, sn.name)


class TestPulsedCoilSetDesign:
    def setup_method(self):
        self.params = PulsedCoilsetDesignFrame.from_dict({
            "A": {"value": 1, "unit": ""},
            "B_premag_stray_max": {"value": 2, "unit": ""},
            "C_Ejima": {"value": 3, "unit": ""},
            "I_p": {"value": 4, "unit": ""},
            "l_i": {"value": 5, "unit": ""},
            "R_0": {"value": 6, "unit": ""},
            "tau_flattop": {"value": 7, "unit": ""},
            "tk_sol_ib": {"value": 8, "unit": ""},
            "v_burn": {"value": 9, "unit": ""},
        })

    class MyPulsedCoilset(PulsedCoilsetDesign):  # noqa: D106
        def optimise(self):
            return self.coilset

    def test_breakdown_settings(self):
        mypcs = self.MyPulsedCoilset(self.params, *[None] * 4)
        assert isinstance(mypcs.bd_config, BreakdownCOPConfig)
        mypcs.bd_config.n_B_stray_points = 9
        assert mypcs.bd_config.n_B_stray_points == 9

        mypcs = self.MyPulsedCoilset(
            self.params, *[None] * 4, breakdown_settings={"n_B_stray_points": 10}
        )
        assert isinstance(mypcs.bd_config, BreakdownCOPConfig)
        assert mypcs.bd_config.n_B_stray_points == 10

        with pytest.raises(TypeError):
            mypcs = self.MyPulsedCoilset(
                self.params, *[None] * 4, breakdown_settings={"gamma": 1e-5}
            )

    def test_equilibrium_settings(self):
        mypcs = self.MyPulsedCoilset(self.params, *[None] * 6)
        assert isinstance(mypcs.eq_config, EQConfig)
        mypcs.eq_config.gamma = 9
        assert mypcs.eq_config.gamma == 9

        mypcs = self.MyPulsedCoilset(
            self.params, *[None] * 4, equilibrium_settings={"gamma": 1e-5}
        )
        assert isinstance(mypcs._eq_config, EQConfig)
        assert mypcs.eq_config.gamma == pytest.approx(1e-5, rel=0, abs=EPS)

        with pytest.raises(TypeError):
            mypcs = self.MyPulsedCoilset(
                self.params, *[None] * 4, equilibrium_settings={"n_B_stray_points": 10}
            )

    @mock.patch("bluemira.equilibria.run.Snapshot", return_value="SNAP")
    def test_take_snapshot(self, snapshot, caplog):
        mypcs = self.MyPulsedCoilset(self.params, *[None] * 4)
        mypcs.take_snapshot("test", "eq", "coilset", "problem", "profiles")
        snapshot.assert_called_with(
            "eq", "coilset", "problem", "profiles", iterator=None, limiter=None
        )

        assert mypcs.snapshots["test"] == "SNAP"
        # Overwrite snapshot
        mypcs.take_snapshot("test", "eq", "coilset", "problem", "profiles")
        assert len(caplog.messages) == 1

    # def test_run_premagnetisation(self):
    # def test_run_reference_equilibrium(self):

    @mock.patch("bluemira.equilibria.run.calc_psib", return_value=0)
    def test_calculate_sof_eof_fluxes(self, calc_psib):
        mypcs = self.MyPulsedCoilset(self.params, *[None] * 4)

        def r_premag(self, val=1):
            self.snapshots[self.BREAKDOWN] = mock.MagicMock()
            self.snapshots[self.BREAKDOWN].eq.breakdown_psi = val

        with mock.patch.object(
            self.MyPulsedCoilset, "run_premagnetisation", new=r_premag
        ):
            # all the same because calc_psib is mocked
            out_val = mypcs.calculate_sof_eof_fluxes()
            np.testing.assert_allclose(out_val, (0, -63))

            # change psi_premag, call with no args
            r_premag(mypcs, 2)
            mypcs.calculate_sof_eof_fluxes()
            np.testing.assert_allclose(out_val, (0, -63))

            # change psi_premag, call with args
            mypcs.calculate_sof_eof_fluxes(3)
            np.testing.assert_allclose(out_val, (0, -63))

        # calc_psib called with different values
        psi_premag2pi = np.array([call[0][0] for call in calc_psib.call_args_list])
        np.testing.assert_allclose(psi_premag2pi[0] * 2, psi_premag2pi[1])
        np.testing.assert_allclose(psi_premag2pi[0] * 3, psi_premag2pi[2])

    # def test_get_sof_eof_opt_problems(self):

    # def test_converge_and_snapshot(self):


class TestSizePFCoilsToCarriedCurrents:
    """
    PF coils are sized 10 MA, the CS 16 MA. Each PF coil is resized to 1.1 times the
    largest current it carries at breakdown, start and end of flat-top.
    """

    MA = 1e6
    CS_SIZE = 16 * MA

    def setup_method(self):
        coilset = CoilSet(
            Coil(6, 8, dx=0.5, dz=0.5, ctype="PF", j_max=1e7, name="PF_1"),
            Coil(14, 2, dx=0.5, dz=0.5, ctype="PF", j_max=1e7, name="PF_2"),
            Coil(2.5, 0, dx=0.4, dz=1.0, ctype="CS", j_max=1e7, name="CS_1"),
        )
        self.design = OptimisedPulsedCoilsetDesign.__new__(OptimisedPulsedCoilsetDesign)
        self.design.coilset = coilset
        self.design.snapshots = {}
        self.problems = []
        for _ in range(2):
            problem = mock.MagicMock()
            problem.eq.coilset = deepcopy(coilset)
            self.problems.append(problem)

    def _snapshot(self, pf_currents):
        coilset = deepcopy(self.design.coilset)
        coilset.get_coiltype("PF").get_control_coils().current = pf_currents
        return SimpleNamespace(coilset=coilset)

    def _carry(self, breakdown, sof, eof):
        """Make the design's runs leave snapshots with these PF currents."""

        def converge_and_snapshot(_problems):
            self.design.snapshots["SOF"] = self._snapshot(sof)
            self.design.snapshots["EOF"] = self._snapshot(eof)

        def run_premagnetisation():
            self.design.snapshots["Breakdown"] = self._snapshot(breakdown)

        self.design.converge_and_snapshot = mock.Mock(side_effect=converge_and_snapshot)
        self.design.run_premagnetisation = mock.Mock(side_effect=run_premagnetisation)
        converge_and_snapshot(self.problems)
        run_premagnetisation()
        self.design.converge_and_snapshot.reset_mock()
        self.design.run_premagnetisation.reset_mock()

    def _sizes(self, coilset):
        return coilset.get_coiltype("PF").get_control_coils().get_max_current()

    def test_PF_coils_are_sized_for_the_currents_they_carry(self):
        # PF_1 carries most at the start of flat-top, PF_2 at breakdown, where it is full
        self._carry(
            breakdown=[1 * self.MA, -10 * self.MA],
            sof=[-4 * self.MA, 2 * self.MA],
            eof=[2 * self.MA, 3 * self.MA],
        )

        self.design._size_pf_coils_to_carried_currents(self.problems)

        expected = [4.4 * self.MA, 10 * self.MA]
        for coilset in (self.design.coilset, *(p.eq.coilset for p in self.problems)):
            np.testing.assert_allclose(self._sizes(coilset), expected)
            assert coilset.get_coiltype("CS").get_max_current(0.0) == self.CS_SIZE
        for problem in self.problems:
            np.testing.assert_allclose(
                problem.set_current_bounds.call_args.args[0], [*expected, self.CS_SIZE]
            )
            problem.eq._remap_greens.assert_called_once()
            assert problem.targets.call_args.kwargs["fixed_coils"] is False
            assert (
                problem.update_magnetic_constraints.call_args.kwargs["fixed_coils"]
                is False
            )
        # Converged once at the new sizes, then found settled
        self.design.converge_and_snapshot.assert_called_once_with(self.problems)
        self.design.run_premagnetisation.assert_called_once_with()

    def test_PF_coils_are_not_made_larger(self):
        self._carry(
            breakdown=[10 * self.MA, 0],
            sof=[0, 9.5 * self.MA],
            eof=[0, 0],
        )

        self.design._size_pf_coils_to_carried_currents(self.problems)

        np.testing.assert_allclose(self._sizes(self.design.coilset), [10 * self.MA] * 2)
        self.design.converge_and_snapshot.assert_not_called()
        for problem in self.problems:
            problem.set_current_bounds.assert_not_called()

    def test_warns_if_sizes_do_not_settle(self, caplog):
        def converge_and_snapshot(_problems):
            # Each convergence needs half of what the coils are sized for
            half = 0.5 * self._sizes(self.design.coilset)
            self.design.snapshots["SOF"] = self._snapshot(half)
            self.design.snapshots["EOF"] = self._snapshot(half)

        self._carry(breakdown=[0, 0], sof=[5 * self.MA] * 2, eof=[0, 0])
        self.design.converge_and_snapshot.side_effect = converge_and_snapshot

        self.design._size_pf_coils_to_carried_currents(self.problems)

        n = OptimisedPulsedCoilsetDesign.PF_RESIZE_ITER_MAX
        assert self.design.converge_and_snapshot.call_count == n
        assert "had not settled" in caplog.text
