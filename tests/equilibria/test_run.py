# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

from dataclasses import fields
from unittest import mock

import pytest

from bluemira.equilibria.run import (
    BreakdownCOPSettings,
    EQSettings,
    PulsedCoilsetDesign,
    Snapshot,
)


class TestSnapshot:
    class Emptyclass:  # noqa: D106
        pass

    def test_Snapshot_copy(self):
        classes = [self.Emptyclass() for _ in range(7)]
        snapshot = Snapshot(*classes)

        for cl, sn in zip(classes, fields(type(snapshot))):
            if sn.name not in ("constraints", "tfcoil"):
                assert cl != getattr(snapshot, sn.name)
            else:
                assert cl == getattr(snapshot, sn.name)


class TestPulsedCoilSetDesign:
    class MyPulsedCoilset(PulsedCoilsetDesign):  # noqa: D106
        def optimise(self):
            return self.coilset

    def test_breakdown_settings(self):
        mypcs = self.MyPulsedCoilset(*[None] * 7)
        assert isinstance(mypcs._bd_settings, BreakdownCOPSettings)
        mypcs._bd_settings.n_B_stray_points = 9
        assert mypcs._bd_settings.n_B_stray_points == 9

        mypcs = self.MyPulsedCoilset(
            *[None] * 7, breakdown_settings={"n_B_stray_points": 10}
        )
        assert isinstance(mypcs._bd_settings, BreakdownCOPSettings)
        assert mypcs._bd_settings.n_B_stray_points == 10

        with pytest.raises(TypeError):
            mypcs = self.MyPulsedCoilset(*[None] * 7, breakdown_settings={"gamma": 1e-5})

    def test_equilibrium_settings(self):
        mypcs = self.MyPulsedCoilset(*[None] * 7)
        assert isinstance(mypcs._eq_settings, EQSettings)
        mypcs._eq_settings.gamma = 9
        assert mypcs._eq_settings.gamma == 9

        mypcs = self.MyPulsedCoilset(*[None] * 7, equilibrium_settings={"gamma": 1e-5})
        assert isinstance(mypcs._eq_settings, EQSettings)
        assert mypcs._eq_settings.gamma == 1e-5

        with pytest.raises(TypeError):
            mypcs = self.MyPulsedCoilset(
                *[None] * 7, equilibrium_settings={"n_B_stray_points": 10}
            )

    @mock.patch("bluemira.equilibria.run.Snapshot", return_value="SNAP")
    def test_take_snapshot(self, snapshot, caplog):
        mypcs = self.MyPulsedCoilset(*[None] * 7)
        mypcs.take_snapshot("test", "eq", "coilset", "problem", "profiles")
        snapshot.assert_called_with("eq", "coilset", "problem", "profiles", limiter=None)

        assert mypcs.snapshots["test"] == "SNAP"
        # Overwrite snapshot
        mypcs.take_snapshot("test", "eq", "coilset", "problem", "profiles")
        assert len(caplog.messages) == 1
