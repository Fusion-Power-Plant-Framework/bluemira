# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from pathlib import Path

import pytest
from eqdsk.models import Sign

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.flux_surfaces import PartialOpenFluxSurface
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.plane import BluemiraPlane
from bluemira.radiation_transport.flux_surfaces_maker import (
    _get_sep_out_intersection,
    _make_flux_surfaces_ibob,
    _process_first_wall,
    get_array_alpha,
    get_array_x_fw,
    get_array_x_mp,
    get_array_z_fw,
    get_array_z_mp,
)

TEST_PATH = get_bluemira_path("equilibria/test_data", subfolder="tests")


def test_process_first_wall():
    converted_from_open_cw = _process_first_wall(
        Coordinates({"x": [1, 1, 2, 2], "z": [1, 2, 2, 1]})
    )
    closed_ccw_x = [2, 2, 1, 1, 2]
    closed_ccw_z = [1, 2, 2, 1, 1]
    for x1, x2, z1, z2 in zip(
        closed_ccw_x,
        converted_from_open_cw.x,
        closed_ccw_z,
        converted_from_open_cw.z,
        strict=False,
    ):
        assert x1 == x2
        assert z1 == z2


def test_gets():
    coords = Coordinates({"x": [0, 1, 1, 2, 2, 1.2, 2], "z": [3, 2, 1, 1, 2, 2, 3]})
    pofs = PartialOpenFluxSurface(coords)
    fw = Coordinates({"x": [0, 3, 3, 0, 0], "z": [0, 0, 2.5, 2.5, 0]})

    xmp = coords.x[0]
    zmp = coords.z[0]
    xfw = coords.x[-1]
    zfw = coords.z[-1]
    assert xmp == get_array_x_mp([pofs])
    assert zmp == get_array_z_mp([pofs])
    assert xfw == get_array_x_fw([pofs])
    assert zfw == get_array_z_fw([pofs])
    assert get_array_alpha([pofs])[0] is None

    pofs.clip(fw)
    assert pofs.alpha == pytest.approx(0.7854, abs=0.0001)


class TestMakeFS:
    @classmethod
    def setup_class(cls):
        eq_name = "DN-DEMO_eqref.json"
        cls.eq = Equilibrium.from_eqdsk(
            Path(TEST_PATH, eq_name), from_cocos=3, qpsi_sign=Sign.NEGATIVE
        )
        cls.o_point = cls.eq.get_OX_points()[0][0]  # 1st o_point
        cls.yz_plane = BluemiraPlane.from_3_points(
            [0, 0, cls.o_point.z], [1, 0, cls.o_point.z], [1, 1, cls.o_point.z]
        )
        cls.first_wall = Coordinates({
            "x": [5.5, 15, 15, 5.5, 5.5],
            "z": [-7.5, -7.5, 7.5, 7.5, -7.5],
        })

    def test_make_flux_surfaces_ibob(self):
        x_sep_omp, x_out_omp = _get_sep_out_intersection(
            self.eq, self.first_wall, self.yz_plane, outboard=True
        )
        x_sep_imp, x_out_imp = _get_sep_out_intersection(
            self.eq, self.first_wall, self.yz_plane, outboard=False
        )
        lower_fs_ob, upper_fs_ob = _make_flux_surfaces_ibob(
            0.001,
            self.eq,
            self.o_point,
            self.yz_plane,
            x_sep_omp,
            x_out_omp,
            outboard=True,
        )
        lower_fs_ib, upper_fs_ib = _make_flux_surfaces_ibob(
            0.001,
            self.eq,
            self.o_point,
            self.yz_plane,
            x_sep_imp,
            x_out_imp,
            outboard=False,
        )
        # Test that we find the expected number of flux surfaces in our list.
        assert len(lower_fs_ob) == 1070
        assert len(upper_fs_ob) == 1070
        assert len(lower_fs_ib) == 1672
        assert len(upper_fs_ib) == 1672

        assert max(lower_fs_ib[0].coords.x) < min(upper_fs_ob[0].coords.x)
        assert max(lower_fs_ib[0].coords.z) <= min(upper_fs_ob[0].coords.z)
        assert max(upper_fs_ib[0].coords.x) < min(lower_fs_ob[0].coords.x)
        assert min(upper_fs_ib[0].coords.z) >= max(lower_fs_ob[0].coords.z)
        assert max(lower_fs_ob[0].coords.z) <= min(upper_fs_ob[0].coords.z)
