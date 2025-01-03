# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import openmc

from bluemira.codes.openmc.make_csg import OpenMCEnvironment
from bluemira.geometry.coordinates import Coordinates
from bluemira.neutronics.wires import (
    CircleInfo,
    WireInfo,
    WireInfoList,
    torus_from_3points,
    torus_from_circle,
)


def check_z_axis_centered(openmc_surface: openmc.Surface):
    """Check that the axis of rotation is centered at (0,0) on the xy-plane."""
    assert openmc_surface.x0 == 0.0
    assert openmc_surface.y0 == 0.0


def check_centered_at_origin(openmc_surface: openmc.Surface):
    """Check that the surface (probably a Z-cylinder) is centered at the origin."""
    assert openmc_surface.x0 == 0.0
    assert openmc_surface.y0 == 0.0
    assert openmc_surface.z0 == 0.0


# They must all be centered at the origin!
class TestCSGEnv:
    def __init__(self):
        env = OpenMCEnvironment()

    def test_ztorus(self):
        """
        https://docs.openmc.org/en/latest/pythonapi/generated/openmc.ZTorus.html

        The only use-case for the z-torus is to approximate arc of a circle in an
        axisymmetric model, so its axis of revolution should be the z-axis.
        """
        origin = Coordinates([0, 0, 0])
        p1 = Coordinates([2, 0, 0.5])
        p2 = Coordinates([1.5, 0, 0])
        p3 = Coordinates([2.5, 0, 0])
        torus_1 = torus_from_3points(p1, p2, p3)
        # surface
        check_z_axis_centered(torus_1)
        assert torus_1.a == 2
        assert torus_1.b == 0.5
        assert torus_1.c == 0.5
        # region
        assert Coordinates([1.9, 0, 0.1]) in -torus_1  # center of the torus
        assert origin not in -torus_1

        torus_2 = torus_from_circle(2.0, 0.5)
        # surface
        check_z_axis_centered(torus_2)
        assert torus_2.a == 2
        assert torus_2.b == 0.5
        assert torus_2.c == 0.5
        # region
        assert Coordinates([1.9, 0, 0.1]) in -torus_2  # center of the torus
        assert Coordinates([0.0, 2.1, 0.1]) in -torus_2  # center of the torus
        assert origin not in -torus_2

        wire_info_list = WireInfoList([
            WireInfo(
                CircleInfo(
                    start_point=[1.5, 0, 0],
                    end_point=[2.5, 0, 0],
                    center=[2, 0, 0],
                    radius=0.5,
                ),
                [[None] * 3, [None] * 3],
            )
        ])
        torus3 = self.env.surfaces_from_info_list(wire_info_list)[0]

    def test_zplane(self):
        """https://docs.openmc.org/en/latest/pythonapi/generated/openmc.ZPlane.html"""
        assert ...

    def test_zcone(self):
        """https://docs.openmc.org/en/latest/pythonapi/generated/openmc.ZCone.html"""
        assert ...

    def test_error_when_not_sharing_neighbouring_planes():
        assert ...
