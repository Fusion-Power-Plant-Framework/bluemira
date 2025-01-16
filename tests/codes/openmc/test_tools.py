# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from collections.abc import Iterable

import numpy as np
import openmc

from bluemira.codes.openmc.tools import (
    OpenMCEnvironment,
    torus_from_3points,
    torus_from_circle,
)
from bluemira.geometry.coordinates import Coordinates
from bluemira.radiation_transport.neutronics.wires import (
    CircleInfo,
    WireInfo,
    WireInfoList,
)


def check_z_axis_centered(openmc_surface: openmc.Surface):
    """Check that the axis of rotation is centered at (0,0) on the xy-plane."""
    assert openmc_surface.x0 == 0.0
    assert openmc_surface.y0 == 0.0


def check_returned_surface_centered_at_z_axis(func):
    """Decorator to check if something is returned properly."""

    def wrapper(*args, **kwargs):
        returned = func(*args, **kwargs)
        if isinstance(returned, Iterable):
            [check_z_axis_centered(surface) for surface in returned]
        else:
            check_z_axis_centered(returned)

    return wrapper


# All surfaces must be centered at the origin, by design

origin = Coordinates([0.0, 0.0, 0.0])


@check_returned_surface_centered_at_z_axis
def test_ztorus():
    """
    https://docs.openmc.org/en/latest/pythonapi/generated/openmc.ZTorus.html

    The only use-case for the z-torus is to approximate arc of a circle in an
    axisymmetric model, so its axis of revolution should be the z-axis.
    """
    p1 = Coordinates([2, 0, 0.5])
    p2 = Coordinates([1.5, 0, 0])
    p3 = Coordinates([2.5, 0, 0])
    torus_1 = torus_from_3points(p1, p2, p3)
    # surface
    assert torus_1.a == 200  # cm
    assert torus_1.b == 50  # cm
    assert torus_1.c == 50  # cm
    # region
    assert Coordinates(np.array([1.9, 0, 0.1]) * 100) in -torus_1  # center of the torus
    assert origin not in -torus_1

    torus_2 = torus_from_circle([2.0, 0.0, 0.0], 0.5)
    # surface
    assert torus_2.a == 200  # cm
    assert torus_2.b == 50  # cm
    assert torus_2.c == 50  # cm
    # region
    assert (
        Coordinates(np.array([1.9, 0, 0.1]) * 100) in -torus_2
    )  # near center of the torus
    assert (
        Coordinates(np.array([0.0, 2.1, 0.1]) * 100) in -torus_2
    )  # near center of the torus
    assert origin not in -torus_2
    assert torus_2.id == (torus_1.id + 1)
    return torus_1, torus_2


@check_returned_surface_centered_at_z_axis
def test_zplane():
    """https://docs.openmc.org/en/latest/pythonapi/generated/openmc.ZPlane.html"""
    assert True


@check_returned_surface_centered_at_z_axis
def test_zcone():
    """https://docs.openmc.org/en/latest/pythonapi/generated/openmc.ZCone.html"""
    assert True


class TestCSGEnv:
    def setup_method(self):
        self.env = OpenMCEnvironment()

    @check_returned_surface_centered_at_z_axis
    def test_torus_from_wire(self):
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
        torus = self.env.surfaces_from_info_list(wire_info_list)[0][1]
        assert (
            Coordinates(np.array([1.9, 0, 0.1]) * 100) in -torus
        )  # near center of the torus
        assert (
            Coordinates(np.array([0.0, 2.1, 0.1]) * 100) in -torus
        )  # near center of the torus
        assert self.origin not in -torus
        return torus

    @check_returned_surface_centered_at_z_axis
    def zplane_from_wire(self):
        pass

    def test_error_when_not_sharing_neighbouring_planes(self):
        assert True
