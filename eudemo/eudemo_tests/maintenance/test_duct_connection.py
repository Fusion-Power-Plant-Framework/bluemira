# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import numpy as np
import pytest

from bluemira.geometry.error import GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import (
    boolean_fuse,
    extrude_shape,
    interpolate_bspline,
    make_circle,
    revolve_shape,
)
from eudemo.maintenance.duct_connection import (
    make_equatorial_port_yz_face,
    pipe_pipe_join,
)


class TestPipePipeJoin:
    c1 = make_circle(radius=2.0, center=(5, -5, 0), axis=(0, 1, 0))
    c2 = make_circle(radius=2.2, center=(5, -5, 0), axis=(0, 1, 0))
    face = BluemiraFace([c2, c1])
    void_face = BluemiraFace(c1)
    target = extrude_shape(face, vec=(0, 10, 0))
    target_void = extrude_shape(void_face, vec=(0, 10, 0))

    @pytest.mark.parametrize(("tool_length", "n_faces"), [(5, 7), (10, 10)])
    def test_pipe_pipe_join(self, tool_length, n_faces):
        c1 = make_circle(radius=1.0, center=(5, 0, 5), axis=(0, 0, 1))
        c2 = make_circle(radius=1.1, center=(5, 0, 5), axis=(0, 0, 1))
        face = BluemiraFace([c2, c1])
        void_face = BluemiraFace(c1)
        tool = extrude_shape(face, vec=(0, 0, -tool_length))
        tool_void = extrude_shape(void_face, vec=(0, 0, -tool_length))
        new_shape_pieces = pipe_pipe_join(self.target, self.target_void, tool, tool_void)
        n_actual = len(boolean_fuse(new_shape_pieces).faces)
        assert n_actual == n_faces


class TestPipePipeJoinSplineVessel:
    """Port joins onto a spline-bounded vessel, port after port.

    The reactor adds every port to the *accumulated* result (see
    ``VacuumVessel.add_ports``), so each join works on a busier shape than the
    last. On a vessel whose wall is a B-spline -- which is what
    ``VacuumVesselBuilder`` produces -- the tidy-up inside ``boolean_fuse``
    used to return a corrupt solid: negative volume, no error raised, and the
    *next* join then died on an invalid compound. Cylindrical or polygonal
    vessels never showed it, hence the spline here.
    """

    X_C, SECTOR = 7.5, 22.5  # torus centre [m], sector angle [deg]
    # Reference volumes after 1, 2 and 3 ports; equal on both CAD backends.
    EXPECTED_VOLUMES = (30.51961, 31.01216, 31.50472)

    @classmethod
    def _spline_loop(cls, radius):
        angle = np.linspace(0, 2 * np.pi, 120, endpoint=False)
        return interpolate_bspline(
            {
                "x": cls.X_C + radius * np.cos(angle),
                "y": 0.0,
                "z": radius * np.sin(angle),
            },
            closed=True,
        )

    @classmethod
    def _vessel(cls):
        outer, inner = cls._spline_loop(3.0), cls._spline_loop(2.4)
        return (
            revolve_shape(BluemiraFace([outer, inner]), degree=cls.SECTOR),
            revolve_shape(BluemiraFace(inner), degree=cls.SECTOR),
        )

    @classmethod
    def _port(cls, z_position, x_end=15.0):
        yz_face = make_equatorial_port_yz_face(
            x_end, -0.6, 0.6, z_position - 0.5, z_position + 0.5, 0.06
        )
        vec = (cls.X_C - x_end, 0, 0)
        shape = extrude_shape(yz_face, vec)
        void = extrude_shape(BluemiraFace(yz_face.boundary[1]), vec)
        shape.rotate(degree=0.5 * cls.SECTOR)
        void.rotate(degree=0.5 * cls.SECTOR)
        return shape, void

    def test_sequential_joins_stay_valid(self):
        target, target_void = self._vessel()

        for z_position, expected in zip(
            (0.0, 1.5, -1.5), self.EXPECTED_VOLUMES, strict=True
        ):
            tool, tool_void = self._port(z_position)
            pieces = pipe_pipe_join(target, target_void, tool, tool_void)
            target = boolean_fuse(pieces)

            assert target.is_valid()
            assert len(target.solids) == 1
            # A corrupt fuse still reports is_valid(); the volume is what gives
            # it away, so assert the value rather than just its sign.
            assert target.volume == pytest.approx(expected, abs=1e-4)

    def test_non_intersecting_port_raises(self):
        """A port that misses the vessel says so, rather than IndexError-ing."""
        target, target_void = self._vessel()
        tool, tool_void = self._port(9.0)  # well above the vessel

        with pytest.raises(GeometryError, match="do not intersect"):
            pipe_pipe_join(target, target_void, tool, tool_void)
