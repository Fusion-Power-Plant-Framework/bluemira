# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from typing import ClassVar

import numpy as np
import pytest

from bluemira.geometry.error import GeometryError
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import PictureFrame
from bluemira.geometry.tools import (
    boolean_common,
    boolean_cut,
    boolean_fuse,
    extrude_shape,
    interpolate_bspline,
    make_circle,
    make_polygon,
    revolve_shape,
)
from eudemo.maintenance import duct_connection
from eudemo.maintenance.duct_connection import (
    VVEquatorialPortDuctBuilder,
    join_ports,
    make_equatorial_port_yz_face,
    pipe_pipe_join,
)
from eudemo.vacuum_vessel import VacuumVesselBuilder


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


class TestJoinPorts:
    """Port joins on a vessel built the way the reactor builds it.

    ``pipe_pipe_join`` loses material here: it fragments and reassembles once per
    port, and the union of its (interior-disjoint) pieces comes out smaller than
    their combined volume. The loss grows with every port and differs between CAD
    backends, so pin the volume rather than a face count.
    """

    VV_PARAMS: ClassVar = {
        "r_vv_ib_in": {"value": 5.1, "unit": "m"},
        "r_vv_ob_in": {"value": 14.5, "unit": "m"},
        "tk_vv_in": {"value": 0.6, "unit": "m"},
        "tk_vv_out": {"value": 1.1, "unit": "m"},
        "g_vv_bb": {"value": 0.02, "unit": "m"},
        "n_TF": {"value": 16, "unit": ""},
        "vv_in_off_deg": {"value": 80, "unit": "deg"},
        "vv_out_off_deg": {"value": 160, "unit": "deg"},
    }
    EP_PARAMS: ClassVar = {
        "R_0": {"value": 9.0, "unit": "m"},
        "n_TF": {"value": 16, "unit": ""},
        "g_cr_ts": {"value": 0.2, "unit": "m"},
        "ep_z_position": {"value": 0.0, "unit": "m"},
        "ep_width": {"value": 1.4, "unit": "m"},
        "ep_height": {"value": 1.2, "unit": "m"},
        "tk_vv_single_wall": {"value": 0.06, "unit": "m"},
    }
    Z_POSITIONS = (0.0, 2.0, -2.0)
    # Both CAD backends agree to ~1e-7, but the value shifts by ~0.02% with
    # upstream geometry changes, so pin it loosely. The incremental route is
    # 0.2% low, well outside this band.
    EXPECTED_VOLUME = 96.42

    @classmethod
    def setup_class(cls):
        ivc_koz = PictureFrame({
            "x1": {"value": 2},
            "ro": {"value": 3.5},
            "ri": {"value": 3},
            "x2": {"value": 10},
        }).create_shape()
        sector = (
            VacuumVesselBuilder(cls.VV_PARAMS, {}, ivc_koz)
            .build()
            .get_component("xyz")
            .get_component("Sector 1")
        )
        cls.target = sector.get_component("Body 1").shape
        cls.target_void = sector.get_component("Vessel voidspace 1").shape

        # Only the bounding box of this wire is read, for the duct's outer end.
        cryostat_xz = make_polygon(
            {"x": [1.0, 17.0, 17.0, 1.0], "z": [-14.0, -14.0, 14.0, 14.0]}, closed=True
        )
        cls.tool_shapes, cls.tool_voids = [], []
        for z_position in cls.Z_POSITIONS:
            params = dict(cls.EP_PARAMS)
            params["ep_z_position"] = {"value": z_position, "unit": "m"}
            xyz = (
                VVEquatorialPortDuctBuilder(
                    params, {"name": f"EP{z_position}"}, cryostat_xz
                )
                .build()
                .get_component("xyz")
            )
            cls.tool_shapes.append(xyz.get_component(f"EP{z_position}").shape)
            cls.tool_voids.append(xyz.get_component(f"EP{z_position} voidspace").shape)

        cls.joined = join_ports(
            cls.target, cls.target_void, cls.tool_shapes, cls.tool_voids
        )

    def test_join_ports_conserves_volume(self):
        assert self.joined.is_valid()
        assert len(self.joined.solids) == 1
        assert self.joined.volume == pytest.approx(self.EXPECTED_VOLUME, rel=1e-3)

    def test_join_ports_beats_the_incremental_route(self):
        """The incremental route must not be silently preferred again.

        It is not merely different -- it drops material that a union cannot drop.
        """
        target, pieces = self.target, None
        for tool, void in zip(self.tool_shapes, self.tool_voids, strict=True):
            if pieces is not None:
                target = boolean_fuse(pieces)
            pieces = pipe_pipe_join(target, self.target_void, tool, void)
        incremental = boolean_fuse(pieces)

        assert incremental.volume < self.joined.volume

    def test_a_hollow_target_shares_no_volume_with_its_own_void(self):
        """The property that makes ``target_void`` a hazard as a cutting tool.

        The wall is built around the chamber, so however intricate the shared
        surface is, the two are interior-disjoint.
        """
        assert boolean_common(self.target, [self.target_void]) == []

    def test_the_body_is_never_cut_by_the_target_void(self, monkeypatch):
        """``target_void`` may trim a duct, but must not cut the joined body.

        Cutting the body by its own chamber is what corrupts the result: OCC
        hands back a solid of negative volume, or none at all, depending on
        which tools accompany it.
        """
        calls = []
        real_cut = duct_connection.boolean_cut

        def spy(shape, tools, *args, **kwargs):
            calls.append((shape, list(tools)))
            return real_cut(shape, tools, *args, **kwargs)

        monkeypatch.setattr(duct_connection, "boolean_cut", spy)
        join_ports(self.target, self.target_void, self.tool_shapes, self.tool_voids)

        assert calls, "join_ports made no cut at all"
        for shape, tools in calls:
            if any(tool is self.target_void for tool in tools):
                assert shape.volume < self.target.volume, (
                    "the target void was used to cut the body, not a duct"
                )

    def test_a_duct_reaching_into_the_chamber_is_removed(self):
        """These ducts do reach in, so the intrusion cut must remove something."""
        assert any(boolean_common(tool, [self.target_void]) for tool in self.tool_shapes)
        assert (
            self.joined.volume
            < boolean_fuse([
                self.target,
                *self.tool_shapes,
            ]).volume
        )

    def test_join_ports_with_a_single_port(self):
        joined = join_ports(
            self.target, self.target_void, self.tool_shapes[:1], self.tool_voids[:1]
        )
        assert joined.is_valid()
        assert len(joined.solids) == 1
        # The opening costs more wall than the thin duct adds back...
        assert joined.volume < boolean_fuse([self.target, self.tool_shapes[0]]).volume
        # ... and one port leaves more of the vessel than three do.
        assert self.joined.volume < joined.volume


class TestJoinPortsSplineVessel:
    """One-pass joins on the spline-bounded vessel.

    Regression for the crash reported on PR #4437: cutting a spline-bounded
    body by the chamber void hands back a solid that fails validation, whether
    or not the void removes any material. The volumes are the ones
    ``TestPipePipeJoinSplineVessel`` pins for the incremental route, so this
    also checks the two routes agree on a shape where the incremental one is
    sound.
    """

    Z_POSITIONS = (0.0, 1.5, -1.5)
    EXPECTED_VOLUME = 31.50472

    @classmethod
    def setup_class(cls):
        cls.target, cls.target_void = TestPipePipeJoinSplineVessel._vessel()
        cls.tool_shapes, cls.tool_voids = [], []
        for z_position in cls.Z_POSITIONS:
            shape, void = TestPipePipeJoinSplineVessel._port(z_position)
            cls.tool_shapes.append(shape)
            cls.tool_voids.append(void)

    def test_all_ports_joined_in_one_pass(self):
        joined = join_ports(
            self.target, self.target_void, self.tool_shapes, self.tool_voids
        )

        assert joined.is_valid()
        assert len(joined.solids) == 1
        assert joined.volume == pytest.approx(self.EXPECTED_VOLUME, abs=1e-4)

    def test_cutting_the_body_by_the_chamber_is_what_breaks(self):
        """Pin the failure this design avoids, so the reason cannot be lost.

        The void here *does* remove material -- these ducts reach into the
        chamber -- and the cut is still unusable. Trimming the ducts instead is
        not an optimisation for the degenerate case, it is the only route that
        survives.
        """
        fused = boolean_fuse([self.target, *self.tool_shapes])

        # The message wraps differently per backend, hence the loose whitespace.
        with pytest.raises(GeometryError, match=r"is not\s+valid"):
            boolean_cut(fused, [self.target_void])


class TestJoinPortsOtherGeometries:
    """Joins on shapes the reactor does not build, and odd port arrangements."""

    @staticmethod
    def _tube(inner_radius=2.0, outer_radius=2.2):
        inner = make_circle(radius=inner_radius, center=(5, -5, 0), axis=(0, 1, 0))
        outer = make_circle(radius=outer_radius, center=(5, -5, 0), axis=(0, 1, 0))
        return (
            extrude_shape(BluemiraFace([outer, inner]), vec=(0, 10, 0)),
            extrude_shape(BluemiraFace(inner), vec=(0, 10, 0)),
        )

    @staticmethod
    def _stub(length, inner_radius=1.0, outer_radius=1.1):
        inner = make_circle(radius=inner_radius, center=(5, 0, 5), axis=(0, 0, 1))
        outer = make_circle(radius=outer_radius, center=(5, 0, 5), axis=(0, 0, 1))
        return (
            extrude_shape(BluemiraFace([outer, inner]), vec=(0, 0, -length)),
            extrude_shape(BluemiraFace(inner), vec=(0, 0, -length)),
        )

    @pytest.mark.parametrize("length", [5, 10])
    def test_a_stub_onto_a_tube(self, length):
        """A stub that stops inside the bore, and one that pierces the far wall."""
        target, target_void = self._tube()
        tool, tool_void = self._stub(length)

        joined = join_ports(target, target_void, [tool], [tool_void])

        assert joined.is_valid()
        assert len(joined.solids) == 1
        # The opening is cut.
        assert joined.volume < boolean_fuse([target, tool]).volume

    def test_a_duct_flush_with_the_bore_leaves_the_chamber_untouched(self):
        """The EU-DEMO arrangement: nothing reaches in, so nothing is trimmed."""
        target, target_void = self._tube()
        tool, tool_void = self._stub(5)
        flush_tool = max(boolean_cut(tool, [target_void]), key=lambda s: s.volume)
        flush_void = max(boolean_cut(tool_void, [target_void]), key=lambda s: s.volume)

        assert boolean_common(flush_tool, [target_void]) == []

        joined = join_ports(target, target_void, [flush_tool], [flush_void])
        expected = max(
            boolean_cut(boolean_fuse([target, flush_tool]), [flush_void]),
            key=lambda s: s.volume,
        )

        assert joined.is_valid()
        assert joined.volume == pytest.approx(expected.volume, rel=1e-9)

    def test_a_join_without_ports_returns_the_target(self):
        target, target_void = self._tube()

        joined = join_ports(target, target_void, [], [])

        assert joined.is_valid()
        assert joined.volume == pytest.approx(target.volume, rel=1e-9)
