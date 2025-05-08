# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from bluemira.base.constants import EPS
from bluemira.base.file import get_bluemira_path
from bluemira.equilibria import Equilibrium
from bluemira.equilibria.error import EquilibriaError, FluxSurfaceError
from bluemira.equilibria.find import find_flux_surface_through_point
from bluemira.equilibria.find_legs import (
    LegFlux,
    calculate_connection_length,
    get_legs_length_and_angle,
)
from bluemira.equilibria.flux_surfaces import (
    ClosedFluxSurface,
    FieldLineTracer,
    OpenFluxSurface,
    PartialOpenFluxSurface,
    calculate_connection_length_flt,
    calculate_connection_length_fs,
    poloidal_angle,
)
from bluemira.equilibria.shapes import flux_surface_cunningham, flux_surface_johner
from bluemira.geometry.coordinates import (
    Coordinates,
    coords_plane_intersect,
    interpolate_points,
)
from bluemira.geometry.plane import BluemiraPlane
from bluemira.geometry.tools import _signed_distance_2D

TEST_PATH = get_bluemira_path("equilibria/test_data", subfolder="tests")


class TestOpenFluxSurfaceStuff:
    @classmethod
    def setup_class(cls):
        eq_name = "eqref_OOB.json"
        cls.eq = Equilibrium.from_eqdsk(Path(TEST_PATH, eq_name), from_cocos=7)

    def test_bad_geometry(self):
        closed_coords = Coordinates({"x": [0, 4, 5, 8, 0], "z": [1, 2, 3, 4, 1]})
        with pytest.raises(FluxSurfaceError):
            _ = OpenFluxSurface(closed_coords)
        with pytest.raises(FluxSurfaceError):
            _ = PartialOpenFluxSurface(closed_coords)

    def test_connection_length(self):
        """
        Use both a flux surface and field line tracing approach to calculate connection
        length and check they are the same or similar.
        """
        x_start, z_start, psi_norm_start = 12, 0, 1.076675434621207
        x_loop, z_loop = find_flux_surface_through_point(
            self.eq.x,
            self.eq.z,
            self.eq.psi(),
            x_start,
            z_start,
            self.eq.psi(x_start, z_start),
        )
        fs = OpenFluxSurface(Coordinates({"x": x_loop, "z": z_loop}))
        lfs, hfs = fs.split(self.eq.get_OX_points()[0][0])
        l_lfs = lfs.connection_length(self.eq)
        l_hfs = hfs.connection_length(self.eq)

        # test discretisation sensitivity
        lfs_loop = deepcopy(lfs.coords)
        lfs_loop = Coordinates(interpolate_points(*lfs_loop.xyz, 3 * len(lfs_loop)))
        lfs_interp = PartialOpenFluxSurface(lfs_loop)
        l_lfs_interp = lfs_interp.connection_length(self.eq)
        assert np.isclose(l_lfs, l_lfs_interp, rtol=5e-3)

        hfs_loop = deepcopy(hfs.coords)
        hfs_loop = Coordinates(interpolate_points(*hfs_loop.xyz, 3 * len(hfs_loop)))
        hfs_interp = PartialOpenFluxSurface(hfs_loop)
        l_hfs_interp = hfs_interp.connection_length(self.eq)
        assert np.isclose(l_hfs, l_hfs_interp, rtol=5e-3)

        # compare with field line tracer
        flt = FieldLineTracer(self.eq)
        l_flt_lfs = flt.trace_field_line(x_start, z_start, n_turns_max=20, forward=True)
        l_flt_hfs = flt.trace_field_line(
            x_start, z_start, n_turns_max=20, forward=False
        ).connection_length
        print(len(l_flt_lfs.coords))
        assert np.isclose(l_flt_lfs.connection_length, l_lfs, rtol=2e-2)
        assert np.isclose(l_flt_hfs, l_hfs, rtol=2e-2)

        # Check the calculate_connection_length functions
        x1, x2 = self.eq.grid.x_min, self.eq.grid.x_max
        z1, z2 = self.eq.grid.z_min, self.eq.grid.z_max
        flux_intercepting_surface = Coordinates({
            "x": [x1, x2, x2, x1, x1],
            "z": [z1, z1, z2, z2, z1],
        })
        l_fsg_fwd = calculate_connection_length_fs(
            eq=self.eq,
            x=x_start,
            z=z_start,
            flux_intercepting_surface=flux_intercepting_surface,
        )
        l_flt_fwd = calculate_connection_length_flt(
            eq=self.eq,
            x=x_start,
            z=z_start,
            flux_intercepting_surface=flux_intercepting_surface,
        )
        l_fsg_bwd = calculate_connection_length_fs(
            eq=self.eq,
            x=x_start,
            z=z_start,
            flux_intercepting_surface=flux_intercepting_surface,
            forward=False,
        )
        l_flt_bwd = calculate_connection_length_flt(
            eq=self.eq,
            x=x_start,
            z=z_start,
            flux_intercepting_surface=flux_intercepting_surface,
            forward=False,
        )
        assert np.isclose(l_fsg_fwd, l_flt_fwd, rtol=2e-2)
        assert np.isclose(l_fsg_bwd, l_flt_bwd, rtol=2e-2)

        # Check the calculate_connection_length function, that calls either
        # method, returns the same result for the same selected point
        xz = Coordinates({"x": [x_start], "z": [z_start]})
        l_fsg = calculate_connection_length(
            self.eq,
            div_target_start_point=xz,
            calculation_method="flux_surface_geometry",
        )
        l_flt = calculate_connection_length(
            self.eq,
            div_target_start_point=xz,
            n_turns_max=20,
            calculation_method="field_line_tracer",
        )
        assert np.isclose(l_fsg, l_lfs, rtol=2e-2)
        assert np.isclose(l_flt, l_flt_lfs.connection_length, rtol=2e-2)

        # Check the calculate_connection_length function, that calls either
        # method, returns the same result for the normalised psi input that
        # matches the normalise psi of the div_target_start_point
        l_fsg_psinorm = calculate_connection_length(
            self.eq,
            div_target_start_point=xz,
            div_norm_psi=psi_norm_start,
            calculation_method="flux_surface_geometry",
        )
        l_flt_psinorm = calculate_connection_length(
            self.eq,
            div_target_start_point=xz,
            div_norm_psi=psi_norm_start,
            n_turns_max=20,
            calculation_method="field_line_tracer",
        )
        assert np.isclose(l_fsg_psinorm, l_fsg, rtol=2e-2)
        assert np.isclose(l_flt_psinorm, l_flt, rtol=2e-2)
        l_fsg_psinorm_bwd = calculate_connection_length(
            self.eq,
            div_norm_psi=psi_norm_start,
            forward=False,
            calculation_method="flux_surface_geometry",
        )
        l_flt_psinorm_bwd = calculate_connection_length(
            self.eq,
            div_norm_psi=psi_norm_start,
            forward=False,
            n_turns_max=20,
            calculation_method="field_line_tracer",
        )
        assert l_fsg_psinorm_bwd > l_fsg_psinorm
        assert l_flt_psinorm_bwd > l_flt_psinorm
        # test multiple intersections of first wall
        first_wall_1 = Coordinates({
            "x": [x1, 9.2, 9.2, 9.5, x2, x2, x1, x1],
            "z": [z1, -6.2, -7.5, -7.2, z1, z2, z2, z1],
        })
        l_fw1 = calculate_connection_length(
            self.eq,
            div_norm_psi=psi_norm_start,
            flux_intercepting_surface=first_wall_1,
            calculation_method="flux_surface_geometry",
        )
        first_wall_2 = Coordinates({
            "x": [x1, x2, x2, x1, 6.2, 6.2, 4.8, x1],
            "z": [z1, z1, z2, z2, -7.2, -7.5, -6.2, z1],
        })
        l_fw2 = calculate_connection_length(
            self.eq,
            div_norm_psi=psi_norm_start,
            flux_intercepting_surface=first_wall_2,
            forward=False,
            calculation_method="flux_surface_geometry",
        )
        assert np.isclose(l_fw1, 46.86960546652787, rtol=2e-2)
        assert np.isclose(l_fw2, 138.37688138779316, rtol=2e-2)

        # Test that we get the expected value for a given psi_norm
        # Use a value different from psi_norm_start to make sure that
        # div_target_start_point is set correctly
        l_flt = calculate_connection_length(
            self.eq,
            div_target_start_point=xz,
            div_norm_psi=1.03,
            n_turns_max=20,
            calculation_method="field_line_tracer",
        )
        assert np.isclose(l_flt, 83.13936166158712, rtol=2e-2)

        # Check the calculate_connection_length result when no point is input
        l_fsg = calculate_connection_length(
            self.eq,
            calculation_method="flux_surface_geometry",
        )
        assert np.isclose(l_fsg, 157.85, rtol=2e-2)

        # Check for a closed flux surface
        # Choose (and make sure) flux surface is closed
        closed_fs = self.eq.get_flux_surface(0.90)
        assert closed_fs.closed
        xz = Coordinates([np.max(closed_fs.x), 0.0, 0.0])
        l_fsg = calculate_connection_length(
            self.eq,
            div_target_start_point=xz,
            calculation_method="flux_surface_geometry",
        )
        l_flt = calculate_connection_length(
            self.eq,
            div_target_start_point=xz,
            n_turns_max=20,
            calculation_method="field_line_tracer",
        )
        assert np.isclose(l_fsg, 0.0, rtol=1e-6)
        assert np.isclose(l_flt, 0.0, rtol=1e-6)

    def test_legflux(self):
        """
        Check the LegFlux functionality that relies on
        OpenFluxSurfece or PartialOpenFluxSurface
        """
        # Get the legs from the equilibria
        leg_dict = LegFlux(self.eq).get_legs()
        # Add the upper legs as 'None' for testing
        leg_dict["upper_inner"] = [None]
        leg_dict["upper_outer"] = [None]
        # Get dictionary of calculated lengts and angles
        length_dict, angle_dict = get_legs_length_and_angle(self.eq, leg_dict)
        # Get dictionary of calculated lengts and angles with FW intersection
        fw = Coordinates({"x": [10, 0, 0, 10, 10], "z": [-7.5, -7.5, 7.5, 7.5, -7.5]})
        length_dict_intersected, angle_dict_intersected = get_legs_length_and_angle(
            self.eq, leg_dict, plasma_facing_boundary=fw
        )
        expected_l_dict = {
            "lower_inner": 72.216,
            "lower_outer": 73.209,
            "upper_inner": 0.0,
            "upper_outer": 0.0,
        }
        expected_a_dict = {
            "lower_inner": 0.7463,
            "lower_outer": 1.116,
            "upper_inner": np.pi,
            "upper_outer": np.pi,
        }
        assert length_dict["lower_inner"] > length_dict_intersected["lower_inner"]
        assert length_dict["lower_outer"] > length_dict_intersected["lower_outer"]
        for key in leg_dict:
            assert np.isclose(angle_dict[key], np.pi, rtol=1e-6)
            assert np.isclose(
                length_dict_intersected[key], expected_l_dict[key], rtol=5e-3
            )
            assert np.isclose(
                angle_dict_intersected[key], expected_a_dict[key], rtol=5e-3
            )


class TestClosedFluxSurface:
    def test_bad_geometry(self):
        open_loop = Coordinates({"x": [0, 4, 5, 8], "z": [1, 2, 3, 4]})
        with pytest.raises(FluxSurfaceError):
            _ = ClosedFluxSurface(open_loop)

    def test_symmetric(self):
        kappa = 1.5
        delta = 0.4
        # Note that n=1000%4 == 0
        fs = flux_surface_cunningham(7, 0, 1, kappa, delta, n=1000)
        fs.close()
        fs = ClosedFluxSurface(fs)
        assert np.isclose(fs.kappa, kappa)
        assert np.isclose(fs.kappa_lower, kappa)
        assert np.isclose(fs.kappa_upper, kappa)
        assert np.isclose(fs.delta_lower, fs.delta_upper)
        assert np.isclose(fs.zeta_lower, fs.zeta_upper)

    def test_johner(self):
        R_0, z_0, a, kappa_u, kappa_l, delta_u, delta_l, a1, a2, a3, a4 = (
            7,
            0,
            2,
            1.9,
            1.6,
            0.4,
            0.33,
            0,
            0,
            45,
            30,
        )
        fs = flux_surface_johner(
            R_0, z_0, a, kappa_u, kappa_l, delta_u, delta_l, a1, a2, a3, a4, n=1000
        )
        fs.close()
        fs = ClosedFluxSurface(fs)

        assert np.isclose(fs.major_radius, R_0)
        assert np.isclose(fs._z_centre, z_0)
        assert np.isclose(fs.minor_radius, a)
        assert np.isclose(fs.kappa, np.average([kappa_l, kappa_u]))
        assert np.isclose(fs.kappa_upper, kappa_u)
        assert np.isclose(fs.kappa_lower, kappa_l)
        assert np.isclose(fs.delta, np.average([delta_l, delta_u]))
        assert np.isclose(fs.delta_upper, delta_u)
        assert np.isclose(fs.delta_lower, delta_l)
        assert not np.isclose(fs.zeta_upper, fs.zeta_lower)


class TestFieldLine:
    @classmethod
    def setup_class(cls):
        eq_name = "eqref_OOB.json"
        cls.eq = Equilibrium.from_eqdsk(Path(TEST_PATH, eq_name), from_cocos=7)
        cls.flt = FieldLineTracer(cls.eq)
        cls.field_line = cls.flt.trace_field_line(13, 0, n_points=1000)

    def test_non_planar_coodinates_raises_error(self):
        with pytest.raises(EquilibriaError):
            FieldLineTracer(
                self.eq,
                Coordinates({
                    "x": [6, 3, 3, 4, 5],
                    "y": [0, 2, 0, 4, 0],
                    "z": [1, 2, 3, 4, 5],
                }),
            )

    def test_connection_length(self):
        assert np.isclose(
            self.field_line.connection_length, self.field_line.coords.length, rtol=5e-2
        )

    def test_connection_length_coordinates_grid(self):
        """
        Check to see behaviour is the same with Coordinates and Grid
        """
        xmin, xmax = self.eq.grid.x_min, self.eq.grid.x_max
        zmin, zmax = self.eq.grid.z_min, self.eq.grid.z_max
        coords = Coordinates({
            "x": [xmin, xmax, xmax, xmin, xmin],
            "y": 0,
            "z": [zmin, zmin, zmax, zmax, zmin],
        })
        flt = FieldLineTracer(self.eq, coords)
        field_line = flt.trace_field_line(13, 0, n_points=1000)
        assert np.isclose(
            field_line.connection_length, field_line.coords.length, rtol=5e-2
        )
        assert np.isclose(
            self.field_line.connection_length, field_line.connection_length
        )
        self._check_endpoint(field_line, coords)

    def test_connection_length_coordinates(self):
        coords = Coordinates({
            "x": [self.eq.grid.x_min, 9, 12, 13, 13, 12.5, 4, self.eq.grid.x_min],
            "y": 0,
            "z": [self.eq.grid.z_min, -7, -7, -6, 6, 5, 7, self.eq.grid.z_min],
        })
        flt = FieldLineTracer(self.eq, coords)
        field_line = flt.trace_field_line(12.5, 0, n_points=1000, forward=False)
        self._check_endpoint(field_line, coords)

    def _check_endpoint(self, field_line, coords, tol=1e-8):
        """
        Check that the end point of a field line lies near enough to the boundary
        """
        end_point = self._extract_endpoint(field_line)
        ep_xz = np.array([end_point[0], end_point[2]])
        assert abs(_signed_distance_2D(ep_xz, coords.xz.T)) < tol

    def _extract_endpoint(self, field_line):
        """
        Get the end point of a field line in 3-D and map it to 2-D
        """
        end_point = field_line.coords.xyz.T[-1]
        r = np.hypot(*end_point[:2])
        z = end_point[2]
        angle = np.linspace(0, 2 * np.pi, 1000)
        circle = Coordinates({"x": r * np.cos(angle), "y": r * np.sin(angle), "z": z})
        inters = coords_plane_intersect(circle, BluemiraPlane(axis=(0, 1, 0)))
        return next(i for i in inters if i[0] > 0)


def test_poloidal_angle():
    eq_name = "DN-DEMO_eqref.json"
    eq = Equilibrium.from_eqdsk(
        Path(TEST_PATH, eq_name), from_cocos=3, qpsi_positive=False
    )
    # Building inputs
    x_strike = 10.0
    z_strike = -7.5
    Bp_strike = eq.Bp(x_strike, z_strike)
    Bt_strike = eq.Bt(x_strike)
    # Glancing angle
    gamma = 5.0
    # Poloidal angle
    theta = poloidal_angle(Bp_strike, Bt_strike, gamma)
    assert theta > gamma
    # By hand, from a different calculation
    assert round(theta, 1) == pytest.approx(20.6, rel=0, abs=EPS)
