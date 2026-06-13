# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from unittest import mock

import numpy as np
import pytest

from bluemira.geometry.error import OutOfBoundsError
from bluemira.geometry.optimisation import optimise_geometry
from bluemira.geometry.optimisation._optimise import KeepOutZone
from bluemira.geometry.optimisation._tools import (
    _canonical_loop_points,
    to_objective,
    to_optimiser_callable,
)
from bluemira.geometry.parameterisations import (
    GeometryParameterisation,
    PictureFrame,
    PrincetonD,
    SextupleArc,
    TripleArc,
)
from bluemira.geometry.tools import make_circle, make_polygon, signed_distance
from bluemira.optimisation.error import GeometryOptimisationError


class TestGeometry:
    @pytest.mark.parametrize(
        "kozs",
        [
            [
                make_polygon(
                    np.array([[3, 13, 13, 3], [0, 0, 0, 0], [-5, -5, 6, 6]]),
                    closed=False,
                )
            ],
            [
                make_polygon(
                    np.array([[3, 13, 13, 3], [0, 0, 0, 0], [-5, -5, 6, 6]]), closed=True
                ),
                make_polygon(
                    np.array([[3, 13, 13, 3], [0, 0, 0, 0], [-5, -5, 6, 6]]),
                    closed=False,
                ),
            ],
        ],
    )
    def test_GeometryOptimisationError_given_unclosed_koz(self, kozs):
        with pytest.raises(GeometryOptimisationError):
            optimise_geometry(
                PictureFrame(), f_objective=lambda _: 1, keep_out_zones=kozs
            )

    @pytest.mark.parametrize("alg", ["SLSQP", "SLSQP_SCIPY"])
    def test_simple_optimisation_with_keep_out_zone(self, alg):
        def length(geom: GeometryParameterisation):
            return geom.create_shape().length

        # Create a PictureFrame with un-rounded edges (a rectangle) and
        # a circular keep-out zone within it.
        # We expect the rectangle to contract such that the distance
        # between the parallel edges is equal to the diameter of the
        # keep-out zone.
        koz_radius = 4.5
        koz_center = (10, 0, 0)
        keep_out_zone = make_circle(radius=koz_radius, center=koz_center, axis=(0, 1, 0))
        parameterisation = PictureFrame({
            # Make sure bounds are set within the keep-out zone so
            # we know it's doing some work
            "x1": {"value": 4.5, "upper_bound": 6, "lower_bound": 3},
            "x2": {"value": 16, "upper_bound": 17.5, "lower_bound": 14.0},
            "z1": {"value": 8, "upper_bound": 15, "lower_bound": 2.5},
            "z2": {"value": -6, "upper_bound": -2.5, "lower_bound": -15},
            "ri": {"value": 0, "fixed": True},
            "ro": {"value": 0, "fixed": True},
        })

        opt_result = optimise_geometry(
            parameterisation,
            length,
            keep_out_zones=[keep_out_zone],
            algorithm=alg,
            opt_conditions={"max_eval": 5000, "ftol_rel": 1e-6},
        )

        optimised_shape = opt_result.geom.create_shape()
        np.testing.assert_array_almost_equal(
            list(optimised_shape.center_of_mass), koz_center, decimal=2
        )
        bounds = optimised_shape.bounding_box
        assert bounds.x_max - bounds.x_min == pytest.approx(2 * koz_radius, rel=0.01)
        assert bounds.z_max - bounds.z_min == pytest.approx(2 * koz_radius, rel=0.01)

    def test_princeton_d(self):
        parameterisation = PrincetonD({
            "x1": {"value": 5.5, "upper_bound": 10, "lower_bound": 5},
            "x2": {"value": 16, "upper_bound": 19.5, "lower_bound": 12.5},
            "dz": {"value": 1, "upper_bound": 1.5, "lower_bound": -1.5},
        })

        original_length = parameterisation.create_shape().length
        koz_radius = 4.5
        koz_center = (12.5, 0, 0)
        keep_out_zone = make_circle(radius=koz_radius, center=koz_center, axis=(0, 1, 0))

        opt_result = optimise_geometry(
            parameterisation,
            lambda geom: geom.create_shape().length,
            keep_out_zones=[keep_out_zone],
            opt_conditions={"max_eval": 5000, "ftol_rel": 1e-6},
        )

        opt_shape = opt_result.geom.create_shape()
        # The x-extrema of the PrincetonD should be right on the edge of
        # the keep-out zone
        assert opt_result.geom.variables["x1"].value == pytest.approx(
            koz_center[0] - koz_radius, rel=1e-3
        )
        assert opt_result.geom.variables["x2"].value == pytest.approx(
            koz_center[0] + koz_radius, rel=1e-3
        )
        assert opt_shape.length < original_length
        signed_dist = signed_distance(opt_shape, keep_out_zone)
        # The PrincetonD should fully enclose the keep-out zone
        np.testing.assert_array_less(signed_dist, np.zeros_like(signed_dist))

    def test_maximise_angles_with_TripleArc(self):
        # Maximise the sum of the angles in a TripleArc. The shape's
        # constraint should guarantee that the sum is never greater than
        # 180 degrees
        angle_vars = ["a1", "a2"]

        def sum_angles(geom: TripleArc) -> float:
            angles = [geom.variables[a].normalised_value for a in angle_vars]
            return np.sum(angles)

        def d_sum_angles(geom: TripleArc) -> np.ndarray:
            grad = np.zeros(len(geom.variables.get_normalised_values()))
            grad[geom.get_x_norm_index("a1")] = 1
            grad[geom.get_x_norm_index("a2")] = 1
            return grad

        arc = TripleArc()

        result = optimise_geometry(
            arc,
            f_objective=lambda geom: -sum_angles(geom),
            df_objective=lambda geom: -d_sum_angles(geom),
            opt_conditions={"max_eval": 5000, "ftol_rel": 1e-6},
        )

        angles = [result.geom.variables[a].value for a in angle_vars]
        # The shape's inequality constraint should mean the sum is
        # strictly less than 180
        assert sum(angles) < 180
        # The maximisation should mean the angles approximately sum to 180
        assert sum(angles) == pytest.approx(180, rel=1e-3)

    def test_maximise_angles_with_SextupleArc(self):
        # Run an optimisation to maximise the size of the angles in a
        # SextupleArc. The shape constraint should ensure the angles do
        # not sum to more than 360 degrees.
        arc = SextupleArc()
        angle_vars = ["a1", "a2", "a3", "a4", "a5"]

        def sum_angles(geom: SextupleArc) -> float:
            angles = [geom.variables[a].normalised_value for a in angle_vars]
            return np.sum(angles)

        result = optimise_geometry(
            arc,
            f_objective=lambda geom: -sum_angles(geom),
            opt_conditions={"max_eval": 5000, "ftol_rel": 1e-6},
            algorithm="SLSQP",
        )

        angles = [result.geom.variables[a].value for a in angle_vars]
        # The shape's inequality constraint should mean the sum is
        # strictly less than 360
        assert sum(angles) < 360
        # The maximisation should mean the angles approximately sum to 360
        assert sum(angles) == pytest.approx(360, rel=1e-2)

    def test_dict_koz_settings_passed_to_discretise(self):
        parameterisation = PictureFrame()
        zone = make_circle(radius=4.5, center=(100, 0, 0), axis=(0, 1, 0))

        with mock.patch.object(zone, "discretise", wraps=zone.discretise) as discr_mock:
            optimise_geometry(
                parameterisation,
                lambda _: 1.0,
                keep_out_zones=[
                    {"wire": zone, "n_discr": 20, "byedges": False, "dl": None}
                ],
                opt_conditions={"max_eval": 1},
            )

        discr_mock.assert_called_once_with(20, byedges=False, dl=None)

    def test_koz_settings_passed_to_discretise(self):
        parameterisation = PictureFrame()
        pf = parameterisation.create_shape()
        zone = make_circle(radius=4.5, center=(100, 0, 0), axis=(0, 1, 0))
        koz = KeepOutZone(
            wire=zone, n_discr=20, byedges=False, dl=None, shape_n_discr=30
        )

        with (
            mock.patch.object(zone, "discretise", wraps=zone.discretise) as discr_mock,
            mock.patch.object(parameterisation, "create_shape", return_value=pf),
            mock.patch.object(pf, "discretise", wraps=pf.discretise) as shape_discr_mock,
        ):
            optimise_geometry(
                parameterisation,
                lambda x: x.create_shape().length,
                keep_out_zones=[koz],
                opt_conditions={"max_eval": 1},
            )

        discr_mock.assert_called_once_with(20, byedges=False, dl=None)
        # Note we expect more than one call to 'geom.discretise' due to
        # gradient approximation of objective and constraints. They
        # should all be using the same discretisation.
        assert all(
            call == mock.call(30, byedges=False)
            for call in shape_discr_mock.call_args_list
        )

    @pytest.mark.parametrize(
        "bad_koz", [{"n_discr": 20, "byedges": False, "dl": None}, 10, None]
    )
    def test_TypeError_given_invalid_koz(self, bad_koz):
        parameterisation = PictureFrame()

        with pytest.raises(TypeError):
            optimise_geometry(
                parameterisation,
                lambda _: 1,
                keep_out_zones=[bad_koz],
                opt_conditions={"max_eval": 1},
            )

    def test_eq_constraint(self):
        def square_constraint(geom: PictureFrame) -> np.ndarray:
            """Constraint to make a PictureFrame a square."""
            x1, x2, z1, z2 = (
                geom.variables.x1,
                geom.variables.x2,
                geom.variables.z1,
                geom.variables.z2,
            )
            return np.array([abs(x2 - x1) - abs(z2 - z1)])

        pf = PictureFrame({
            "x1": {"value": 2, "upper_bound": 4, "lower_bound": 0},
            "x2": {"value": 7.5, "upper_bound": 10, "lower_bound": 5},
            "z1": {"value": 5, "upper_bound": 10, "lower_bound": 4},
            "z2": {"value": -5, "upper_bound": -4, "lower_bound": -10},
            "ri": {"value": 0, "fixed": True},
            "ro": {"value": 0, "fixed": True},
        })
        eq_constraint = {
            "f_constraint": square_constraint,
            "df_constraint": None,
            "tolerance": np.array([1e-8]),
        }

        result = optimise_geometry(
            pf,
            f_objective=lambda g: g.create_shape().length,
            eq_constraints=[eq_constraint],
            opt_conditions={"ftol_rel": 1e-5},
        )

        assert square_constraint(result.geom)[0] == pytest.approx(0, abs=1e-8)


class TestOutOfBoundsPenalty:
    """The geometry-optimisation wrappers penalise (rather than crash on) a
    parameterisation that builds invalid geometry, so an optimiser probing an
    infeasible point recovers. Reproduces the class of EUDEMO demo crashes where
    ``create_shape`` raised ``OutOfBoundsError`` mid-optimisation.
    """

    @staticmethod
    def _fake_geom():
        # Stand-in geometry whose ``variables.set_values_from_norm`` is a no-op.
        def _noop(_x):
            return None

        variables = type("V", (), {"set_values_from_norm": staticmethod(_noop)})
        return type("G", (), {"variables": variables})()

    @staticmethod
    def _raises(excess):
        def objective(_geom):
            raise OutOfBoundsError("out of bounds", excess=excess)

        return objective

    def test_objective_penalises_out_of_bounds_instead_of_raising(self):
        feasible = [10.0, 20.0]

        def objective(_geom):
            if feasible:
                return feasible.pop(0)
            raise OutOfBoundsError("out of bounds", excess=0.5)

        f = to_objective(objective, self._fake_geom())
        assert f([0]) == pytest.approx(10.0)
        assert f([0]) == pytest.approx(20.0)  # worst feasible value seen is now 20
        penalty = f([0])  # next call raises -> penalised, not propagated
        assert np.isfinite(penalty)
        assert penalty > 20.0  # strictly worse than any feasible value

    def test_objective_penalty_grows_with_excess(self):
        small = to_objective(self._raises(0.1), self._fake_geom())([0])
        big = to_objective(self._raises(2.0), self._fake_geom())([0])
        assert big > small

    def test_constraint_penalises_with_matching_shape(self):
        outputs = [np.array([1.0, -2.0, 0.5])]

        def constraint(_geom):
            if outputs:
                return outputs.pop(0)
            raise OutOfBoundsError("out of bounds", excess=1.0)

        f = to_optimiser_callable(constraint, self._fake_geom())
        assert f([0]).shape == (3,)
        penalty = f([0])  # raises -> violated constraint vector of the same shape
        assert penalty.shape == (3,)
        assert np.all(penalty > 0)  # reads as strongly violated

    def test_objective_survives_real_invalid_parameterisation(self):
        # Faithful path: a real TripleArc whose angles are out of bounds makes
        # create_shape raise; the wrapped objective penalises instead of crashing.
        arc = TripleArc()
        arc.adjust_variable("a1", value=120)
        arc.adjust_variable("a2", value=120)  # sum 240 -> create_shape raises
        x = arc.variables.get_normalised_values()
        f = to_objective(lambda g: g.create_shape().length, arc)
        assert np.isfinite(f(x))


class TestCanonicalLoopPoints:
    """``_canonical_loop_points`` resamples a closed loop to a fixed point
    sequence that does not depend on where the discretisation starts or which
    way it winds. This keeps the keep-out-zone constraint vector identical
    between geometry backends, whose wire edge ordering otherwise phase-shifts
    the samples and drives the optimiser into a different basin (the EUDEMO
    wall-silhouette divergence).
    """

    @staticmethod
    def _ellipse(n_pts: int, phase: float = 0.0, *, reverse: bool = False) -> np.ndarray:
        # Dense closed loop: ellipse centred at (10, 0), whose outer-midplane
        # anchor therefore sits at (13, 0).
        theta = np.linspace(0.0, 2 * np.pi, n_pts, endpoint=False) + phase
        pts = np.vstack([10 + 3 * np.cos(theta), 5 * np.sin(theta)])
        return pts[:, ::-1] if reverse else pts

    def test_returns_requested_count(self):
        assert _canonical_loop_points(self._ellipse(400), 47).shape == (2, 47)

    def test_anchored_at_outer_midplane(self):
        out = _canonical_loop_points(self._ellipse(400), 60)
        np.testing.assert_allclose(out[:, 0], [13.0, 0.0], atol=5e-3)

    def test_invariant_to_sample_phase(self):
        # Two dense samplings of the same loop starting at different points
        # (as the two backends do) collapse to the same canonical points.
        a = _canonical_loop_points(self._ellipse(400, phase=0.0), 60)
        b = _canonical_loop_points(self._ellipse(400, phase=0.37), 60)
        assert np.max(np.linalg.norm(a - b, axis=0)) < 1e-2

    def test_invariant_to_winding(self):
        a = _canonical_loop_points(self._ellipse(400), 60)
        b = _canonical_loop_points(self._ellipse(400, reverse=True), 60)
        np.testing.assert_allclose(a, b, atol=1e-9)

    def test_points_lie_on_loop(self):
        out = _canonical_loop_points(self._ellipse(400), 60)
        on_ellipse = ((out[0] - 10) / 3) ** 2 + (out[1] / 5) ** 2
        np.testing.assert_allclose(on_ellipse, 1.0, atol=1e-2)
