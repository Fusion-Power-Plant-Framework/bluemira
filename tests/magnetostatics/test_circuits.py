# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

import json
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy.interpolate import UnivariateSpline, interp1d

from bluemira.base.constants import EPS
from bluemira.base.file import get_bluemira_path
from bluemira.geometry._private_tools import offset
from bluemira.geometry.coordinates import Coordinates, vector_lengthnorm
from bluemira.geometry.parameterisations import PictureFrame, PrincetonD, TripleArc
from bluemira.geometry.tools import make_circle
from bluemira.magnetostatics.baseclass import SourceGroup
from bluemira.magnetostatics.circuits import (
    ArbitraryPlanarPolyhedralXSCircuit,
    ArbitraryPlanarRectangularXSCircuit,
    HelmholtzCage,
)
from bluemira.magnetostatics.circular_arc import CircularArcCurrentSource
from bluemira.magnetostatics.polyhedral_prism import _field_fabbri
from bluemira.magnetostatics.semianalytic_2d import (
    semianalytic_Bx,
    semianalytic_Bz,
    semianalytic_psi,
)
from bluemira.magnetostatics.trapezoidal_prism import TrapezoidalPrismCurrentSource
from tests.magnetostatics.setup_methods import make_xs_from_bd, plane_setup


def test_analyticalsolvergrouper():
    xc, zc = 4, 4
    current = 1e6
    dx_coil, dz_coil = 0.5, 0.75

    # Build a corresponding arbitrary current loop
    circle = make_circle(center=[0, 0, zc], radius=xc).discretize(ndiscr=10)
    a = ArbitraryPlanarRectangularXSCircuit(circle, dx_coil, dz_coil, current)
    circle2 = make_circle(center=[0, 0, -zc], radius=xc).discretize(ndiscr=10)
    a2 = ArbitraryPlanarRectangularXSCircuit(circle2, dx_coil, dz_coil, current)
    solver = SourceGroup([a, a2])

    rng = np.random.default_rng()
    points = rng.uniform(low=-10, high=10, size=(10, 3))
    for point in points:
        field = solver.field(*point)  # random point :)
        field2 = a.field(*point) + a2.field(*point)
        assert np.all(field == field2)

    field = solver.field(*points.T)
    new_current = 2e6
    solver.set_current(new_current)
    field_new = solver.field(*points.T)

    assert np.allclose(field_new, new_current / current * field)


def test_sourcegroup_set_current():
    circle = make_circle(radius=10).discretize(ndiscr=50)
    dx_coil, dz_coil = 0.5, 0.75
    a = ArbitraryPlanarRectangularXSCircuit(circle, dx_coil, dz_coil, current=1)
    x, y, z = 4, 4, 4
    response = a.field(x, y, z)
    new_current = 1e6
    a.set_current(new_current)
    new_response = a.field(x, y, z)
    assert np.allclose(new_response, new_current * response)


def test_mixedsourcesolver():
    current = 1e6
    dx = 0.125
    dz = 0.25
    bar_1 = TrapezoidalPrismCurrentSource(
        [0, 0, 2], [-2, 0, 0], [0, 0, -1], [0, 1, 0], dx, dz, 0, 0, current
    )
    bar_2 = TrapezoidalPrismCurrentSource(
        [-2, 0, 0], [0, 0, -2], [1, 0, 0], [0, 1, 0], dx, dz, 0, 0, current
    )
    bar_3 = TrapezoidalPrismCurrentSource(
        [0, 0, -2], [2, 0, 0], [0, 0, 1], [0, 1, 0], dx, dz, 0, 0, current
    )
    bar_4 = TrapezoidalPrismCurrentSource(
        [2, 0, 0], [0, 0, 2], [-1, 0, 0], [0, 1, 0], dx, dz, 0, 0, current
    )

    arc_1 = CircularArcCurrentSource(
        [-1, 0, 1], [0, 0, 1], [-1, 0, 0], [0, 1, 0], dz, dx, 1, 90, current
    )
    arc_2 = CircularArcCurrentSource(
        [-1, 0, -1], [-1, 0, 0], [0, 0, -1], [0, 1, 0], dz, dx, 1, 90, current
    )
    arc_3 = CircularArcCurrentSource(
        [1, 0, -1], [0, 0, -1], [1, 0, 0], [0, 1, 0], dz, dx, 1, 90, current
    )
    arc_4 = CircularArcCurrentSource(
        [1, 0, 1], [1, 0, 0], [0, 0, 1], [0, 1, 0], dz, dx, 1, 90, current
    )

    solver = SourceGroup([bar_1, bar_2, bar_3, bar_4, arc_1, arc_2, arc_3, arc_4])

    nx, nz = 20, 20
    nx2, nz2 = nx // 2, nz // 2
    x = np.linspace(-4, 4, nx)
    z = np.linspace(-4, 4, nz)
    xx, zz = np.meshgrid(x, z, indexing="ij")

    _, Bt, _ = solver.field(xx, np.zeros_like(xx), zz)

    # Test symmetry of the field in four quadranrs (rotation by matrix manipulations :))
    # Bottom-left (reference)
    bt_bl = Bt[:nx2, :nz2]
    # Bottom-right
    bt_br = Bt[nx2:, :nz2][::-1].T
    # Top-right
    bt_tr = Bt[nx2:, nz2:][::-1].T[::-1]
    # Top-left
    bt_tl = Bt[:nx2, nz2:].T[::-1]

    assert np.allclose(bt_bl, bt_br)
    assert np.allclose(bt_bl, bt_tr)
    assert np.allclose(bt_bl, bt_tl)

    solver.plot()
    _, ax = plt.subplots()
    ax.contourf(xx, zz, Bt)
    ax.set_aspect("equal")


class TestArbitraryPlanarXSCircuit:
    pd_inputs: ClassVar = {"x1": {"value": 4}, "x2": {"value": 16}, "dz": {"value": 0}}

    pf_inputs: ClassVar = {
        "x1": {"value": 5},
        "x2": {"value": 10},
        "z1": {"value": 10},
        "z2": {"value": -9},
        "ri": {"value": 0.4},
        "ro": {"value": 1},
    }
    ta_inputs: ClassVar = {
        "x1": {"value": 4},
        "dz": {"value": 0},
        "sl": {"value": 6.5},
        "f1": {"value": 3},
        "f2": {"value": 4},
        "a1": {"value": 20},
        "a2": {"value": 40},
    }

    p_inputs = (pd_inputs, ta_inputs, pf_inputs)
    clockwises = [False] * len(p_inputs) + [True] * len(p_inputs)
    p_inputs = p_inputs * 2  # noqa: PIE794
    parameterisations = tuple(
        [
            PrincetonD,
            TripleArc,
            PictureFrame,
        ]
        * 2
    )

    @pytest.mark.parametrize(
        ("parameterisation", "inputs", "clockwise"),
        zip(parameterisations, p_inputs, clockwises),
    )
    def test_circuits_are_continuous_and_chained(
        self, parameterisation, inputs, clockwise
    ):
        shape = parameterisation(inputs).create_shape()
        coords = shape.discretize(ndiscr=50, byedges=True)
        coords.set_ccw((0, -1, 0))
        if clockwise:
            coords.set_ccw((0, 1, 0))
        circuit = ArbitraryPlanarRectangularXSCircuit(
            coords,
            0.25,
            0.5,
            1.0,
        )
        open_circuit = ArbitraryPlanarRectangularXSCircuit(
            coords[:, :25].T, 0.25, 0.5, 1.0
        )
        n_chain = int(self._calc_daisychain(circuit))
        n_sources = len(circuit.sources) - 1
        assert n_chain == n_sources
        assert self._check_continuity(circuit.sources[-1], circuit.sources[0])
        assert self._calc_daisychain(open_circuit) == len(open_circuit.sources) - 1

    @pytest.mark.parametrize("clockwise", [False, True])
    def test_a_circuit_from_a_clockwise_circle_is_continuous(self, clockwise):
        shape = make_circle(5, (0, 9, 0), axis=(0, 0, 1))
        coords = shape.discretize(ndiscr=30, byedges=True)
        if clockwise:
            coords.set_ccw((0, 0, 1))
        else:
            coords.set_ccw((0, 0, -1))
        circuit = ArbitraryPlanarRectangularXSCircuit(
            coords,
            0.25,
            0.5,
            1.0,
        )
        assert self._calc_daisychain(circuit) == len(circuit.sources) - 1
        assert self._check_continuity(circuit.sources[-1], circuit.sources[0])

    def _calc_daisychain(self, circuit):
        chain = []
        for i, source_1 in enumerate(circuit.sources[:-1]):
            source_2 = circuit.sources[i + 1]
            daisy = self._check_continuity(source_1, source_2)
            chain.append(daisy)
        return sum(chain)

    @staticmethod
    def _check_continuity(source_1, source_2):
        s1_rect = source_1._points[1][:4]
        s2_rect = source_2._points[0][:4]
        return np.allclose(s1_rect, s2_rect)


class TestArbitraryPlanarPolyhedralCircuit:
    discretisations = (
        make_circle(10).discretize(15),
        make_circle(5).discretize(5),
    )

    @pytest.mark.parametrize("coordinates", discretisations)
    @pytest.mark.parametrize("clockwise", [True, False])
    def test_circuits_are_continuous_and_chained(
        self,
        coordinates,
        clockwise,
    ):
        if clockwise:
            coordinates.set_ccw((0, -1, 0))
        else:
            coordinates.set_ccw((0, 1, 0))

        circuit = ArbitraryPlanarPolyhedralXSCircuit(
            coordinates,
            make_xs_from_bd(0.25, 0.5),
            1.0,
        )

        n_chain = int(self._calc_daisychain(circuit))
        n_sources = len(circuit.sources) - 1
        assert n_chain == n_sources
        assert self._check_continuity(circuit.sources[-1], circuit.sources[0])

    def _calc_daisychain(self, circuit):
        chain = []
        for i, source_1 in enumerate(circuit.sources[:-1]):
            source_2 = circuit.sources[i + 1]
            daisy = self._check_continuity(source_1, source_2)
            chain.append(daisy)
        return sum(chain)

    @staticmethod
    def _check_continuity(source_1, source_2):
        s1_rect = source_1._points[1][:4]
        s2_rect = source_2._points[0][:4]
        return np.allclose(s1_rect, s2_rect)


class TestPolyhedralCircuitPlotting:
    @classmethod
    def setup_class(cls):
        shape = make_circle(6)
        xs = Coordinates({"x": [-1, 1, -1], "z": [-1, 0, 1]})
        xs.translate(xs.center_of_mass)

        cls.circuit = ArbitraryPlanarPolyhedralXSCircuit(
            shape.discretize(ndiscr=15), xs, current=1e6
        )

    def test_field_plot(self):
        x = np.linspace(0.1, 8, 50)
        z = np.linspace(-12, 12, 50)
        xx, zz = np.meshgrid(x, z)
        yy = np.zeros_like(xx)
        self.circuit.plot()
        ax = plt.gca()
        Bx, By, Bz = self.circuit.field(xx, yy, zz)
        B = np.sqrt(Bx**2 + By**2 + Bz**2)
        cm = ax.contourf(xx, B, zz, zdir="y", offset=0)
        cb = plt.gcf().colorbar(cm, shrink=0.46)
        cb.set_label("$B$ [T]")
        plt.show()


class TestPolyhedralFaceContinuity:
    @classmethod
    def setup_class(cls):
        shape = make_circle(radius=10, axis=(0, 1, 0))
        xs = Coordinates({"x": [-1, -1, 1, -1], "z": [-1, 1, 0, -1]})
        xs.translate(xs.center_of_mass)

        cls.circuit = ArbitraryPlanarPolyhedralXSCircuit(
            shape.discretize(ndiscr=15), xs, current=1e6
        )

    def test_field_faces(self):
        point = np.array([2, 2, 2])
        source1 = self.circuit.sources[4]
        source2 = self.circuit.sources[5]
        field1 = source1._rho * _field_fabbri(
            source1._dcm[1],
            np.array([source1._face_points[-1]]),
            np.array([source1._face_normals[-1]]),
            np.array([source1._mid_points[-1]]),
            np.array([3]),
            point,
        )
        field2 = source2._rho * _field_fabbri(
            source2._dcm[1],
            np.array([source2._face_points[0]]),
            np.array([source2._face_normals[0]]),
            np.array([source2._mid_points[0]]),
            np.array([3]),
            point,
        )
        assert np.allclose(field1, -field2, rtol=0, atol=EPS)


class TestCariddiBenchmark:
    """
    This is a code comparison benchmark to some work from F. Villone (CREATE) in
    their report DEMO_D_2M97UY
    """

    @classmethod
    def setup_class(cls):
        root = get_bluemira_path("magnetostatics/test_data", subfolder="tests")
        width = 0.64
        depth = 1.15
        B_0 = 5.77
        R_0 = 8.87
        n_TF = 18

        with open(root + "/DEMO_2015_cariddi_ripple_xz.json") as f:
            data = json.load(f)
            cls.cariddi_ripple = data["z"]

        with open(root + "/DEMO_2015_ripple_xz.json") as f:
            data = json.load(f)
            cls.x_rip = data["x"]
            cls.z_rip = data["z"]

        with open(root + "/DEMO_2015_TF_xz.json") as f:
            data = json.load(f)
            x = data["x"]
            z = data["z"]
            coil_loop = Coordinates({"x": x, "y": 0, "z": z})
            coil_loop.close()
            coil_loop.set_ccw((0, 1, 0))
            linterp = np.linspace(0, 1, 300)
            ll = vector_lengthnorm(*coil_loop)
            x = interp1d(ll, coil_loop.x)(linterp)
            z = interp1d(ll, coil_loop.z)(linterp)
            x, z = offset(x, z, width / 2)

            coil_loop = Coordinates({"x": x, "y": 0, "z": z})

        # Smooth out graphically determined TF centreline...
        length_norm = vector_lengthnorm(x, z)
        l_interp = np.linspace(0, 1, 100)
        x = UnivariateSpline(length_norm, x, s=0.02)(l_interp)
        z = UnivariateSpline(length_norm, z, s=0.02)(l_interp)

        coil_loop = Coordinates({"x": x[10:-10], "y": 0, "z": z[10:-10]})
        coil_loop.close()
        cls.coil_loop = coil_loop

        circuit = ArbitraryPlanarRectangularXSCircuit(
            coil_loop, width / 2, depth / 2, current=1.0
        )

        # Set the current in the HelmholtzCage to generate the desired B_T,0 field
        cage = HelmholtzCage(circuit, n_TF)
        field_response = cage.field(R_0, 0, 0)[1]
        current = B_0 / field_response
        cage.set_current(current)
        cls.cage = cage

    def test_cariddi(self):
        ripple = self.cage.ripple(self.x_rip[1:19], np.zeros(18), self.z_rip[1:19])

        _, (ax2, ax) = plt.subplots(1, 2)
        ax.scatter(list(range(1, 19)), self.cariddi_ripple, marker="o", label="CARIDDI")
        ax.scatter(list(range(1, 19)), ripple, marker="x", label="bluemira", zorder=20)
        ax.legend(loc="upper left")

        ax.set_ylabel("$\\delta_{\\phi}$ [%]")
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        ax.set_xlabel("Point index")
        ax.set_xticks(np.arange(1, 19, 2))

        ax2.plot(self.coil_loop.x, self.coil_loop.z, color="b")
        ax2.plot(self.x_rip[1:19], self.z_rip[1:19], marker=".", color="r")

        assert np.max(np.abs(ripple - self.cariddi_ripple)) < 0.04


class TestPolyhedral2DRing:
    @classmethod
    def setup_class(cls):
        cls.radius = 4
        cls.z = 4
        cls.current = 1e6
        n = 151
        ring = make_circle(cls.radius, [0, 0, cls.z], 0, 360, [0, 0, 1])
        xs = Coordinates({"x": [-1, -1, 1, 1, -1], "z": [-1, 1, 1, -1, -1]})
        xs.translate(xs.center_of_mass)
        cls.poly_circuit = ArbitraryPlanarPolyhedralXSCircuit(
            ring.discretize(ndiscr=n), xs, current=cls.current
        )

    @pytest.mark.longrun
    def test_Bx_Bz_2D(self):
        x = np.linspace(0.1, 10, 20)
        z = np.linspace(0.1, 10, 20)
        xx, zz = np.meshgrid(x, z)
        yy = np.zeros_like(xx)
        Bx, _, Bz = self.poly_circuit.field(xx, yy, zz)
        cBx = semianalytic_Bx(self.radius, self.z, xx, zz, 1.0, 1.0)
        cBz = semianalytic_Bz(self.radius, self.z, xx, zz, 1.0, 1.0)
        Bx_coil = self.current * cBx
        Bz_coil = self.current * cBz
        np.testing.assert_allclose(Bx, Bx_coil, rtol=1e-2)
        np.testing.assert_allclose(Bz, Bz_coil, rtol=1e-2)

    def test_vector_potential_flux(self):
        """
        Tests non-user-facing functionality for vector potential
        """
        coordinates = make_circle(10).discretize(31)
        xs = Coordinates({"x": [-1, 1, 1, -1], "z": [-1, -1, 1, 1]})
        poly = ArbitraryPlanarPolyhedralXSCircuit(coordinates, xs, current=1)
        xx, yy, zz, _, _, _ = plane_setup("y", 5, 15, -5, 5, n=20)
        ay = np.zeros((20, 20))
        for s in poly.sources:
            _, aiy, _ = s.vector_potential(xx, yy, zz)
            ay += aiy

        psi_true = semianalytic_psi(10, 0, xx, zz, 1, 1)
        psi_calc = xx * ay

        _, aix = plt.subplots(1, 2)
        aix[0].contourf(xx, zz, psi_calc)
        aix[0].set_aspect("equal")
        aix[1].contourf(xx, zz, psi_true)
        aix[1].set_aspect("equal")
        plt.show()
        np.testing.assert_allclose(psi_calc, psi_true, rtol=0.015)

    @pytest.mark.longrun
    @pytest.mark.parametrize(
        ("point"),
        [(2, 0, 6), (6, 0, 6), (2, 0, 2), (6, 0, 2)],
    )
    def test_continuity(self, point):
        cBx = semianalytic_Bx(self.radius, self.z, point[0], point[2], 1.0, 1.0)
        cBz = semianalytic_Bz(self.radius, self.z, point[0], point[2], 1.0, 1.0)
        Bx_coil = self.current * cBx
        Bz_coil = self.current * cBz
        Bx, _, Bz = self.poly_circuit.field(*point)

        # only passes at this tolerance
        np.testing.assert_allclose(Bx, Bx_coil, rtol=1e-3)
        np.testing.assert_allclose(Bz, Bz_coil, rtol=1e-3)


class TestArbitraryPlanarPolyhedralPFCoil:
    coordinates = make_circle(10).discretize(31)
    xs = Coordinates({"x": [-1, 1, 1, -1], "z": [-1, -1, 1, 1]})
    poly = ArbitraryPlanarPolyhedralXSCircuit(coordinates, xs, current=1)
    trap = ArbitraryPlanarRectangularXSCircuit(coordinates, 1, 1, current=1)

    @pytest.mark.parametrize("plane", ["x", "y", "z"])
    def test_fields(self, plane):
        xx, yy, zz, i, j, k = plane_setup(plane, 5, 15, -5, 5, n=20)

        f = plt.figure()
        ax = f.add_subplot(1, 2, 1, projection="3d")
        self.trap.plot(ax)
        ax.set_title("ArbitraryRectangular")
        Bx, By, Bz = self.trap.field(xx, yy, zz)
        B_new = np.sqrt(Bx**2 + By**2 + Bz**2)
        args_new = [xx, yy, zz, B_new]
        ax.contourf(args_new[i], args_new[j], args_new[k], zdir=plane, offset=0)

        ax = f.add_subplot(1, 2, 2, projection="3d")
        self.poly.plot(ax)
        ax.set_title("ArbitraryPolyhedral")
        Bx, By, Bz = self.poly.field(xx, yy, zz)
        B_new2 = np.sqrt(Bx**2 + By**2 + Bz**2)
        args_new = [xx, yy, zz, B_new2]
        cm = ax.contourf(args_new[i], args_new[j], args_new[k], zdir=plane, offset=0)
        f.colorbar(cm)
        plt.show()
        np.testing.assert_allclose(B_new, B_new2)
