# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Unit and integration tests for FreeGSNKE forward Grad-Shafranov solver bridge.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from eqdsk.errors import NoSingleConventionError

from bluemira.base.file import get_bluemira_path
from bluemira.equilibria import (
    Circuit,
    Coil,
    CoilSet,
    CustomProfile,
    Equilibrium,
    ForwardGSSolver,
    ForwardSolveResult,
    Grid,
    Limiter,
    coilset_to_freegsnke_tokamak,
    profile_to_freegsnke,
    run_forward_solve,
)
from bluemira.equilibria.error import EquilibriaError


@pytest.fixture
def simple_grid() -> Grid:
    """Fixture providing a standard test grid."""
    return Grid(0.5, 2.5, -2.0, 2.0, 33, 65)


@pytest.fixture
def simple_coilset() -> CoilSet:
    """Fixture providing a simple 2-coil set."""
    c1 = Coil(1.5, 1.0, current=1e5, dx=0.05, dz=0.05, name="PF_upper")
    c2 = Coil(1.5, -1.0, current=1e5, dx=0.05, dz=0.05, name="PF_lower")
    return CoilSet(c1, c2)


@pytest.fixture
def mastu_equilibrium() -> Equilibrium:
    """Fixture providing a configured MAST-U-like Equilibrium."""
    path = get_bluemira_path("equilibria", subfolder="examples")
    with open(Path(path, "MAST-U_like_active_coils.json")) as f:
        coil_dict = json.load(f)

    circuits = []
    for n, d in coil_dict.items():
        coils = []
        if n == "Solenoid":
            for i, (r, z) in enumerate(zip(d["R"], d["Z"], strict=False)):
                coil = Coil(
                    r,
                    z,
                    current=5000,
                    dx=d["dR"] / 2,
                    dz=0.00489474,
                    ctype="CS",
                    name=f"{n}_{i}",
                )
                coils.append(coil)
        else:
            for i, (r1, z1, r2, z2) in enumerate(
                zip(d["1"]["R"], d["1"]["Z"], d["2"]["R"], d["2"]["Z"], strict=False)
            ):
                coils.append(
                    Coil(
                        r1,
                        z1,
                        current=0,
                        dx=d["1"]["dR"] / 2,
                        dz=d["1"]["dZ"] / 2,
                        ctype="PF",
                        name=f"{n}U_{i}",
                    )
                )
                coils.append(
                    Coil(
                        r2,
                        z2,
                        current=0,
                        dx=d["2"]["dR"] / 2,
                        dz=d["2"]["dZ"] / 2,
                        ctype="PF",
                        name=f"{n}L_{i}",
                    )
                )
        circuits.append(Circuit(*coils))

    full_coilset = CoilSet(*circuits)

    ref_eq = Equilibrium.from_eqdsk(Path(path, "MASTU-FREEGSNKE.eqdsk"), from_cocos=7)
    pn = np.linspace(0, 1, 50)
    profiles = CustomProfile(
        ref_eq.profiles.pprime(pn),
        ref_eq.ffprime(pn),
        R_0=0.85,
        B_0=0.588,
        I_p=6e5,
    )
    grid = Grid(0.1, 2.0, -2.2, 2.2, 65, 129)
    return Equilibrium(full_coilset, grid, profiles, force_symmetry=True)


class TestBridgeConversions:
    """Test data structure conversion between Bluemira and FreeGSNKE."""

    def test_coilset_to_freegsnke_tokamak_single_coils(
        self, simple_coilset: CoilSet, simple_grid: Grid
    ):
        tokamak = coilset_to_freegsnke_tokamak(simple_coilset, grid=simple_grid)
        assert len(tokamak.coil_names) == 2
        assert "PF_upper" in tokamak.coil_names
        assert "PF_lower" in tokamak.coil_names
        currents = tokamak.getCurrents()
        assert np.isclose(currents["PF_upper"], 1e5)
        assert np.isclose(currents["PF_lower"], 1e5)

    def test_coilset_to_freegsnke_tokamak_circuit(self, simple_grid: Grid):
        c1 = Coil(1.2, 0.8, current=5e4, dx=0.05, dz=0.05, name="c1")
        c2 = Coil(1.2, -0.8, current=5e4, dx=0.05, dz=0.05, name="c2")
        circ = Circuit(c1, c2)
        coilset = CoilSet(circ)

        tokamak = coilset_to_freegsnke_tokamak(coilset, grid=simple_grid)
        assert "circuit_0" in tokamak.coil_names
        currents = tokamak.getCurrents()
        assert np.isclose(currents["circuit_0"], 5e4)

    def test_coilset_to_freegsnke_tokamak_empty_error(self, simple_grid: Grid):
        empty_coilset = CoilSet.__new__(CoilSet)
        empty_coilset._coils = ()
        with pytest.raises(EquilibriaError, match="empty CoilSet"):
            coilset_to_freegsnke_tokamak(empty_coilset, grid=simple_grid)

    def test_coilset_to_freegsnke_tokamak_missing_limiter_and_grid_error(
        self, simple_coilset: CoilSet
    ):
        with pytest.raises(EquilibriaError, match="limiter or grid boundary"):
            coilset_to_freegsnke_tokamak(simple_coilset)

    def test_coilset_with_limiter(self, simple_coilset: CoilSet):
        lim = Limiter([0.6, 2.4, 2.4, 0.6], [-1.8, -1.8, 1.8, 1.8])
        tokamak = coilset_to_freegsnke_tokamak(simple_coilset, limiter=lim)
        assert tokamak.limiter is not None

    def test_profile_to_freegsnke(self, simple_grid: Grid, simple_coilset: CoilSet):
        pn = np.linspace(0.0, 1.0, 50)
        p_bluemira = 1e4 * (1.0 - 0.8 * pn)
        f_bluemira = 0.5 * (1.0 - 0.8 * pn)
        profile = CustomProfile(p_bluemira, f_bluemira, R_0=1.5, B_0=2.0, I_p=5e5)

        tokamak = coilset_to_freegsnke_tokamak(simple_coilset, grid=simple_grid)
        from freegsnke.equilibrium_update import Equilibrium as FreeGSNKE_Equilibrium

        freegsnke_eq = FreeGSNKE_Equilibrium(
            tokamak=tokamak,
            Rmin=simple_grid.x_min,
            Rmax=simple_grid.x_max,
            Zmin=simple_grid.z_min,
            Zmax=simple_grid.z_max,
            nx=simple_grid.nx,
            ny=simple_grid.nz,
        )
        f_prof = profile_to_freegsnke(profile, freegsnke_eq, num_points=50)

        # Verify exact 2*pi scaling for COCOS-7
        np.testing.assert_allclose(f_prof.pprime_data, 2.0 * np.pi * p_bluemira)
        np.testing.assert_allclose(f_prof.ffprime_data, 2.0 * np.pi * f_bluemira)
        assert np.isclose(f_prof.Ip, 5e5)
        assert np.isclose(f_prof.fvac(), 1.5 * 2.0)


class TestForwardSolve:
    """Test running forward solves on Bluemira Equilibrium."""

    def test_forward_solve_missing_coils_error(self, simple_grid: Grid):
        pn = np.linspace(0.0, 1.0, 50)
        profile = CustomProfile(1e4 * (1.0 - 0.8 * pn), 0.5 * (1.0 - 0.8 * pn), R_0=1.5, B_0=2.0, I_p=5e5)
        empty_coilset = CoilSet.__new__(CoilSet)
        empty_coilset._coils = ()
        eq = Equilibrium.__new__(Equilibrium)
        eq.coilset = empty_coilset
        eq.grid = simple_grid
        eq.profiles = profile
        with pytest.raises(EquilibriaError, match="no coils"):
            eq.forward_solve()

    def test_forward_solve_missing_profiles_error(
        self, simple_coilset: CoilSet, simple_grid: Grid
    ):
        pn = np.linspace(0.0, 1.0, 50)
        profile = CustomProfile(1e4 * (1.0 - 0.8 * pn), 0.5 * (1.0 - 0.8 * pn), R_0=1.5, B_0=2.0, I_p=5e5)
        eq = Equilibrium(simple_coilset, simple_grid, profile)
        eq.profiles = None
        with pytest.raises(EquilibriaError, match="no profiles"):
            eq.forward_solve()

    def test_forward_solve_execution(self, mastu_equilibrium: Equilibrium):
        eq = mastu_equilibrium
        result = eq.forward_solve(
            target_relative_tolerance=0.05,
            max_iterations=30,
            verbose=False,
            suppress=True,
        )

        assert isinstance(result, ForwardSolveResult)
        assert result.iterations >= 0
        assert result.time_taken > 0
        assert np.isfinite(result.relative_error)
        assert np.isfinite(result.psi_axis)
        assert np.isfinite(result.psi_boundary)

        # Check in-place updates on Bluemira Equilibrium
        plasma_psi = eq.plasma.psi()
        assert plasma_psi.shape == (eq.grid.nx, eq.grid.nz)
        assert np.any(np.abs(plasma_psi) > 0.0)

        total_psi = eq.psi()
        assert total_psi.shape == (eq.grid.nx, eq.grid.nz)
        assert np.isclose(eq._I_p, 6e5, rtol=1e-3)
        assert eq.psi_ax is not None
        assert eq.psi_b is not None

    def test_forward_gs_solver_runner(self, mastu_equilibrium: Equilibrium):
        eq = mastu_equilibrium
        solver = ForwardGSSolver(
            eq,
            target_relative_tolerance=0.05,
            max_iterations=30,
            order=2,
            force_up_down_symmetric=True,
        )
        result = solver.solve(suppress=True)
        assert isinstance(result, ForwardSolveResult)
        assert result.iterations >= 0


class TestEQDSKCOCOSAutoIdentification:
    """Test EQDSK reading with auto COCOS determination and ambiguity errors."""

    def test_ambiguous_eqdsk_without_from_cocos_raises(self):
        path = get_bluemira_path("equilibria", subfolder="examples")
        with pytest.raises(NoSingleConventionError):
            Equilibrium.from_eqdsk(Path(path, "MASTU-FREEGSNKE.eqdsk"))

    def test_specifying_from_cocos_succeeds(self):
        path = get_bluemira_path("equilibria", subfolder="examples")
        eq = Equilibrium.from_eqdsk(
            Path(path, "MASTU-FREEGSNKE.eqdsk"),
            from_cocos=7,
        )
        assert eq is not None
        assert eq.grid is not None
