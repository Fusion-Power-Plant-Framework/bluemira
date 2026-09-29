# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Bridge interface between Bluemira and FreeGSNKE.

This module provides data adapters and execution wrappers to delegate
Grad-Shafranov equilibrium solves from Bluemira to FreeGSNKE.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from freegsnke.build_machine import (
    apply_tokamak_components,
    build_tokamak_components,
)
from freegsnke.equilibrium_update import Equilibrium as FreeGSNKE_Equilibrium
from freegsnke.GSstaticsolver import NKGSsolver
from freegsnke.jtor_update import GeneralPprimeFFprime
from freegsnke.machine_update import Machine

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.equilibria.coils import Circuit, Coil, CoilSet, SymmetricCircuit
from bluemira.equilibria.error import EquilibriaError

if TYPE_CHECKING:
    from bluemira.equilibria.equilibrium import Equilibrium
    from bluemira.equilibria.grid import Grid
    from bluemira.equilibria.limiter import Limiter
    from bluemira.equilibria.profiles import Profile


@dataclass
class ForwardSolveResult:
    """
    Diagnostic result container for forward Grad-Shafranov solves.

    Attributes
    ----------
    converged:
        Whether the solver achieved the requested relative tolerance.
    iterations:
        Number of nonlinear iterations performed.
    relative_error:
        Final relative residual error achieved.
    psi_axis:
        Poloidal flux at the magnetic axis in Bluemira units [Wb].
    psi_boundary:
        Poloidal flux at the plasma boundary/LCFS in Bluemira units [Wb].
    plasma_current:
        Total plasma current [A].
    time_taken:
        Execution duration in seconds.
    has_relevant_xpoint:
        Whether the equilibrium has an active X-point defining the LCFS.
    """

    converged: bool
    iterations: int
    relative_error: float
    psi_axis: float
    psi_boundary: float
    plasma_current: float
    time_taken: float
    has_relevant_xpoint: bool


def coilset_to_freegsnke_tokamak(
    coilset: CoilSet,
    limiter: Limiter | None = None,
    grid: Grid | None = None,
) -> Machine:
    """
    Convert a Bluemira CoilSet and Limiter into a FreeGSNKE Machine.

    Parameters
    ----------
    coilset:
        Bluemira CoilSet containing individual coils and circuits.
    limiter:
        Optional Bluemira Limiter defining the wall/limiter boundary.
    grid:
        Optional Bluemira Grid used as fallback boundary if no limiter is provided.

    Returns
    -------
    Machine:
        FreeGSNKE Machine instance initialized with active coils and limiter/wall.

    Raises
    ------
    EquilibriaError:
        If coilset is empty or neither limiter nor grid is supplied.
    """
    if not coilset._coils:
        raise EquilibriaError("Cannot convert an empty CoilSet to FreeGSNKE Machine.")

    active_coils_data: dict[str, Any] = {}
    coil_currents: dict[str, float] = {}

    for idx, element in enumerate(coilset._coils):
        if isinstance(element, (Circuit, SymmetricCircuit)):
            elem_name = (
                element.name
                if isinstance(element.name, str)
                else f"circuit_{idx}"
            )
            circuit_dict: dict[str, Any] = {}
            for sub_idx, subcoil in enumerate(element._coils):
                sub_name = (
                    subcoil.name
                    if getattr(subcoil, "name", None)
                    else f"sub_{sub_idx}"
                )
                if sub_name in circuit_dict:
                    sub_name = f"{sub_name}_{sub_idx}"
                circuit_dict[sub_name] = {
                    "R": [float(subcoil.x)],
                    "Z": [float(subcoil.z)],
                    "dR": float(2.0 * subcoil.dx),
                    "dZ": float(2.0 * subcoil.dz),
                    "resistivity": 1.68e-8,
                    "polarity": 1.0,
                    "multiplier": 1.0,
                }
            active_coils_data[elem_name] = circuit_dict
            coil_currents[elem_name] = float(np.asarray(element.current).flat[0])
        elif isinstance(element, Coil):
            elem_name = element.name if element.name else f"coil_{idx}"
            active_coils_data[elem_name] = {
                "R": [float(element.x)],
                "Z": [float(element.z)],
                "dR": float(2.0 * element.dx),
                "dZ": float(2.0 * element.dz),
                "resistivity": 1.68e-8,
                "polarity": 1.0,
                "multiplier": 1.0,
            }
            coil_currents[elem_name] = float(np.asarray(element.current).flat[0])
        else:
            bluemira_warn(f"Skipping unsupported coil element of type {type(element)}")

    if limiter is not None and hasattr(limiter, "x") and len(limiter.x) > 0:
        limiter_data = [
            {"R": float(r), "Z": float(z)}
            for r, z in zip(limiter.x, limiter.z, strict=False)
        ]
    elif grid is not None:
        dr = 0.01 * (grid.x_max - grid.x_min)
        dz = 0.01 * (grid.z_max - grid.z_min)
        limiter_data = [
            {"R": float(grid.x_min + dr), "Z": float(grid.z_min + dz)},
            {"R": float(grid.x_max - dr), "Z": float(grid.z_min + dz)},
            {"R": float(grid.x_max - dr), "Z": float(grid.z_max - dz)},
            {"R": float(grid.x_min + dr), "Z": float(grid.z_max - dz)},
        ]
    else:
        raise EquilibriaError(
            "A limiter or grid boundary must be provided to construct a FreeGSNKE Machine."
        )

    components = build_tokamak_components(
        active_coils_data=active_coils_data,
        limiter_data=limiter_data,
    )
    tokamak = Machine(
        components["coil_circuits"],
        wall=components["wall"],
        limiter=components["limiter"],
    )
    apply_tokamak_components(tokamak, components, rebuild_R_and_M=False)

    for name, current in coil_currents.items():
        if name in tokamak.coil_names:
            tokamak.set_coil_current(name, current)

    return tokamak


def profile_to_freegsnke(
    profile: Profile,
    freegsnke_eq: FreeGSNKE_Equilibrium,
    num_points: int = 101,
) -> GeneralPprimeFFprime:
    """
    Convert a Bluemira Profile to FreeGSNKE GeneralPprimeFFprime profile object.

    Applies exact COCOS-7 flux scaling (multiplication by 2*pi for p' and ff')
    to account for Bluemira's COCOS-11 (Wb) vs FreeGSNKE's COCOS-7 (Wb/rad).

    Parameters
    ----------
    profile:
        Bluemira Profile instance.
    freegsnke_eq:
        Associated FreeGSNKE Equilibrium instance.
    num_points:
        Number of discretization points along normalized poloidal flux psi_n in [0, 1].

    Returns
    -------
    GeneralPprimeFFprime:
        FreeGSNKE profile ready for forward solve.
    """
    psi_n = np.linspace(0.0, 1.0, num_points)

    pprime_bluemira = profile.pprime(psi_n)
    ffprime_bluemira = profile.ffprime(psi_n)

    # Conversion: psi_f = psi_b / (2*pi)
    # dp/dpsi_f = (dp/dpsi_b) * (dpsi_b/dpsi_f) = 2*pi * dp/dpsi_b
    # F dF/dpsi_f = 2*pi * F dF/dpsi_b
    pprime_freegsnke = 2.0 * np.pi * np.asarray(pprime_bluemira, dtype=np.float64)
    ffprime_freegsnke = 2.0 * np.pi * np.asarray(ffprime_bluemira, dtype=np.float64)

    ip = float(profile.I_p) if profile.I_p is not None else 0.0

    if getattr(profile, "B_0", None) is not None and getattr(profile, "R_0", None) is not None:
        fvac = float(profile.R_0 * profile.B_0)
    elif hasattr(profile, "fvac") and callable(profile.fvac):
        fvac = float(profile.fvac())
    else:
        fvac = 1.0

    return GeneralPprimeFFprime(
        freegsnke_eq,
        Ip=ip,
        fvac=fvac,
        psi_n=psi_n,
        pprime_data=pprime_freegsnke,
        ffprime_data=ffprime_freegsnke,
        interpolator="univariate_spline",
    )


def update_bluemira_from_freegsnke(
    bluemira_eq: Equilibrium,
    freegsnke_eq: FreeGSNKE_Equilibrium,
    freegsnke_profiles: GeneralPprimeFFprime,
) -> None:
    """
    Transfer converged solution state from FreeGSNKE back into Bluemira Equilibrium.

    Converts poloidal flux from COCOS-7 (Wb/rad) to COCOS-11 (Wb) via 2*pi scaling,
    updates the internal plasma state, updates toroidal current density, and
    refreshes critical points and boundary topology.

    Parameters
    ----------
    bluemira_eq:
        Bluemira Equilibrium to update in-place.
    freegsnke_eq:
        Converged FreeGSNKE Equilibrium instance.
    freegsnke_profiles:
        FreeGSNKE Profile instance containing computed toroidal current density.
    """
    # Convert plasma poloidal flux: FreeGSNKE (Wb/rad) -> Bluemira (Wb)
    bluemira_plasma_psi = freegsnke_eq.plasma_psi * (2.0 * np.pi)

    # Toroidal current density Jtor has identical physical units [A/m^2]
    jtor = np.asarray(freegsnke_profiles.jtor, dtype=np.float64).copy()

    # Update plasma coil representation and flux interpolators
    bluemira_eq._update_plasma(bluemira_plasma_psi, jtor)
    bluemira_eq._jtor = jtor

    if hasattr(freegsnke_eq, "_current") and freegsnke_eq._current is not None:
        bluemira_eq._I_p = float(freegsnke_eq._current)

    if hasattr(freegsnke_eq, "psi_axis") and freegsnke_eq.psi_axis is not None:
        bluemira_eq.psi_ax = float(freegsnke_eq.psi_axis * (2.0 * np.pi))

    if hasattr(freegsnke_eq, "psi_bndry") and freegsnke_eq.psi_bndry is not None:
        bluemira_eq.psi_b = float(freegsnke_eq.psi_bndry * (2.0 * np.pi))

    bluemira_eq._plasmacoil = None
    bluemira_eq._clear_OX_points()

    # Re-detect topology
    try:
        bluemira_eq.get_OX_points(force_update=True)
    except Exception as exc:  # noqa: BLE001
        bluemira_warn(
            f"Could not automatically detect OX points following forward solve: {exc}"
        )


def run_forward_solve(
    bluemira_eq: Equilibrium,
    *,
    target_relative_tolerance: float = 1e-6,
    max_iterations: int = 100,
    order: int = 2,
    force_up_down_symmetric: bool | None = None,
    picard_handover: float = 0.11,
    verbose: bool = False,
    suppress: bool = True,
    **solver_kwargs: Any,
) -> ForwardSolveResult:
    """
    Execute FreeGSNKE's static forward Grad-Shafranov solve on a Bluemira Equilibrium.

    Parameters
    ----------
    bluemira_eq:
        Bluemira Equilibrium containing coils, grid, profiles, and optional limiter.
    target_relative_tolerance:
        Relative nonlinear residual convergence threshold. Default is 1e-6.
    max_iterations:
        Maximum nonlinear Newton-Krylov iterations. Default is 100.
    order:
        Finite-difference spatial operator order (2 or 4). Default is 2.
    force_up_down_symmetric:
        Whether to enforce up-down symmetry at each iteration. If None,
        defaults to `bluemira_eq._force_symmetry`.
    Picard_handover:
        Residual tolerance handover threshold between Picard and Newton-Krylov steps.
    verbose:
        Enable verbose iteration logging to stdout. Default is False.
    suppress:
        Suppress FreeGSNKE stdout output. Default is True.
    **solver_kwargs:
        Additional keyword arguments forwarded to FreeGSNKE's `NKGSsolver.forward_solve`.

    Returns
    -------
    ForwardSolveResult:
        Convergence diagnostics and execution metrics.

    Raises
    ------
    EquilibriaError:
        If required equilibrium components are missing or solver fails.
    """
    t0 = time.perf_counter()

    if bluemira_eq.coilset is None or not bluemira_eq.coilset._coils:
        raise EquilibriaError(
            "Cannot perform forward solve: Equilibrium has no coils configured."
        )

    if bluemira_eq.profiles is None:
        raise EquilibriaError(
            "Cannot perform forward solve: Equilibrium has no profiles configured."
        )

    if bluemira_eq.grid is None:
        raise EquilibriaError(
            "Cannot perform forward solve: Equilibrium has no grid configured."
        )

    # 1. Build FreeGSNKE Machine
    tokamak = coilset_to_freegsnke_tokamak(
        bluemira_eq.coilset,
        limiter=bluemira_eq.limiter,
        grid=bluemira_eq.grid,
    )

    # 2. Build FreeGSNKE Equilibrium
    freegsnke_eq = FreeGSNKE_Equilibrium(
        tokamak=tokamak,
        Rmin=float(bluemira_eq.grid.x_min),
        Rmax=float(bluemira_eq.grid.x_max),
        Zmin=float(bluemira_eq.grid.z_min),
        Zmax=float(bluemira_eq.grid.z_max),
        nx=int(bluemira_eq.grid.nx),
        ny=int(bluemira_eq.grid.nz),
    )

    # Warm-start from existing plasma psi if available
    if (
        hasattr(bluemira_eq, "plasma")
        and bluemira_eq.plasma is not None
        and hasattr(bluemira_eq.plasma, "psi")
    ):
        try:
            current_psi = bluemira_eq.plasma.psi()
            if current_psi is not None and np.any(np.abs(current_psi) > 1e-12):
                freegsnke_eq.plasma_psi = current_psi / (2.0 * np.pi)
        except Exception:  # noqa: BLE001
            pass

    # 3. Build FreeGSNKE Profile
    freegsnke_profiles = profile_to_freegsnke(bluemira_eq.profiles, freegsnke_eq)

    # 4. Configure Symmetry
    symmetric = (
        bool(getattr(bluemira_eq, "force_symmetry", False))
        if force_up_down_symmetric is None
        else bool(force_up_down_symmetric)
    )

    # 5. Build and execute Solver
    solver = NKGSsolver(
        freegsnke_eq,
        gs_operator_order=order,
    )

    solver.forward_solve(
        freegsnke_eq,
        freegsnke_profiles,
        target_relative_tolerance=target_relative_tolerance,
        max_solving_iterations=max_iterations,
        Picard_handover=picard_handover,
        force_up_down_symmetric=symmetric,
        verbose=verbose,
        suppress=suppress,
        **solver_kwargs,
    )

    # 6. Back-propagate solution to Bluemira
    update_bluemira_from_freegsnke(bluemira_eq, freegsnke_eq, freegsnke_profiles)

    time_taken = time.perf_counter() - t0
    rel_error = getattr(solver, "relative_change", float("nan"))
    norm_rel = getattr(solver, "norm_rel_change", [])
    iterations = max(0, len(norm_rel) - 1) if norm_rel else 0
    converged = bool(rel_error <= target_relative_tolerance)

    return ForwardSolveResult(
        converged=converged,
        iterations=iterations,
        relative_error=float(rel_error),
        psi_axis=float(bluemira_eq.psi_ax) if bluemira_eq.psi_ax is not None else float("nan"),
        psi_boundary=float(bluemira_eq.psi_b) if bluemira_eq.psi_b is not None else float("nan"),
        plasma_current=float(bluemira_eq._I_p) if bluemira_eq._I_p is not None else float("nan"),
        time_taken=time_taken,
        has_relevant_xpoint=bool(getattr(freegsnke_eq, "has_relevant_xpoint", False)),
    )


class ForwardGSSolver:
    """
    Object-oriented runner for FreeGSNKE static forward Grad-Shafranov solves.

    Parameters
    ----------
    eq:
        Bluemira Equilibrium instance to solve.
    target_relative_tolerance:
        Relative convergence tolerance. Default is 1e-6.
    max_iterations:
        Maximum iterations. Default is 100.
    order:
        Finite-difference operator order (2 or 4). Default is 2.
    force_up_down_symmetric:
        Whether to enforce up-down symmetry.
    Picard_handover:
        Threshold to switch from Picard to Newton-Krylov.
    """

    def __init__(
        self,
        eq: Equilibrium,
        *,
        target_relative_tolerance: float = 1e-6,
        max_iterations: int = 100,
        order: int = 2,
        force_up_down_symmetric: bool | None = None,
        picard_handover: float = 0.11,
    ):
        self.eq = eq
        self.target_relative_tolerance = target_relative_tolerance
        self.max_iterations = max_iterations
        self.order = order
        self.force_up_down_symmetric = force_up_down_symmetric
        self.picard_handover = picard_handover

    def solve(
        self,
        verbose: bool = False,
        suppress: bool = True,
        **kwargs: Any,
    ) -> ForwardSolveResult:
        """
        Execute the forward solve.

        Parameters
        ----------
        verbose:
            Print iteration diagnostics.
        suppress:
            Suppress console output.
        **kwargs:
            Additional arguments forwarded to `run_forward_solve`.

        Returns
        -------
        ForwardSolveResult:
            Convergence metrics and diagnostics.
        """
        return run_forward_solve(
            self.eq,
            target_relative_tolerance=self.target_relative_tolerance,
            max_iterations=self.max_iterations,
            order=self.order,
            force_up_down_symmetric=self.force_up_down_symmetric,
            picard_handover=self.picard_handover,
            verbose=verbose,
            suppress=suppress,
            **kwargs,
        )
