# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Coilset optimisation problem classes and tools."""

from bluemira.equilibria.optimisation.harmonics.harmonics_approx_functions import (
    coil_harmonic_amplitude_matrix,
    coils_outside_lcfs_sphere,
    collocation_points,
    get_psi_harmonic_amplitudes,
    harmonic_amplitude_marix,
    lcfs_fit_metric,
    plot_psi_comparision,
    spherical_harmonic_approximation,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_constraint_functions import (
    SphericalHarmonicConstraintFunction,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_constraints import (
    SphericalHarmonicConstraint,
)

__all__ = [
    "coil_harmonic_amplitude_matrix",
    "harmonic_amplitude_marix",
    "collocation_points",
    "lcfs_fit_metric",
    "coils_outside_lcfs_sphere",
    "get_psi_harmonic_amplitudes",
    "spherical_harmonic_approximation",
    "plot_psi_comparision",
    "SphericalHarmonicConstraintFunction",
    "SphericalHarmonicConstraint",
]
