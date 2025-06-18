# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Coilset optimisation problem classes and tools."""

from bluemira.equilibria.optimisation.harmonics.harmonics_approx_functions import (
    PointType,
    coil_harmonic_amplitude_matrix,
    coils_outside_fs_sphere,
    collocation_points,
    fs_fit_metric,
    get_psi_harmonic_amplitudes,
    harmonic_amplitude_marix,
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
    "PointType",
    "SphericalHarmonicConstraint",
    "SphericalHarmonicConstraintFunction",
    "coil_harmonic_amplitude_matrix",
    "coils_outside_fs_sphere",
    "collocation_points",
    "fs_fit_metric",
    "get_psi_harmonic_amplitudes",
    "harmonic_amplitude_marix",
    "plot_psi_comparision",
    "spherical_harmonic_approximation",
]
