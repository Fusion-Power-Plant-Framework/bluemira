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
    ToroidalHarmonicConstraintFunction,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_constraints import (
    SphericalHarmonicConstraint,
    ToroidalHarmonicConstraint,
)
from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import (  # noqa: E501
    coil_toroidal_harmonic_amplitude_matrix,
    f_hypergeometric,
    legendre_p,
    legendre_q,
    toroidal_harmonic_approximate_psi,
    toroidal_harmonic_approximation,
    toroidal_harmonic_grid_and_coil_setup,
)

__all__ = [
    "PointType",
    "SphericalHarmonicConstraint",
    "SphericalHarmonicConstraintFunction",
    "ToroidalHarmonicConstraint",
    "ToroidalHarmonicConstraintFunction",
    "coil_harmonic_amplitude_matrix",
    "coil_toroidal_harmonic_amplitude_matrix",
    "coils_outside_fs_sphere",
    "collocation_points",
    "f_hypergeometric",
    "fs_fit_metric",
    "get_psi_harmonic_amplitudes",
    "harmonic_amplitude_marix",
    "legendre_p",
    "legendre_q",
    "plot_psi_comparision",
    "spherical_harmonic_approximation",
    "toroidal_harmonic_approximate_psi",
    "toroidal_harmonic_approximation",
    "toroidal_harmonic_grid_and_coil_setup",
]
