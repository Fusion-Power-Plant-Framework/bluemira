# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Harmonics constraint functions.
"""

from typing import Union

import numpy as np
import numpy.typing as npt

from bluemira.base.look_and_feel import bluemira_warn
from bluemira.equilibria.coils import CoilSet
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.optimisation.constraints import (
    UpdateableConstraint,
    _get_dummy_equilibrium,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_approx_functions import (
    coil_harmonic_amplitude_matrix,
)
from bluemira.equilibria.optimisation.harmonics.harmonics_constraint_functions import (
    SphericalHarmonicConstraintFunction,
)
from bluemira.utilities.tools import is_num


class SphericalHarmonicConstraint(UpdateableConstraint):
    """
    Spherical harmonic constraints for the desired core plasma
    of a spherical tokamak equilibria.

    Parameters
    ----------
    ref_harmonics:
        Initial harmonic amplitudes obtained from desired core plasma
        (Returned by spherical_harmonic_approximation)
    scale:
        Current scale with which to calculate the constraints
    eq:
        Equilibrium used to for coilset.
    r_t: float
        (Returned by spherical_harmonic_approximation)
    sh_coil_names:
        Names of the coils to use with SH approximation
        (Returned by spherical_harmonic_approximation)

    """

    def __init__(
        self,
        ref_harmonics: npt.NDArray,
        r_t: float,
        sh_coil_names: list,
        tolerance: Union[float, npt.NDArray] = None,
        constraint_type: str = "equality",
    ):
        if tolerance is None:
            ord_mag = np.floor(np.log10(np.absolute(ref_harmonics))) - 3
            tolerance = [10**x for x in ord_mag]
        if is_num(tolerance):
            tolerance = tolerance * np.ones(len(ref_harmonics))
        elif len(tolerance) != len(ref_harmonics):
            raise ValueError(f"Tolerance vector not of length {len(ref_harmonics)}")
        else:
            pass

        self.constraint_type = constraint_type
        self.tolerance = tolerance

        self.target_harmonics = ref_harmonics
        self.max_degree = len(ref_harmonics) + 1

        self.sh_coil_names = sh_coil_names
        self.r_t = r_t

        self._args = {
            "a_mat": None,
            "b_vec": None,
            "value": 0.0,
            "scale": 1,
        }

    @property
    def control_coil_names(self):
        """
        The names of the allowed control coils when using the
        SH approximation constraints.
        """
        return self.sh_coil_names

    def prepare(self, equilibrium: Equilibrium, I_not_dI=False, fixed_coils=False):
        """
        Prepare the constraint for use in an equilibrium optimisation problem.
        """
        if len(equilibrium.coilset.control) != len(self.sh_coil_names):
            bluemira_warn(
                "You are using too many control coils in your optimisation problem."
                "Please make sure you only use the coils allowed by your SH approx."
            )

        # Passive coils are a TODO in bluemira core
        if I_not_dI:
            equilibrium = _get_dummy_equilibrium(equilibrium)

        # TODO
        if fixed_coils:
            pass

        self._args["a_mat"] = self.control_response(equilibrium.coilset)
        self._args["b_vec"] = self.evaluate()

    def control_response(self, coilset: CoilSet):
        """
        Calculate control response of a CoilSet to the constraint.
        """
        # SH coefficients from function of the current distribution outside of the sphere
        # containing the plasma, i.e., LCFS (r_lcfs)
        # N.B., cannot use coil located within r_lcfs as part of this method.
        return coil_harmonic_amplitude_matrix(
            coilset,
            self.max_degree,
            self.r_t,
            self.sh_coil_names,
        )

    def evaluate(self) -> npt.NDArray:
        """
        Calculate the value of the constraint in an Equilibrium.
        """
        return self.target_harmonics

    def f_constraint(self) -> SphericalHarmonicConstraintFunction:
        """Constraint function."""
        f_constraint = SphericalHarmonicConstraintFunction(**self._args)
        f_constraint.constraint_type = self.constraint_type
        return f_constraint
