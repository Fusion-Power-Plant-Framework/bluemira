# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Harmonics constraint functions.
"""

import matplotlib.patches as patch
import matplotlib.pyplot as plt
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
    ToroidalHarmonicConstraintFunction,
)
from bluemira.equilibria.optimisation.harmonics.toroidal_harmonics_approx_functions import (  # noqa: E501
    ToroidalHarmonicsParams,
    coil_toroidal_harmonic_amplitude_matrix,
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
    r_t: float
        (Returned by spherical_harmonic_approximation)
    sh_coil_names:
        Names of the coils to use with SH approximation
        (Returned by spherical_harmonic_approximation)
    tolerance:
        Tolerance with which the constraint(s) will be met

    """

    def __init__(
        self,
        ref_harmonics: npt.NDArray[np.float64],
        r_t: float,
        sh_coil_names: list,
        tolerance: float | npt.NDArray | None = None,
        smallest_tol: float = 1e-6,
        constraint_type: str = "equality",
        *,
        invert: bool = False,
    ):
        if tolerance is None:
            ord_mag = np.floor(np.log10(np.absolute(ref_harmonics))) - 3
            tolerance = [max(smallest_tol, 10**x) for x in ord_mag]
            np.nan_to_num(
                tolerance, nan=smallest_tol, posinf=smallest_tol, neginf=smallest_tol
            )
        elif is_num(tolerance):
            tolerance *= np.ones(len(ref_harmonics))
        elif len(tolerance) != len(ref_harmonics):
            raise ValueError(f"Tolerance vector not of length {len(ref_harmonics)}")

        self.constraint_type = constraint_type
        self.tolerance = tolerance

        self.target_harmonics = ref_harmonics
        self.max_degree = len(ref_harmonics)

        if invert and constraint_type == "equality":
            bluemira_warn(
                "Have used 'invert=True' while using 'equality' type for the"
                "Spherical Harmonic Constraint."
                "Please double check your constraint inputs."
            )

        self.invert = invert

        self.sh_coil_names = sh_coil_names
        self.r_t = r_t

        self._args = {
            "a_mat": None,
            "b_vec": None,
            "value": 0.0,
            "scale": 1e6,
        }

    @property
    def control_coil_names(self):
        """
        The names of the allowed control coils when using the
        SH approximation constraints.
        """
        return self.sh_coil_names

    def prepare(self, equilibrium: Equilibrium, *, I_not_dI=False, fixed_coils=False):
        """
        Prepare the constraint for use in an equilibrium optimisation problem.

        Raises
        ------
        ValueError
            Constraint requires fixed coils
        """
        if len(equilibrium.coilset.control) != len(self.sh_coil_names):
            bluemira_warn(
                "You are using too many control coils in your optimisation problem."
                "Please make sure you only use the coils allowed by your SH approx."
            )

        # Passive coils are a TODO in bluemira core
        if I_not_dI:
            equilibrium = _get_dummy_equilibrium(equilibrium)

        if not fixed_coils:
            raise ValueError("SphericalHarmonicConstraint requires fixed coils")

        self._args["a_mat"] = self.control_response(equilibrium.coilset)
        self._args["b_vec"] = self.target_harmonics - self.evaluate(equilibrium)
        if self.invert:
            self._args["a_mat"] *= -1
            self._args["b_vec"] *= -1

    def control_response(self, coilset: CoilSet) -> np.ndarray:
        """
        Calculate control response of a CoilSet to the constraint.

        Returns
        -------
        :
            The coil harmonic amplitude matrix
        """
        # SH coefficients from function of the current distribution outside of the sphere
        # containing the plasma, i.e., LCFS (r_lcfs)
        # N.B., cannot use coil located within r_lcfs as part of this method.
        return coil_harmonic_amplitude_matrix(
            coilset, self.max_degree, self.r_t, self.sh_coil_names
        )

    def evaluate(self, _eq: Equilibrium) -> npt.NDArray[np.float64]:
        """
        Calculate the value of the constraint in an Equilibrium.
        """  # noqa: DOC201
        return np.zeros(len(self.target_harmonics))

    def f_constraint(self) -> SphericalHarmonicConstraintFunction:
        """Constraint function."""  # noqa: DOC201
        f_constraint = SphericalHarmonicConstraintFunction(name=self.name, **self._args)
        f_constraint.constraint_type = self.constraint_type
        return f_constraint

    def plot(self, ax=None):
        """
        Plot the constraint onto an Axes.
        """
        if ax is None:
            _, ax = plt.subplots()

        ax.add_patch(patch.Circle((0, 0), self.r_t, ec="orange", fill=True, fc="orange"))


class ToroidalHarmonicConstraint(UpdateableConstraint):
    """
    Toroidal harmonic constraints for the desired core plasma
    of a conventional aspect ratio stokamak equilibria.

    Parameters
    ----------
    ref_harmonics:
        Initial harmonic amplitudes obtained from desired core plasma
        (Returned by toroidal_harmonic_approximation)
    th_params:
        ToroidalHarmonicsParams dataclass containing necessary parameters for use in TH
        approximation
    tolerance:
        Tolerance with which the constraint(s) will be met

    """

    def __init__(
        self,
        ref_harmonics_cos: npt.NDArray[np.float64],
        ref_harmonics_sin: npt.NDArray[np.float64],
        th_params: ToroidalHarmonicsParams,
        tolerance: float | None = None,
        constraint_type: str = "equality",
    ):
        self.constraint_type = constraint_type
        if isinstance(tolerance, float):
            tolerance *= np.ones(len(ref_harmonics_cos) + len(ref_harmonics_sin))
        else:
            tolerance = 1e-3 * np.append(ref_harmonics_cos, ref_harmonics_sin, axis=0)
            tolerance = np.abs(tolerance)
        self.tolerance = tolerance

        self.max_degree = len(ref_harmonics_cos)

        if constraint_type == "equality":
            self.tolerance = tolerance
            self.target_harmonics_cos = ref_harmonics_cos
            self.target_harmonics_sin = ref_harmonics_sin
        else:
            self.tolerance = np.append(tolerance, tolerance, axis=0)
            self.target_harmonics_cos = np.append(
                ref_harmonics_cos, ref_harmonics_cos, axis=0
            )
            self.target_harmonics_sin = np.append(
                ref_harmonics_sin, ref_harmonics_sin, axis=0
            )

        self.th_params = th_params

        self._args = {
            "a_mat_cos": None,
            "a_mat_sin": None,
            "b_vec_cos": None,
            "b_vec_sin": None,
            "value": 0.0,
            "scale": 1e6,
        }

    @property
    def control_coil_names(self):
        """
        The names of the allowed control coils when using the
        TH approximation constraints.
        """
        return self.th_params.th_coil_names

    def prepare(self, equilibrium: Equilibrium, *, I_not_dI=False, fixed_coils=True):
        """
        Prepare the constraint for use in an equilibrium optimisation problem.

        Raises
        ------
        ValueError
            Constraint requires fixed coils
        """
        if len(equilibrium.coilset.control) != len(self.th_params.th_coil_names):
            bluemira_warn(
                "You are using too many control coils in your optimisation problem."
                "Please make sure you only use the coils allowed by your TH approx."
            )

        # Passive coils are a TODO in bluemira core
        if I_not_dI:
            equilibrium = _get_dummy_equilibrium(equilibrium)

        if not fixed_coils:
            raise ValueError("ToroidalHarmonicConstraint requires fixed coils")

        self._args["a_mat_cos"], self._args["a_mat_sin"] = self.control_response(
            equilibrium.coilset
        )
        self._args["b_vec_cos"] = self.target_harmonics_cos - self.evaluate(equilibrium)
        self._args["b_vec_sin"] = self.target_harmonics_sin - self.evaluate(equilibrium)
        if self.constraint_type == "inequality":
            self._args["a_mat_cos"] = np.append(
                self._args["a_mat_cos"], -1 * self._args["a_mat_cos"], axis=0
            )
            self._args["a_mat_sin"] = np.append(
                self._args["a_mat_sin"], -1 * self._args["a_mat_sin"], axis=0
            )
            self._args["b_vec_cos"][2:] *= -1
            self._args["b_vec_sin"][2:] *= -1

    def control_response(self, coilset: CoilSet) -> np.ndarray:
        """
        Calculate control response of a CoilSet to the constraint.

        Returns
        -------
        :
            The coil harmonic amplitude matrix
        """
        # TH coefficients from function of the current distribution outside of the region
        # containing the plasma, i.e., LCFS
        # N.B., cannot use coil located within LCFS as part of this method.
        return coil_toroidal_harmonic_amplitude_matrix(
            coilset, self.th_params, max_degree=self.max_degree
        )

    def evaluate(self, _eq: Equilibrium) -> npt.NDArray[np.float64]:
        """
        Calculate the value of the constraint in an Equilibrium.
        """  # noqa: DOC201
        return np.zeros(len(self.target_harmonics_cos))

    def f_constraint(self) -> ToroidalHarmonicConstraintFunction:
        """Constraint function."""  # noqa: DOC201
        f_constraint = ToroidalHarmonicConstraintFunction(name=self.name, **self._args)
        f_constraint.constraint_type = self.constraint_type
        return f_constraint

    def plot(self, ax=None):
        """
        Plot the constraint onto an Axes.
        """
        if ax is None:
            _, ax = plt.subplots()
        max_R = np.max(self.th_params.R)  # noqa: N806
        min_R = np.min(self.th_params.R)  # noqa: N806
        max_Z = np.max(self.th_params.Z)  # noqa: N806
        min_Z = np.min(self.th_params.Z)  # noqa: N806
        centre_R = (max_R - min_R) / 2 + min_R  # noqa: N806
        centre_Z = (max_Z - np.abs(min_Z)) / 2  # noqa: N806
        radius = (max_R - min_R) / 2

        ax.add_patch(
            patch.Circle(
                (centre_R, centre_Z), radius, ec="orange", fill=True, fc="orange"
            )
        )
