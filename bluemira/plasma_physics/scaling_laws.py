# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
A collection of common 0-D plasma physics scaling laws.
"""

from collections.abc import Iterable

import numpy as np

from bluemira.base.constants import raw_uc


class PowerLawScaling:
    """
    Simple power law scaling object, of the form:

    \t:math:`c~\\pm~cerr \\times {a_{1}}^{n1\\pm err1}{a_{2}}^{n2\\pm err2}...`

    Parameters
    ----------
    constant: float
        The constant of the equation
    constant_err: float
        The error on the constant
    exponents: Iterable
        The ordered list of exponents
    exp_errs: Union[np.ndarray, List, None]
        The ordered list of errors of the exponents
    """

    def __init__(
        self,
        constant: float,
        constant_err: float,
        exponents: Iterable[float],
        exp_errs: np.ndarray | list | None = None,
    ):
        self.c = constant
        self.constant_err = constant_err
        self.exponents = np.array(exponents)
        if exp_errs is None:
            self.errors = None
        else:
            self.errors = np.array(exp_errs)

    def __call__(self, *args):
        """
        Call the PowerLawScaling object for a set of arguments.

        Returns
        -------
        :
            power law scaling

        Raises
        ------
        ValueError
            Number of arguments not equal to number of exponents
        """
        if len(args) != len(self):
            raise ValueError(
                "Number of arguments should be the same as the "
                f"power law length. {len(args)} != {len(self)}"
            )
        return self.calculate(*args)

    def calculate(self, *args, constant=None, exponents=None):
        """
        Call the PowerLawScaling object for a set of arguments.

        Returns
        -------
        :
            power law scaling
        """
        if constant is None:
            constant = self.c
        if exponents is None:
            exponents = self.exponents
        return constant * np.prod(np.power(args, exponents))

    def calculate_range(self, *args) -> tuple[float, float]:
        """
        Calculate the range of the PowerLawScaling within the specified errors for a set
        of arguments

        Returns
        -------
        min_value:
            Minimum value of the power law according to the specified errors
        max_value:
            Maximum value of the power law according to the specified errors

        Raises
        ------
        ValueError
            No constant error or error ranges
        """
        if self.constant_err == 0.0 and self.errors is None:
            raise ValueError(
                "No errors provided on PowerLawScaling, cannot calculate range."
            )

        constant_range = [self.c - self.constant_err, self.c + self.constant_err]

        min_terms = np.zeros(len(self))
        max_terms = np.zeros(len(self))
        for i, (arg, exp, err) in enumerate(
            zip(args, self.exponents, self.errors, strict=False)
        ):
            term_values = [arg ** (exp - err), arg ** (exp + err)]
            min_terms[i] = min(term_values)
            max_terms[i] = max(term_values)

        return min(constant_range) * np.prod(min_terms), max(constant_range) * np.prod(
            max_terms
        )

    def __len__(self) -> int:
        """
        Get the length of the PowerLawScaling object.

        Returns
        -------
        :
            length of PowerLawScaling object
        """
        return len(self.exponents)


def lambda_q(
    B_t: float, q_cyl: float, p_sol: float, R_0: float, *, error: bool = False
) -> float | tuple[float, float, float]:
    """
    Scrape-off layer power width scaling (Eich et al., 2011) [4]

    Parameters
    ----------
    B_t:
        Toroidal field [T]
    q_cyl:
        Cylindrical safety factor
    p_sol:
        Power in the scrape-off layer [W]
    R_0:
        Major radius [m]
    method:
        Scaling to use when calculating lambda_q
    error:
        Whether or not to report the value with +/- errors

    Returns
    -------
    value:
        Scrape-off layer width at the outboard midplane [m]
    min_value:
        (if error) Minimum value of the power law according to the specified errors
    max_value:
        (if error) Maximum value of the power law according to the specified errors

    Notes
    -----
    [4] :doi:`Eich et al., 2011 <10.1103/PhysRevLett.107.215001>`

    \t:math:`\\lambda_q=(0.73\\pm0.38)B_t^{-0.78\\pm0.25}q_{95}^{1.2\\pm0.27}P_{SOL}^{0.1\\pm0.11}R_{0}^{0.02\\pm0.2}`
    """
    law = PowerLawScaling(
        constant=0.73e-3,
        constant_err=0.38e-3,
        exponents=[-0.78, 1.2, 0.1, 0.02],
        exp_errs=[0.25, 0.27, 0.11, 0.20],
    )
    p_sol = raw_uc(p_sol, "W", "MW")
    value = law(B_t, q_cyl, p_sol, R_0)
    if error:
        min_value, max_value = law.calculate_range(B_t, q_cyl, p_sol, R_0)
        return value, min_value, max_value
    return value


def P_LH(  # noqa: N802
    n_e: float, B_t: float, A: float, R_0: float, *, error: bool = False
) -> float | tuple[float, float, float]:
    """
    Power requirement for accessing H-mode, Martin scaling [3]

    Parameters
    ----------
    n_e:
        Electron density [1/m^3]
    B_t:
        Toroidal field at the major radius [T]
    A:
        Plasma aspect ratio
    R_0:
        Plasma major radius [m]
    error:
        Whether or not to return error bar values

    Returns
    -------
    value:
        Power required to access H-mode [W]
    min_value:
        (if error) Minimum value of the power law according to the specified errors
    max_value:
        (if error) Maximum value of the power law according to the specified errors


    Notes
    -----
    [3] Martin et al., 2008, :doi:`10.1088/1742-6596/123/1/012033` equation (3)

    \t:math:`P_{LH}=2.15e^{\\pm 0.107}n_{e20}^{0.782 \\pm 0.037}`
    \t:math:`B_{T}^{0.772 \\pm 0.031}a^{0.975 \\pm 0.08}R_{0}^{0.999 \\pm 0.101}`
    """
    law = PowerLawScaling(
        constant=2.15e6,
        constant_err=0.0,
        exponents=[0, 0.782, 0.772, 0.975, 0.999],
        exp_errs=[0.107, 0.037, 0.031, 0.08, 0.101],
    )
    leading_term = np.exp(1)
    n_e20 = raw_uc(n_e, "1/m^3", "1e20/m^3")
    a = R_0 / A
    value = law(leading_term, n_e20, B_t, a, R_0)

    if error:
        min_value, max_value = law.calculate_range(leading_term, n_e20, B_t, a, R_0)
        return value, min_value, max_value
    return value


def IPB98y2(  # noqa: N802
    I_p: float,
    B_t: float,
    p_sep: float,
    n: float,
    mass: float,
    R_0: float,
    A: float,
    kappa: float,
) -> float:
    """
    ITER IPB98(y, 2) Confinement time scaling for ELMy H-mode [2]

    Parameters
    ----------
    I_p:
        Plasma current [A]
    B_t:
        Toroidal field at R_0 [T]
    p_sep:
        Separatrix power [W]  (a.k.a. loss power (corrected for charge exchange and
        orbit losses))
    n:
        Line average plasma density [1/m^3]
    mass:
        Average ion mass [a.m.u.]
    R_0:
        Major radius [m]
    A:
        Aspect ratio
    kappa:
        Plasma elongation

    Returns
    -------
    :
        Energy confinement time [s]

    Notes
    -----
    [2] :doi:`ITER Physics Expert Group, Nucl. Fus. 39, 12 <10.1088/0029-5515/39/12/302>`

    \t:math:`\\tau_{E}=0.0562I_p^{0.93}B_t^{0.15}P_{sep}^{-0.69}n^{0.41}M^{0.19}R_0^{1.97}A^{-0.58}\\kappa^{0.78}`
    """
    I_p = raw_uc(I_p, "A", "MA")
    p_sep = raw_uc(p_sep, "W", "MW")
    n = raw_uc(n, "1/m^3", "1e19/m^3")

    law = PowerLawScaling(
        constant=0.0562,
        constant_err=0.0,
        exponents=[0.93, 0.15, -0.69, 0.41, 0.19, 1.97, -0.58, 0.78],
    )
    return law(I_p, B_t, p_sep, n, mass, R_0, A, kappa)
