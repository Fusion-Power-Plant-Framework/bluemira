# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2022 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
A collection of common 0-D plasma physics scaling laws.
"""

import numpy as np

from bluemira.base.look_and_feel import bluemira_warn


class PowerLawScaling:
    """
    Simple power law scaling object, of the form:

    \t:math:`c~\\pm~cerr \\times {a_{1}}^{n1\\pm err1}{a_{2}}^{n2\\pm err2}...`

    if constant_err is specified, or of the form:

    \t:math:`ce^{\\pm cexperr} \\times {a_{1}}^{n1\\pm err1}{a_{2}}^{n2\\pm err2}...`

    Parameters
    ----------
    constant: float
        The constant of the equation
    constant_err: float
        The error on the constant
    constant_exp_err: Union[float, None]
        The exponent error on the constant (cannot be specified with cerr)
    exponents: Union[np.array, List, None]
        The ordered list of exponents
    exp_errs: Union[np.array, List, None]
        The ordered list of errors of the exponents
    """  # noqa: W505

    def __init__(
        self,
        constant=1.0,
        constant_err=0.0,
        constant_exp_err=None,
        exponents=None,
        exp_errs=None,
    ):
        self.c = constant
        self.cerr = constant_err
        self.cexperr = constant_exp_err
        self.exponents = np.array(exponents)
        self.errors = np.array(exp_errs)

    def __call__(self, *args):
        """
        Call the PowerLawScaling object for a set of arguments.
        """
        if len(args) != len(self):
            raise ValueError(
                "Number of arguments should be the same as the "
                f"power law length. {len(args)} != {len(self)}"
            )
        return self.calculate(*args)

    def calculate(self, *args, exponents=None):
        """
        Call the PowerLawScaling object for a set of arguments.
        """
        if exponents is None:
            exponents = self.exponents
        return self.c * np.prod(np.power(args, exponents))

    def error(self, *args):
        """
        Calculate the error of the PowerLawScaling for a set of arguments.
        """
        if self.cexperr is None:
            c = [(self.c + self.cerr) / self.c, (self.c - self.cerr) / self.c]
        else:
            if self.cerr != 0:
                bluemira_warn("PowerLawScaling object overspecified, ignoring cerr.")
            c = [np.exp(self.cexperr), np.exp(-self.cexperr)]
        up = max(c) * self.calculate(*args, exponents=self.exponents + self.errors)
        down = min(c) * self.calculate(*args, exponents=self.exponents - self.errors)
        return [self.calculate(*args), min(down, up), max(down, up)]

    def __len__(self):
        """
        Get the length of the PowerLawScaling object.
        """
        return len(self.exponents)


def lambda_q(B_t: float, q_cyl: float, p_sol: float, R_0: float, error: bool = False):
    """
    Scrape-off layer power width scaling (Eich et al., 2011) [4]

    \t:math:`\\lambda_q=(0.73\\pm0.38)B_t^{-0.78\\pm0.25}q_{95}^{1.2\\pm0.27}P_{SOL}^{0.1\\pm0.11}R_{0}^{0.02\\pm0.2}`

    Parameters
    ----------
    B_t: float
        Toroidal field [T]
    q_cyl: float
        Cylindrical safety factor
    p_sol: float
        Power in the scrape-off layer [MW]
    R_0: float
        Major radius [m]
    method: str
        Scaling to use when calculating lambda_q
    error: bool
        Whether or not to report the value with +/- errors

    Returns
    -------
    lambda_q: float
        Scrape-off layer width at the outboard midplane [m]

    Notes
    -----
    [4] Eich et al., 2011
        <https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.107.215001>
    """  # noqa: W505
    law = PowerLawScaling(
        constant=0.73e-3,
        constant_err=0.38e-3,
        exponents=[-0.78, 1.2, 0.1, 0.02],
        exp_errs=[0.25, 0.27, 0.11, 0.20],
    )
    if error:
        return law.error(B_t, q_cyl, p_sol, R_0)
    else:
        return law(B_t, q_cyl, p_sol, R_0)
