# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I. Maione, S. McIntosh, J. Morris,
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
A collection of plotting tools.
"""

from bluemira.base.constants import GREEK_ALPHABET, GREEK_ALPHABET_CAPS


__all__ = ["str_to_latex"]


def gsymbolify(string):
    """
    Convert a string to a LaTEX printable greek letter if detected.

    Parameters
    ----------
    string: str
        The string to add Greek symbols to

    Returns
    -------
    string: str
        The modified string. Returns input if no changes made
    """
    if string in GREEK_ALPHABET or string in GREEK_ALPHABET_CAPS:
        return "\\" + string
    else:
        return string


def str_to_latex(string):
    """
    Create a new string which can be printed in LaTEX nicely.

    Parameters
    ----------
    string: str
        The string to be converted

    Returns
    -------
    string: str
        The mathified string

    'I_m_p' ==> '$I_{m_{p}}$'
    """
    s = string.split("_")
    s = [gsymbolify(sec) for sec in s]
    ss = "".join(["_" + "{" + lab for i, lab in enumerate(s[1:])])
    return "$" + s[0] + ss + "}" * (len(s) - 1) + "$"
