# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

from bluemira.utilities.plot_tools import gsymbolify, str_to_latex


class TestStrToLatex:
    def test_single(self):
        result = str_to_latex("PF_1")
        assert result == "$PF_{1}$"

        result = str_to_latex("I_m_p")
        assert result == "$I_{m_{p}}$"


class TestGsymbolify:
    def test_lowercase(self):
        string = gsymbolify("beta")
        assert string == "\\beta"

    def test_uppercase(self):
        string = gsymbolify("Beta")
        assert string == "\\Beta"

    def test_nothing(self):
        string = gsymbolify("nothing")
        assert string == "nothing"
