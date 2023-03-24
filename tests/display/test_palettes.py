# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
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
Tests for palettes module
"""
import matplotlib.colors as colors
import pytest

from bluemira.display.palettes import (
    ColorPalette,
    background_colour_string,
    make_alpha_palette,
    make_rgb_alpha,
)


class TestColorPalette:
    def setup_method(self):
        self.pal = ColorPalette({"C1": "#000000", "C2": "#ffffff"})

    def test_keys(self):
        assert self.pal.keys() == set(("C1", "C2"))

    def test_next(self):
        assert next(self.pal) == "#000000"
        assert next(self.pal) == "#ffffff"
        assert next(self.pal) == "#000000"

    def test_set_get_item(self):
        self.pal[0] = "#aaaaaa"
        assert colors.to_hex(self.pal["C1"][0]) == "#aaaaaa"

        self.pal["C1"] = "#000000"
        assert colors.to_hex(self.pal["C1"][0]) == "#000000"

    def test_repr_term(self):
        assert (
            self.pal._repr_colour_str()
            == "\x1b[48:2::0:0:0m \x1b[49m\x1b[48:2::255:255:255m \x1b[49m"
        )

    def test_repr_html(self):
        assert self.pal._repr_html() == (
            '<svg  width="110" height="55">'
            '<rect x="0" y="0" width="55" height="55"'
            ' style="fill:#000000;stroke-width:2;stroke:rgb(255,255,255)"/>'
            '<rect x="55" y="0" width="55" height="55"'
            ' style="fill:#ffffff;stroke-width:2;stroke:rgb(255,255,255)"/>'
            "</svg>"
        )


def test_background_colour_string():
    assert background_colour_string("#123456") == "\x1b[48:2::18:52:86m \x1b[49m"
    assert background_colour_string("#123") == "\x1b[48:2::1:2:3m \x1b[49m"


def test_make_rgb_alpha():
    assert make_rgb_alpha((1, 2, 3), 0.5) == (1.0, 1.5, 2.0)
    assert make_rgb_alpha((1, 2, 3), 0.5, (0.5, 0.5, 0.5)) == (0.75, 1.25, 1.75)


@pytest.mark.parametrize("pal", [ColorPalette({"C1": "#000000"}), "#000000"])
def test_make_alpha_palette(pal):
    new_pal = make_alpha_palette(pal, 3)
    assert [colors.to_hex(p) for p in new_pal._palette] == [
        "#000000",
        "#555555",
        "#aaaaaa",
    ]
    new_pal = make_alpha_palette(pal, 3, "red")
    assert [colors.to_hex(p) for p in new_pal._palette] == [
        "#000000",
        "#550000",
        "#aa0000",
    ]
