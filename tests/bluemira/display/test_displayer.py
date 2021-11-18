# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
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
Tests for the displayer module.
"""

import pytest
from unittest.mock import patch

import contextlib
import numpy as np

from bluemira.base.components import (
    GroupingComponent,
    PhysicalComponent,
)
from bluemira.geometry.tools import make_polygon, extrude_shape

from bluemira.display import displayer
from bluemira.display.error import DisplayError


import tests
from tests.bluemira.display.helpers import PatchQApp, PatchQuarterWidget


class TestDisplayCADOptions:
    def test_default_options(self):
        """
        Check the values of the default options correspond to the global defaults.
        """
        the_options = displayer.DisplayCADOptions()
        assert the_options.as_dict() == displayer.DEFAULT_DISPLAY_OPTIONS
        assert the_options.as_dict() is not displayer.DEFAULT_DISPLAY_OPTIONS

    def test_options(self):
        """
        Check the values can be set by passing kwargs.
        """
        the_options = displayer.DisplayCADOptions(color=(1.0, 0.0, 0.0))
        options_dict = the_options.as_dict()
        for key, val in options_dict.items():
            if key == "color":
                assert val != displayer.DEFAULT_DISPLAY_OPTIONS[key]
            else:
                assert val == displayer.DEFAULT_DISPLAY_OPTIONS[key]

    def test_modify_options(self):
        """
        Check the display options can be modified by passing in keyword args.
        """
        the_options = displayer.DisplayCADOptions()
        the_options.modify(color=(1.0, 0.0, 0.0))
        options_dict = the_options.as_dict()
        for key, val in options_dict.items():
            if key == "color":
                assert val != displayer.DEFAULT_DISPLAY_OPTIONS[key]
            else:
                assert val == displayer.DEFAULT_DISPLAY_OPTIONS[key]

    def test_properties(self):
        """
        Check the display option properties can be accessed
        """
        the_options = displayer.DisplayCADOptions()
        for key, val in displayer.DEFAULT_DISPLAY_OPTIONS.items():
            assert getattr(the_options, key) == val

        the_options.color = (1.0, 0.0, 0.0)
        assert the_options.color != displayer.DEFAULT_DISPLAY_OPTIONS["color"]
        assert (
            the_options.transparency == displayer.DEFAULT_DISPLAY_OPTIONS["transparency"]
        )

        the_options.transparency = 1.0
        assert the_options.color != displayer.DEFAULT_DISPLAY_OPTIONS["color"]
        assert (
            the_options.transparency != displayer.DEFAULT_DISPLAY_OPTIONS["transparency"]
        )


class TestComponentDisplayer:
    def test_display(self):
        square_points = np.array(
            [
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (1.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
            ]
        )

        wire1 = make_polygon(square_points, closed=True)
        wire2 = make_polygon(square_points + 1.0, closed=True)

        group = GroupingComponent("Parent")
        child1 = PhysicalComponent(
            "Child1",
            shape=wire1,
            parent=group,
        )
        child1.display_cad_options.color = (0.0, 1.0, 0.0)
        child2 = PhysicalComponent("Child2", shape=wire2, parent=group)

        with contextlib.nullcontext() if tests.PLOTTING else patch(
            "bluemira.geometry._freecadapi.QApplication", PatchQApp
        ):
            with contextlib.nullcontext() if tests.PLOTTING else patch(
                "bluemira.geometry._freecadapi.quarter.QuarterWidget", PatchQuarterWidget
            ):
                child1.show_cad()
                group.show_cad()
                child2.display_cad_options = displayer.DisplayCADOptions(
                    color=(1.0, 0.0, 0.0)
                )
                group.show_cad()
                group.show_cad(color=(0.0, 0.0, 1.0))
                displayer.ComponentDisplayer().show_cad(group, color=(1.0, 0.0, 0.0))

        with pytest.raises(DisplayError):
            child2.display_cad_options = (0.0, 0.0, 1.0)


class TestGeometryDisplayer:
    square_points = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
    ]

    def test_display(self):
        wire1 = make_polygon(self.square_points, label="wire1", closed=False)
        box1 = extrude_shape(wire1, vec=(0.0, 0.0, 1.0), label="box1")

        with contextlib.nullcontext() if tests.PLOTTING else patch(
            "bluemira.geometry._freecadapi.QApplication", PatchQApp
        ):
            with contextlib.nullcontext() if tests.PLOTTING else patch(
                "bluemira.geometry._freecadapi.quarter.QuarterWidget", PatchQuarterWidget
            ):
                displayer.show_cad(wire1)
                displayer.show_cad(
                    box1, displayer.DisplayCADOptions(color=(1.0, 0.0, 1.0))
                )
                displayer.show_cad([wire1, box1])
                displayer.show_cad(
                    [wire1, box1], displayer.DisplayCADOptions(color=(1.0, 0.0, 1.0))
                )
                displayer.show_cad(
                    [wire1, box1],
                    [
                        displayer.DisplayCADOptions(color=(1.0, 0.0, 0.0)),
                        displayer.DisplayCADOptions(
                            color=(0.0, 1.0, 0.0), transparency=0.2
                        ),
                    ],
                )
                displayer.show_cad(
                    [wire1, box1], color=(1.0, 0.0, 0.0), transparency=0.2
                )

                with pytest.raises(DisplayError):
                    displayer.show_cad(
                        wire1,
                        [
                            displayer.DisplayCADOptions(color=(1.0, 0.0, 0.0)),
                            displayer.DisplayCADOptions(color=(0.0, 1.0, 0.0)),
                        ],
                    )
