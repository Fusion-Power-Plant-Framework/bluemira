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
CAD display utilities
"""
import numpy as np
from OCC.Display.SimpleGui import init_display

from bluemira.base.look_and_feel import bluemira_warn

try:
    from OCC.Core.Quantity import Quantity_Color, Quantity_TOC_RGB
except ImportError:
    from OCC.Quantity import Quantity_Color, Quantity_TOC_RGB


class QtDisplayer:
    """
    Soft wrapper around simple OCC GUI / Qt viewer

    Parameters
    ----------
    wireframe: bool (default=False)
        Enable wireframe view mode
    """

    def __init__(self, wireframe=False):

        self.qt_display, self.start_qt_display, menu, f_menu = init_display()
        self.menu, self.func_menu = menu, f_menu
        # Set white background
        try:
            self.qt_display.set_bg_gradient_color([255, 255, 255], [255, 255, 255])
        except TypeError:
            self.qt_display.set_bg_gradient_color(255, 255, 255, 255, 255, 255)
        if wireframe:
            self.qt_display.SetModeWireFrame()

    def show(self):
        """
        Displays qt window
        """
        self.qt_display.FitAll()
        self.start_qt_display()

    def add_shape(self, shape, color=None, transparency=0):
        """
        Adds a shape to the QtDisplayer

        Parameters
        ----------
        shape: OCC shape object
            The shape to be displayed
        color: tuple(3)
            The RGB color to display the shape with
        transparency: float (0==>1)
            The transparency to display the shape with. NOTE: OCC QtDisplayer
            has reverse definition of transparency, i.e.:
            0: not transparent
            1: fully transparent (although not in practice)
        """
        if color is None:  # used for simple viewer
            color = np.random.rand(3)
        if len(color) != 3:
            color = color[:3]
            bluemira_warn("RGB color tuples being violated somehow.")
        if transparency != 0:
            try:
                qc = Quantity_Color(*color, Quantity_TOC_RGB)
                # Removes drawn black boundaries for transparent part
                self.qt_display.default_drawer.SetFaceBoundaryDraw(False)
                self.qt_display.DisplayShape(shape, color=qc, transparency=transparency)
                # Reset black boundaries for future shapes
                self.qt_display.default_drawer.SetFaceBoundaryDraw(True)
            except AttributeError:
                # default_drawer doesn't exist in this version of OCC/OCE
                qc = Quantity_Color(*color, Quantity_TOC_RGB)
                qc.ChangeIntensity(-50)
                self.qt_display.DisplayColoredShape(shape, qc)
        else:
            qc = Quantity_Color(*color, Quantity_TOC_RGB)
            qc.ChangeIntensity(-50)
            self.qt_display.DisplayColoredShape(shape, qc)
