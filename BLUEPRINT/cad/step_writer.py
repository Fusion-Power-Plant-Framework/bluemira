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
Writer for STP files.
"""

from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.Interface import Interface_Static_SetCVal
from OCC.Core.Quantity import Quantity_Color, Quantity_NOC_RED
from OCC.Core.STEPCAFControl import STEPCAFControl_Writer
from OCC.Core.STEPControl import STEPControl_AsIs
from OCC.Core.TCollection import TCollection_ExtendedString
from OCC.Core.TDF import TDF_LabelSequence
from OCC.Core.TDocStd import TDocStd_Document
from OCC.Core.XCAFDoc import XCAFDoc_ColorGen, XCAFDoc_DocumentTool
from OCC.Core.XSControl import XSControl_WorkSession

from BLUEPRINT.base.error import CADError


class StepWriter(object):
    """
    STEP writer that support layers & colours

    Notes
    -----
    Based on (the seemingly defunct) aocxchange package.
    """

    def __init__(self, filename, layer_name="layer-00", partname=None):
        self.filename = filename
        self.partname = partname

        self.doc = TDocStd_Document(TCollection_ExtendedString("MDTV-CAF"))

        self.shape_tool = XCAFDoc_DocumentTool().ShapeTool(self.doc.Main())
        self.colours = XCAFDoc_DocumentTool().ColorTool(self.doc.Main())
        self.layers = XCAFDoc_DocumentTool().LayerTool(self.doc.Main())
        _ = TDF_LabelSequence()
        _ = TDF_LabelSequence()

        self.top_label = self.shape_tool.NewShape()

        self.current_colour = Quantity_Color(Quantity_NOC_RED)
        self.current_layer = self.layers.AddLayer(TCollection_ExtendedString(layer_name))
        self.layer_names = {}

    def set_colour(self, r=1, g=1, b=1, colour=None):
        """
        Set the colour.

        Parameters
        ----------
        r: float
        g: float
        b: float
        colour: Quantity_Color
        """
        if colour is not None:
            self.current_colour = colour
        else:
            clr = Quantity_Color(r, g, b, 0)
            self.current_colour = clr

    def set_layer(self, layer_name):
        """
        Set the current layer name.

        If the layer has already been set before, that TDF_Label will be used.

        Parameters
        ----------
        layer_name: str
            The name of the layer.
        """
        if layer_name in self.layer_names:
            self.current_layer = self.layer_names[layer_name]
        else:
            self.current_layer = self.layers.AddLayer(
                TCollection_ExtendedString(layer_name)
            )
            self.layer_names[layer_name] = self.current_layer

    def add_shape(self, shape, colour=None, layer=None):
        """
        Add a shape to export.

        A layer and colour can be specified.

        Parameters
        ----------
        shape : TopoDS_Shape
            the TopoDS_Shape to export
        colour :
            can be a tuple: (r,g,b) or a Quantity_Color instance
        layer : str
            layer name

        Notes
        -----
        The set colours and layers will be used for any further objects that are added.
        """
        shp_label = self.shape_tool.AddShape(shape)

        if colour is None:
            self.colours.SetColor(shp_label, self.current_colour, XCAFDoc_ColorGen)
        else:
            if isinstance(colour, Quantity_Color):
                self.current_colour = colour
            else:
                if len(colour) != 3:
                    msg = "expected a tuple with three values < 1."
                    raise ValueError(msg)
                r, g, b = colour
                self.set_colour(r, g, b)
            self.colours.SetColor(shp_label, self.current_colour, XCAFDoc_ColorGen)

        if layer is None:
            self.layers.SetLayer(shp_label, self.current_layer)
        else:
            self.set_layer(layer)
            self.layers.SetLayer(shp_label, self.current_layer)

    def write_file(self):
        """
        Write the file.
        """
        work_session = XSControl_WorkSession()
        writer = STEPCAFControl_Writer(work_session, False)

        if self.partname is not None:
            Interface_Static_SetCVal("write.step.product.name", self.partname)

        transfer_status = writer.Transfer(self.doc, STEPControl_AsIs)
        if transfer_status != IFSelect_RetDone:
            msg = "An error occurred while transferring a shape to the STEP writer"
            raise CADError(msg)

        write_status = writer.Write(self.filename)
        if write_status != IFSelect_RetDone:
            msg = "An error occurred while writing the STEP file"
            raise CADError(msg)
