from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

from PySide2.QtWidgets import QApplication

from pivy import coin
from pivy import quarter

import freecad  # noqa: F401
import FreeCAD
import FreeCADGui
import Part

import bluemira.geometry._freecadapi as freecadapi
from bluemira.geometry.error import GeometryError


@dataclass(frozen=True)
class DisplayOptions:
    rgb: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    shininess: float = 0.2
    transparency: float = 0.0


def _colourise(
    node: coin.SoNode,
    options: DisplayOptions,
):
    if isinstance(node, coin.SoMaterial):
        node.ambientColor.setValue(coin.SbColor(*options.rgb))
        node.diffuseColor.setValue(coin.SbColor(*options.rgb))
        node.specularColor.setValue(coin.SbColor(*options.rgb))
        node.shininess.setValue(options.shininess)
        node.transparency.setValue(options.transparency)
    for child in node.getChildren() or []:
        _colourise(child, options)


def display(
    parts: Union[Part.Shape, List[Part.Shape]],
    options: Optional[Union[DisplayOptions, List[DisplayOptions]]] = None,
):
    if not isinstance(parts, list):
        parts = [parts]

    if options is None:
        options = [DisplayOptions()] * len(parts)
    elif not isinstance(options, list):
        options = [options] * len(parts)

    if len(options) != len(parts):
        raise GeometryError(
            "If options for display are provided then there must be as many options as "
            "there are parts to display."
        )

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    if not hasattr(FreeCADGui, "subgraphFromObject"):
        FreeCADGui.setupWithoutGUI()

    doc = FreeCAD.newDocument()

    root = coin.SoSeparator()

    for part, option in zip(parts, options):
        obj = doc.addObject("Part::Feature")
        obj.Shape = part
        doc.recompute()
        subgraph = FreeCADGui.subgraphFromObject(obj)
        _colourise(subgraph, option)
        root.addChild(subgraph)

    viewer = quarter.QuarterWidget()
    viewer.setBackgroundColor(coin.SbColor(1, 1, 1))
    viewer.setTransparencyType(coin.SoGLRenderAction.SCREEN_DOOR)
    viewer.setSceneGraph(root)

    viewer.setWindowTitle("Bluemira Display")
    viewer.show()
    app.exec_()


if __name__ == "__main__":
    box = Part.makeBox(1.0, 1.0, 1.0)
    box_options = DisplayOptions(rgb=(1.0, 0.0, 0.0))
    sphere = Part.makeSphere(1.0)
    sphere_options = DisplayOptions(rgb=(0.0, 1.0, 0.0), transparency=0.5)
    display([box, sphere], [box_options, sphere_options])
    square_points = [
        (0.0, 0.0, 0.0),
        (2.0, 0.0, 0.0),
        (2.0, 2.0, 0.0),
        (0.0, 2.0, 0.0),
    ]
    open_wire: Part.Wire = freecadapi.make_polygon(square_points)
    face = Part.Face([open_wire])
    face_options = DisplayOptions(rgb=(0.0, 0.0, 1.0), transparency=0.2)
    display(face, face_options)
