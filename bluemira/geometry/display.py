from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import bluemira.geometry._freecadapi as freecadapi
from bluemira.geometry.error import GeometryError


@dataclass(frozen=True)
class DisplayOptions:
    rgb: Tuple[float, float, float] = (0.5, 0.5, 0.5)
    transparency: float = 0.0


class Displayer:
    def __init__(self, parts, options, api="bluemira.geometry._freecadapi"):
        self.parts = parts
        self.options = options
        self.display_func = get_module(api).display

    def display(self):
        self.display_func(self.parts, self.options)


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
