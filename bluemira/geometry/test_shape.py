from matplotlib.pyplot import plot

from bluemira.display import plot_2d
from bluemira.geometry.parameterisations import (
    ResistiveCurvedPictureFrame,
    SCCurvedPictureFrame,
)
from bluemira.geometry.wire import BluemiraWire

p = ResistiveCurvedPictureFrame()
shape = p.create_shape()
# wires = shape._wires

plot_2d(shape, plane="xz")
