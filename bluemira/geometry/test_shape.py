from matplotlib.pyplot import plot

from bluemira.display import plot_2d
from bluemira.geometry.parameterisations import BotDomeTaperedInnerCurvedPictureFrame
from bluemira.geometry.wire import BluemiraWire

p = BotDomeTaperedInnerCurvedPictureFrame()
shape = p.create_shape()
# wires = shape._wires

plot_2d(shape, plane="xz")
