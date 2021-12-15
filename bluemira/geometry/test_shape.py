from matplotlib.pyplot import plot
from bluemira.geometry.parameterisations import SCCurvedPictureFrame
from bluemira.display import plot_2d
from bluemira.geometry.wire import BluemiraWire

p = SCCurvedPictureFrame()
shape = p.create_shape()
# wires = shape._wires

plot_2d(shape, plane="xz")
