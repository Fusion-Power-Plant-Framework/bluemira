import matplotlib.pyplot as plt
import numpy as np

from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.tools import make_circle
from bluemira.magnetostatics.circuits import ArbitraryPlanarPolyhedralXSCircuit

ring = make_circle(
    radius=4, center=[0, 0, 0], start_angle=0, end_angle=360, axis=[0, 1, 0]
)
xs = Coordinates({"x": [-1, 1, 1, -1], "z": [0, 1, -1, 0]})
xs.translate(xs.center_of_mass)
source = ArbitraryPlanarPolyhedralXSCircuit(ring.discretize(ndiscr=9), xs, current=1e6)

x = np.linspace(0, 6, 100)
z = np.linspace(-6, 6, 100)
xx, zz = np.meshgrid(x, z)

Bx, By, Bz = source.field(xx, np.zeros_like(xx), zz)
B = np.sqrt(Bx**2 + By**2 + Bz**2)
f = plt.figure()
ax = f.add_subplot(1, 1, 1, projection="3d")
source.plot(ax)
cm = ax.contourf(xx, B, zz, cmap="magma", zdir="y", offset=0)
f.colorbar(cm, label="$B$ [T]")
