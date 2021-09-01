import numpy as np
import matplotlib.pyplot as plt
from bluemira.magnetostatics.circuits import ArbitraryPlanarRectangularXSCircuit

x = np.array([0, 1, 3, 4, 4, 3, 1, 0, 0])
y = np.zeros(len(x))
z = np.array([1, 0, 0, 1, 3, 4, 4, 3, 1])

source = ArbitraryPlanarRectangularXSCircuit(
    shape=np.c_[x, y, z], breadth=0.5, depth=0.25, current=1e6
)

x = np.linspace(-1, 5, 100)
z = np.linspace(-1, 5, 100)
xx, zz = np.meshgrid(x, z)

Bx, By, Bz = source.field(xx, np.zeros_like(xx), zz)
B = np.sqrt(Bx ** 2 + By ** 2 + Bz ** 2)
source.plot()
f, ax = plt.subplots()
cm = ax.contourf(xx, zz, B, cmap="magma")
f.colorbar(cm, label="$B$ [T]")
ax.set_aspect("equal")
ax.set_xlabel("$x$")
ax.set_ylabel("$z$")
