from bluemira.magnetostatics.trapezoidal_prism import TrapezoidalPrismCurrentSource
from bluemira.magnetostatics.polyhedral_prism import PolyhedralPrismCurrentSource

import matplotlib.pyplot as plt
import numpy as np


# expected result
source = TrapezoidalPrismCurrentSource(
    np.array([0, 0, 0]),
    np.array([0, 0, 4]),
    np.array([1, 0, 0]),
    np.array([0, 1, 0]),
    np.sqrt(2) / 2,
    np.sqrt(2) / 2,
    0,
    0,
    1e6,
)
source.rotate(45, axis="z")
# polyhedral result (for same shape)
source2 = PolyhedralPrismCurrentSource(
    np.array([0, 0, 0]),
    np.array([0, 1, 0]),
    np.array([0, 0, 2]),
    np.array([1, 0, 0]),
    np.array([1, 1, 0]),
    4,
    2,
    1,
    0,
    0,
    1e6,
)

source.plot()
source2.plot()
plt.show()

source.plot()
n = 100
x = np.linspace(-2, 2, n)
y = np.linspace(-2, 2, n)
xx, yy = np.meshgrid(x, y)
zz = np.zeros_like(xx)

Bx, By, Bz = source.field(xx, yy, zz)
B = np.sqrt(Bx**2 + By**2 + Bz**2)

ax = plt.gca()
cm = ax.contourf(xx, yy, B, zdir="z", offset=0)
plt.colorbar(cm, shrink=0.46)
plt.title("Trapezoidal result")
plt.show()

source2.plot()
xx, yy = np.meshgrid(x, y)
zz = np.zeros_like(xx)

Bx, By, Bz = source2.field(xx, yy, zz)
B = np.sqrt(Bx**2 + By**2 + Bz**2)

ax = plt.gca()
cm = ax.contourf(xx, yy, B, zdir="z", offset=0)
plt.colorbar(cm, shrink=0.46)
plt.title("Polyhedral result")
plt.show()
