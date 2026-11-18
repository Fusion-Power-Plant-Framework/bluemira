import matplotlib.pyplot as plt
import numpy as np

from bluemira.magnetostatics.biot_savart import BiotSavartFilament

n = 2
x = np.zeros(n)
y = np.linspace(0, 5, n)
z = np.zeros(n)

source = BiotSavartFilament(np.array([x, y, z]).T, radius=0.4)

x = np.linspace(0, 2, 100)
z = np.linspace(-2, 2, 100)
xx, zz = np.meshgrid(x, z)

Bx, By, Bz = source.field(xx, 4 * np.ones_like(xx), zz)
B = np.sqrt(Bx**2 + By**2 + Bz**2)

y = np.linspace(-1, 5, 100)
yy, zz2 = np.meshgrid(y, z)
Bx2, By2, Bz2 = source.field(np.zeros_like(yy), yy, zz2)
B2 = np.sqrt(Bx2**2 + By2**2 + Bz2**2)

levels = np.linspace(0, max(B.max(), B2.max()), 21)

source.plot()
ax = plt.gca()
ax.contourf(B2, yy, zz2, zdir="x", offset=0, levels=levels, cmap="magma")
ax.contourf(xx, B, zz, zdir="y", offset=4, levels=levels, cmap="magma")

limits = [(x.min(), x.max()), (y.min(), y.max()), (z.min(), z.max())]
ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2])
ax.set_box_aspect([hi - lo for lo, hi in limits])
ax.set_xticks([0, 1, 2])

ax.computed_zorder = False
