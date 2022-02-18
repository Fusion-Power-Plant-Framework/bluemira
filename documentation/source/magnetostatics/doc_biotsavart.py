import matplotlib.pyplot as plt
import numpy as np

from bluemira.magnetostatics.biot_savart import BiotSavartFilament

n = 200
x = np.zeros(n)
y = np.linspace(0, 10, n)
z = np.zeros(n)

source = BiotSavartFilament(np.array([x, y, z]).T, radius=0.4)

x = np.linspace(-2, 2, 100)
z = np.linspace(-2, 2, 100)
xx, zz = np.meshgrid(x, z)

Bx, By, Bz = source.field(xx, 8 * np.ones_like(xx), zz)
B = np.sqrt(Bx**2 + By**2 + Bz**2)

source.plot()
ax = plt.gca()
ax.contourf(xx, B, zz, zdir="y", offset=8, cmap="magma")
