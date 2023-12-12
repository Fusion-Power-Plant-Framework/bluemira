import matplotlib.pyplot as plt
import numpy as np

from bluemira.geometry.coordinates import Coordinates
from bluemira.magnetostatics.polyhedral_prism import PolyhedralPrismCurrentSource

s = 0.5
d = 0.5 * np.sqrt(3)
x = np.array([2 * s, s, -s, -2 * s, -s, s])
y = np.zeros(6)
z = np.array([0, d, d, 0, -d, -d])

source = PolyhedralPrismCurrentSource(
    origin=[1, 1, 1],  # the centroid of the current source
    ds=[0, 0, 6],  # length of the source is determined by the norm of ds
    normal=[0, 1, 0],
    t_vec=[1, 0, 0],
    xs_coordinates=Coordinates(np.c_[x, y, z]),  # Points specified in x-z
    alpha=45.0,  # angle at the tip of the current source
    beta=45,  # angle at the tail of the current source (must be the same!)
    current=1e6,
)

x = np.linspace(0, 4, 100) - 1
y = np.linspace(0, 4, 100) - 1
xx, yy = np.meshgrid(x, y)

# Calculate field values in global x, y, z Cartesian coordinates.
Bx, By, Bz = source.field(xx, yy, np.ones_like(xx))
B = np.sqrt(Bx**2 + By**2 + Bz**2)

source.plot()
ax = plt.gca()
ax.contourf(xx, yy, B, zdir="z", offset=1)
