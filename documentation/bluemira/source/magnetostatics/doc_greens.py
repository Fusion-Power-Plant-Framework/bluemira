import numpy as np
from bluemira.magnetostatics.greens import greens_psi, greens_Bx, greens_Bz

coil_x, coil_z = 4, 5
x = np.linspace(0.1, 10, 100)
z = np.linspace(0, 10, 100)
xx, zz = np.meshgrid(x, z)

psi = greens_psi(coil_x, coil_z, xx, zz)
Bx = greens_Bx(coil_x, coil_z, xx, zz)
Bz = greens_Bz(coil_x, coil_z, xx, zz)
