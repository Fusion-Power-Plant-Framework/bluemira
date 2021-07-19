import numpy as np
from bluemira.magnetostatics.semianalytic_2d import (
    semianalytic_psi,
    semianalytic_Bx,
    semianalytic_Bz,
)

coil_x, coil_z = 4, 5
coil_dx, coil_dz = 1, 2
x = np.linspace(0.1, 10, 100)
z = np.linspace(0, 10, 100)
xx, zz = np.meshgrid(x, z)

psi = semianalytic_psi(coil_x, coil_z, xx, zz, coil_dx, coil_dz)
Bx = semianalytic_Bx(coil_x, coil_z, xx, zz, coil_dx, coil_dz)
Bz = semianalytic_Bz(coil_x, coil_z, xx, zz, coil_dx, coil_dz)
