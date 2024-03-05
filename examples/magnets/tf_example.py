import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.constants import MU_0, MU_0_2PI, MU_0_4PI
from bluemira.magnets.cable import DummySquareCable
from bluemira.magnets.case import CaseTF
from bluemira.magnets.conductor import SquareConductor, optimize_jacket_conductor
from bluemira.magnets.materials import Copper300, Material
from bluemira.magnets.strand import Strand, Wire_NbTi
from bluemira.magnets.utils import (
    delayed_exp_func,
)
from bluemira.magnets.winding_pack import WindingPack

"""
Example for TF coils internal structure optimization.

Note: in this example the conductor operational current is given as input (no check
or optimization that takes into account the conductor critical current). 
"""

class DummyInsulator(Material):
    def res(self, **kwargs):
        return 1e6

    def ym(self, **kwargs):
        return 1e7


# plot options
show = True
homogenized = False  # if True plot the WP as a homogenized block

# input values
Ri = 5  # [m] max external radius of the internal TF leg
Re = 12  # [m] max internal radius of the external TF part

R0 = 9  # [m] major machine radius
B0 = 6  # [T] magnetic field @R0

n_TF = 16  # number of TF coils
I_TF = B0 * R0 / MU_0_2PI / n_TF  # total current in each TF coil
B_TF_i = MU_0_2PI * n_TF * I_TF / Ri  # max magnetic field on the inner TF leg
pm = B_TF_i ** 2 / (2 * MU_0)  # magnetic pressure on the inner TF leg
# vertical tension acting on the equatorial section of inner TF leg
t_z = 0.5 * np.log(Ri / Re) * MU_0_4PI * n_TF * I_TF ** 2

I0 = 1.0e4  # operational current in each conductor
n_cond = int(np.ceil(I_TF / I0))  # number of necessary conductors

T0 = 6.8
t_delay = 3
tau = 20
t0 = 0
tf = tau
hotspot_target_temperature = 250.0

# allowable stress values
allowable_sigma_jacket = 667e6  # [Pa] for the conductor jacket
allowable_sigma_case = 867e6  # [Pa] for the case

# Current and magnetic field behaviour
I = delayed_exp_func(I0, tau, t_delay)
B = delayed_exp_func(B_TF_i, tau, t_delay)

# define the conductor (with a dummy number of stabilizer strands )
sc_strand = Wire_NbTi()
copper300 = Copper300()
insulator = DummyInsulator()
stab_strand = Strand([copper300], [1])
cable = DummySquareCable(sc_strand, stab_strand, 100, 100, 0.005)
# c.plot(0,0, show=True)

conductor = SquareConductor(
    cable=cable, mat_jacket=copper300, mat_ins=insulator, dx_jacket=0.01, dx_ins=5e-3
)

# optimize the number of stabilizer strands using the hot spot criteria
result = cable.optimize_n_stab_ths(
    t0, tf, T0, hotspot_target_temperature, B, I, bounds=[1, 10000], show=show
)

# optimize the cable jacket thickness considering 0D stress model for the single cable
print(f"before optimization: conductor dx_jacket = {conductor.dx_jacket}")
result_opt_jacket = optimize_jacket_conductor(
    conductor, pm, T0, B_TF_i, allowable_sigma_jacket, bounds=[1e-7, 1]
)
print(f"after optimization: conductor dx_jacket = {conductor.dx_jacket}")

# creation of case
wp1 = WindingPack(conductor, 1, 1)  # just a dummy WP to create the case
case = CaseTF(
    Ri=5, dy_ps=0.1, dy_vault=0.6, theta_TF=360 / n_TF, mat_case=copper300, WPs=[wp1]
)

# if show:
#     ax = case.plot(homogenized=False)
#     ax.set_aspect("equal")
#     plt.show()

case.rearrange_conductors_in_wp_type1(n_cond, conductor, case.R_wp_i[0], 0.2, 0.075, 4)
if show:
    ax = case.plot(homogenized=homogenized)
    ax.set_aspect("equal")
    plt.title("Before vault optimization")
    plt.show()


case.optimize_vault_radial_thickness(
    pm, t_z, T=T0, B=B_TF_i, allowable_sigma=allowable_sigma_case, bounds=[1e-2, 1]
)
if show:
    ax = case.plot(homogenized=homogenized)
    ax.set_aspect("equal")
    plt.title("After vault optimization")
    plt.show()
