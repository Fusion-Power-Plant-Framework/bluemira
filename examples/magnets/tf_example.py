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


class DummyInsulator(Material):
    def res(self, **kwargs):
        return 1e6

    def ym(self, **kwargs):
        return 1e7


# plot options
show = True
homogenized = True

# input values
Ri = 5  # [m] max external radius of the internal TF leg
Re = 12  # [m] max internal radius of the external TF part

R0 = 9
B0 = 6

n_TF = 16
I_TF = B0 * R0 / MU_0_2PI / n_TF  # total current in each TF coil
B_TF_i = MU_0_2PI * n_TF * I_TF / Ri  # max magnetic field on the inner TF leg

I0 = 1.0e4  # operational current in each conductor
n_cond = int(np.ceil(I_TF / I0))

T0 = 6.8
t_delay = 3
tau = 20
t0 = 0
tf = tau
hotspot_target_temperature = 250.0

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
pm = B_TF_i ** 2 / (2 * MU_0)
allowable_sigma = 667e6

print(f"before optimization: conductor dx_jacket = {conductor.dx_jacket}")
result_opt_jacket = optimize_jacket_conductor(
    conductor, pm, T0, B_TF_i, allowable_sigma, bounds=[1e-7, 1]
)
print(f"after optimization: conductor dx_jacket = {conductor.dx_jacket}")

# creation of case
wp1 = WindingPack(conductor, 1, 1)
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

t_z = 0.5 * np.log(case.Ri / Re) * MU_0_4PI * n_TF * I_TF ** 2
pm = B_TF_i ** 2 / (2 * MU_0)
allowable_sigma = 867e6

case.optimize_vault_radial_thickness(
    pm, t_z, T=T0, B=B_TF_i, allowable_sigma=allowable_sigma
)
if show:
    ax = case.plot(homogenized=homogenized)
    ax.set_aspect("equal")
    plt.title("After vault optimization")
    plt.show()
