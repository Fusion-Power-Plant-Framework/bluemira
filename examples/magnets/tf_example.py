import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.constants import MU_0, MU_0_2PI, MU_0_4PI
from bluemira.magnets.cable import DummySquareCableLTS, DummySquareCableHTS
from bluemira.magnets.case import CaseTF
from bluemira.magnets.conductor import SquareConductor, optimize_jacket_conductor
from bluemira.magnets.materials import Copper300, Material, AISI_316LN
from bluemira.magnets.strand import Strand, Wire_Nb3Sn
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
        return 12e9


# plot options
show = True
homogenized = False  # if True plot the WP as a homogenized block

# input values
Ri = 4.0286  # [m] max external radius of the internal TF leg
Re = 16.075  # [m] max internal radius of the external TF part

A = 2.8
R0 = 8.6  # [m] major machine radius
B0 = 4.39  # [T] magnetic field @R0

n_TF = 16  # number of TF coils
I_TF = B0 * R0 / MU_0_2PI / n_TF  # total current in each TF coil
B_TF_i = 1.08 * (MU_0_2PI * n_TF * I_TF / Ri)  # max magnetic field on the inner TF leg
pm = B_TF_i ** 2 / (2 * MU_0)  # magnetic pressure on the inner TF leg
# vertical tension acting on the equatorial section of inner TF leg
t_z = 0.5 * np.log(Re / Ri) * MU_0_4PI * n_TF * I_TF ** 2

I0 = 70.0e3  # operational current in each conductor
n_cond = int(np.ceil(I_TF / I0))  # number of necessary conductors

T_sc = 4.2  # operational temperature of superconducting cable
T_margin = 1.5  # temperature margin
T0 = T_sc + T_margin
t_delay = 3
tau = 18.0
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
sc_strand = Wire_Nb3Sn(d_strand=1e-3)
copper300 = Copper300()
insulator = DummyInsulator()
stab_strand = Strand([copper300], [1], d_strand=1e-3)

Ic_sc = sc_strand.Ic(B=B_TF_i, T=T_sc, T_margin=T_margin)
n_sc_strand = np.floor(I0 / Ic_sc)

if B_TF_i < 15:
    cable = DummySquareCableLTS(sc_strand, stab_strand, n_sc_strand, 100, 1e-6,
                                void_fraction=0.7)
else:
    cable = DummySquareCableHTS(sc_strand, stab_strand, n_sc_strand, 100, 1e-6,
                                void_fraction=0.7)
# c.plot(0,0, show=True)

ss316 = AISI_316LN()
conductor = SquareConductor(
    cable=cable, mat_jacket=ss316, mat_ins=insulator, dx_jacket=0.01, dx_ins=1e-3
)

# optimize the number of stabilizer strands using the hot spot criteria
print(f"before optimization: conductor dx_cable = {cable.dx}")
T_for_hts = T0
result = cable.optimize_n_stab_ths(
    t0, tf, T_for_hts, hotspot_target_temperature, B, I, bounds=[1, 10000], show=show
)
print(f"after optimization: conductor dx_cable = {cable.dx}")

# optimize the cable jacket thickness considering 0D stress model for the single cable
print(f"before optimization: conductor dx_jacket = {conductor.dx_jacket}")
result_opt_jacket = optimize_jacket_conductor(
    conductor, pm, T0, B_TF_i, allowable_sigma_jacket, bounds=[1e-5, 0.2]
)
print(f"after optimization: conductor dx_jacket = {conductor.dx_jacket}")

# creation of case
wp1 = WindingPack(conductor, 1, 1)  # just a dummy WP to create the case
case = CaseTF(
    Ri=Ri, dy_ps=0.06, dy_vault=0.3, theta_TF=360 / n_TF, mat_case=ss316, WPs=[wp1]
)

# if show:
#     ax = case.plot(homogenized=False)
#     ax.set_aspect("equal")
#     plt.show()

case.rearrange_conductors_in_wp_type1(n_cond, conductor, case.R_wp_i[0], 0.1, 0.075, 2)
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
