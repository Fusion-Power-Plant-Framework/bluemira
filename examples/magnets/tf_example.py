import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.constants import MU_0, MU_0_2PI, MU_0_4PI
from bluemira.magnets.cable import DummyRoundCableLTS, DummyRoundCableHTS
from bluemira.magnets.case_tf import CaseTF
from bluemira.magnets.conductor import SquareConductor
from bluemira.magnets.materials import (OperationalPoint, Copper300, AISI_316LN,
                                        DummyInsulator)
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

# plot options
show = True
homogenized = False  # if True plot the WP as a homogenized block

# input values
B0 = 4.39  # [T] magnetic field @R0
R0 = 8.6  # [m] major machine radius
A = 2.8
n_TF = 16  # number of TF coils
d = 1.9
ripple = 6e-3
a = R0 / A
Ri = R0 - a - d  # [m] max external radius of the internal TF leg
Re = (R0 + a) * (1 / ripple) ** (
        1 / n_TF)  # [m] max internal radius of the external TF part
dr_plasma_side = R0 * 2 / 3 * 1e-2

R_VV = Ri * 1.05  # Vacuum vessel radius
S_VV = 90e6  # Vacuum vessel steel limit

I_TF = B0 * R0 / MU_0_2PI / n_TF  # total current in each TF coil
B_TF_i = 1.08 * (MU_0_2PI * n_TF * I_TF / Ri)  # max magnetic field on the inner TF leg
pm = B_TF_i ** 2 / (2 * MU_0)  # magnetic pressure on the inner TF leg
# vertical tension acting on the equatorial section of inner TF leg
# i.e. half of the whole F_Z
t_z = (0.5 * np.log(Re / Ri) * MU_0_4PI * n_TF * I_TF ** 2)

Iop = 90.0e3  # operational current in each conductor
n_cond = int(np.ceil(I_TF / Iop))  # number of necessary conductors

n_spire = np.floor(I_TF / Iop)
L = MU_0 * R0 * (n_TF * n_spire) ** 2 * (1 - np.sqrt(1 - (R0 - Ri) / R0)) / n_TF * 1.1
E = 1 / 2 * L * n_TF * Iop ** 2 * 1e-9
V_MAX = (7 * R0 - 3) / 6 * 1.1e3
Tau_discharge1 = (L * Iop / V_MAX)
Tau_discharge2 = B0 * I_TF * n_TF * (R0 / A) ** 2 / (R_VV * S_VV)
Tau_discharge = max([Tau_discharge1, Tau_discharge2])

T_sc = 4.2  # operational temperature of superconducting cable
T_margin = 1.5  # temperature margin
T0 = T_sc + T_margin
t_delay = 3
t0 = 0
tf = Tau_discharge
hotspot_target_temperature = 250.0

# allowable stress values
safety_factor = 1.5 * 1.3
S_Y = 1e9 / safety_factor  # [Pa] steel allowable limit

# Current and magnetic field behaviour
I = delayed_exp_func(Iop, Tau_discharge, t_delay)
B = delayed_exp_func(B_TF_i, Tau_discharge, t_delay)

# define the conductor (with a dummy number of stabilizer strands )
sc_strand = Wire_Nb3Sn(d_strand=1e-3)
copper300 = Copper300()
insulator = DummyInsulator()
stab_strand = Strand([copper300], [1], d_strand=1.0e-3)

# Calculate number of superconducting strands considering the strand critical current
Ic_sc = sc_strand.Ic(B=B_TF_i, T=T_sc, T_margin=T_margin)
n_sc_strand = int(np.ceil(Iop / Ic_sc))

if B_TF_i < 15:
    cable = DummyRoundCableLTS(sc_strand, stab_strand, n_sc_strand, 100, 1e-2,
                               void_fraction=0.7)
else:
    cable = DummyRoundCableHTS(sc_strand, stab_strand, n_sc_strand, 100, 1e-2,
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

op = OperationalPoint(B=B_TF_i, T=T0)

show = False

# Some iterations to optimize case vault and cable jacket consistently. This should go
# in a kind of "optimization" or "convergence" loop.
for i in range(10):
    print(f"Internal optimazion - iteration {i}")
    # optimize the cable jacket thickness considering 0D stress model for the single cable
    print(f"before optimization: conductor dx_jacket = {conductor.dx_jacket}")
    t_z_cable_jacket = 0
    if i == 0:
        t_z_cable_jacket = t_z / 2 / n_spire
    else:
        t_z_cable_jacket = t_z * case.area_wps_jacket / (
                case.area_jacket + case.area_wps_jacket) / (
                               np.sum([w.nx * w.ny for w in case.WPs]))
    conductor.optimize_jacket_conductor(
        pm, t_z_cable_jacket, T0, B_TF_i,
        S_Y, bounds=[1e-5, 0.2]
    )
    print(f"after optimization: conductor dx_jacket = {conductor.dx_jacket}")

    # creation of case
    wp1 = WindingPack(conductor, 1, 1)  # just a dummy WP to create the case
    case = CaseTF(
        Ri=Ri, dy_ps=dr_plasma_side, dy_vault=0.6, theta_TF=360 / n_TF, mat_case=ss316,
        WPs=[wp1]
    )

    if show:
        ax = case.plot(homogenized=False)
        ax.set_aspect("equal")
        plt.show()

    case.rearrange_conductors_in_wp(n_cond, conductor, case.R_wp_i[0],
                                          case.dx_i * 0.7, 0.05, 4)

    if show:
        ax = case.plot(homogenized=homogenized)
        ax.set_aspect("equal")
        plt.title("Before vault optimization")
        plt.show()

    case.optimize_vault_radial_thickness(
        pm, t_z, T=T0, B=B_TF_i, allowable_sigma=S_Y, bounds=[1e-2, 1]
    )
    if show:
        ax = case.plot(homogenized=homogenized)
        ax.set_aspect("equal")
        plt.title("After vault optimization")
        plt.show()

show = True
if show:
    ax = case.plot(homogenized=homogenized)
    ax.set_aspect("equal")
    plt.title("After vault optimization")
    plt.show()
