# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Example for TF coils internal structure optimization.

Note: in this example the conductor operational current is given as input.
"""

import matplotlib.pyplot as plt
import numpy as np

from bluemira.base.constants import MU_0, MU_0_2PI, MU_0_4PI
from bluemira.magnets.cable import DummyRoundCableHTS, DummyRoundCableLTS
from bluemira.magnets.case_tf import CaseTF
from bluemira.magnets.conductor import SquareConductor
from bluemira.magnets.materials import AISI_316LN, Copper300, DummyInsulator
from bluemira.magnets.strand import Strand, WireNb3Sn
from bluemira.magnets.utils import (
    delayed_exp_func,
)
from bluemira.magnets.winding_pack import WindingPack

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
        1 / n_TF
)  # [m] max internal radius of the external TF part
dr_plasma_side = R0 * 2 / 3 * 1e-2

R_VV = Ri * 1.05  # Vacuum vessel radius
S_VV = 90e6  # Vacuum vessel steel limit

I_TF = B0 * R0 / MU_0_2PI / n_TF  # total current in each TF coil
B_TF_i = 1.08 * (MU_0_2PI * n_TF * I_TF / Ri)  # max magnetic field on the inner TF leg
pm = B_TF_i ** 2 / (2 * MU_0)  # magnetic pressure on the inner TF leg
# vertical tension acting on the equatorial section of inner TF leg
# i.e. half of the whole F_Z
t_z = 0.5 * np.log(Re / Ri) * MU_0_4PI * n_TF * I_TF ** 2

Iop = 90.0e3  # operational current in each conductor
# n_cond = int(np.ceil(I_TF / Iop))  # number of necessary conductors

n_cond = np.floor(I_TF / Iop)
L = MU_0 * R0 * (n_TF * n_cond) ** 2 * (1 - np.sqrt(1 - (R0 - Ri) / R0)) / n_TF * 1.1
E = 1 / 2 * L * n_TF * Iop ** 2 * 1e-9
V_MAX = (7 * R0 - 3) / 6 * 1.1e3
Tau_discharge1 = L * Iop / V_MAX
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
I = delayed_exp_func(Iop, Tau_discharge, t_delay)  # noqa: E741
B = delayed_exp_func(B_TF_i, Tau_discharge, t_delay)

# define the conductor (with a dummy number of stabilizer strands )
sc_strand = WireNb3Sn(d_strand=1e-3)
copper300 = Copper300()
insulator = DummyInsulator()
stab_strand = Strand([copper300], [1], d_strand=1.0e-3)

# Calculate number of superconducting strands considering the strand critical current
Ic_sc = sc_strand.Ic(B=B_TF_i, T=T_sc, T_margin=T_margin)
n_sc_strand = int(np.ceil(Iop / Ic_sc))

B_ref = 15

if B_TF_i < B_ref:
    cable = DummyRoundCableLTS(
        sc_strand, stab_strand, n_sc_strand, 1, 1e-2, void_fraction=0.7
    )
else:
    cable = DummyRoundCableHTS(
        sc_strand, stab_strand, n_sc_strand, 1, 1e-2, void_fraction=0.7
    )
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

# creation of case
wp1 = WindingPack(conductor, 1, 1)  # just a dummy WP to create the case
case = CaseTF(
    Ri=Ri,
    dy_ps=dr_plasma_side,
    dy_vault=0.6,
    theta_TF=360 / n_TF,
    mat_case=ss316,
    WPs=[wp1],
)

show = False

# Some iterations to optimize case vault and cable jacket consistently. This should go
# in a kind of "optimization" or "convergence" loop.
max_niter = 10
eps = 1e-3
i = 0
tot_err = 100 * eps

while i < max_niter and tot_err > eps:
    i += 1
    print(f"Internal optimazion - iteration {i}")
    print(f"before optimization: conductor dx_jacket = {conductor.dx_jacket}")
    cond_dx_jacket0 = conductor.dx_jacket
    t_z_cable_jacket = (
            t_z
            * case.area_wps_jacket
            / (case.area_jacket + case.area_wps_jacket)
            / (np.sum([w.nx * w.ny for w in case.WPs]))
    )
    conductor.optimize_jacket_conductor(
        pm, t_z_cable_jacket, T0, B_TF_i, S_Y, bounds=[1e-5, 0.2]
    )
    print(f"after optimization: conductor dx_jacket = {conductor.dx_jacket}")
    delta_conductor_dx_jacket = abs(conductor.dx_jacket - cond_dx_jacket0)
    err_dy_jacket = delta_conductor_dx_jacket / conductor.dy_jacket

    case.rearrange_conductors_in_wp(
        n_cond, conductor, case.R_wp_i[0], case.dx_i * 0.7, 0.05, 4
    )

    case_dy_vault0 = case.dy_vault
    print(f"before optimization: case dy_vault = {case.dy_vault}")
    case.optimize_vault_radial_thickness(
        pm, t_z, T=T0, B=B_TF_i, allowable_sigma=S_Y, bounds=[1e-2, 1]
    )
    print(f"after optimization: case dy_vault = {case.dy_vault}")
    delta_case_dy_vault = abs(case.dy_vault - case_dy_vault0)
    err_dy_vault = delta_case_dy_vault / case.dy_vault
    tot_err = err_dy_vault + err_dy_jacket

    print(f"err_dy_jacket = {err_dy_jacket}")
    print(f"err_dy_vault = {err_dy_vault}")
    print(f"tot_err = {tot_err}")


show = True
if show:
    scalex = np.array([2, 1])
    scaley = np.array([1, 1.2])

    ax = case.plot(homogenized=homogenized)
    ax.set_aspect("equal")

    # Fix the x and y limits
    ax.set_xlim(-scalex[0] * case.dx_i, scalex[1] * case.dx_i)
    ax.set_ylim(scaley[0] * 0, scaley[1] * case.Ri)

    deltax = [-case.dx_i / 2, case.dx_i / 2]

    ax.plot([-scalex[0] * case.dx_i, -case.dx_i / 2], [case.Ri, case.Ri], "k:")

    for i in range(len(case.WPs)):
        ax.plot(
            [-scalex[0] * case.dx_i, -case.dx_i / 2],
            [case.R_wp_i[i], case.R_wp_i[i]],
            "k:",
        )

    ax.plot(
        [-scalex[0] * case.dx_i, -case.dx_i / 2],
        [case.R_wp_k[-1], case.R_wp_k[-1]],
        "k:",
    )
    ax.plot([-scalex[0] * case.dx_i, -case.dx_i / 2], [case.Rk, case.Rk], "k:")

    ax.set_title("Equatorial cross section of the TF WP")
    ax.set_xlabel("Toroidal direction [m]")
    ax.set_ylabel("Radial direction [m]")

    plt.show()
