import matplotlib.pyplot as plt
from bluemira.magnets.cable import DummyRectangularCableLTS
from bluemira.magnets.strand import SuperconductingStrand
import numpy as np
from eurofusion_materials.library.magnet_branch_mats import (
    COPPER_100,
    COPPER_300,
    DUMMY_INSULATOR_MAG,
    NB3SN_MAG,
    SS316_LN_MAG,
)
from matproplib import OperationalConditions

op_cond = OperationalConditions(temperature=5.7, magnetic_field=10.0, strain=0.0055)

print("Nb3Sn resistivity:", NB3SN_MAG.electrical_resistivity(op_cond))
print("Copper 100 resistivity:", COPPER_100.electrical_resistivity(op_cond))
print("Copper 300 resistivity:", COPPER_300.electrical_resistivity(op_cond))
print("SS316 LN resistivity:", SS316_LN_MAG.electrical_resistivity(op_cond))
print(
    "Dummy insulator resistivity:", DUMMY_INSULATOR_MAG.electrical_resistivity(op_cond)
)
print("Nb3Sn specific heat:", NB3SN_MAG.specific_heat_capacity(op_cond))
print("Copper 100 specific heat:", COPPER_100.specific_heat_capacity(op_cond))
print("Copper 300 specific heat:", COPPER_300.specific_heat_capacity(op_cond))


from bluemira.base.constants import MU_0, MU_0_2PI, MU_0_4PI
from bluemira.magnets.case_tf import create_case_tf_from_dict

case_tf_dict = {
    "name_in_registry": "TrapezoidalCaseTF",
    "name": "TrapezoidalCaseTF",
    "Ri": 3.708571428571428,
    "dy_ps": 0.05733333333333333,
    "dy_vault": 0.4529579163961617,
    "theta_TF": 22.5,
    "mat_case": SS316_LN_MAG,
    "WPs": [
        {
            "name_in_registry": "WindingPack",
            "name": "WindingPack",
            "conductor": {
                "name_in_registry": "SymmetricConductor",
                "name": "SymmetricConductor",
                "cable": {
                    "name_in_registry": "DummyRectangularCableLTS",
                    "name": "DummyRectangularCableLTS",
                    "n_sc_strand": 321,
                    "n_stab_strand": 476,
                    "d_cooling_channel": 0.01,
                    "void_fraction": 0.7,
                    "cos_theta": 0.97,
                    "sc_strand": {
                        "name_in_registry": "SuperconductingStrand",
                        "name": "Nb3Sn_strand",
                        "d_strand": 0.001,
                        "temperature": 5.7,
                        "materials": [
                            {"material": NB3SN_MAG, "fraction": 0.5},
                            {"material": COPPER_100, "fraction": 0.5},
                        ],
                    },
                    "stab_strand": {
                        "name_in_registry": "Strand",
                        "name": "Stabilizer",
                        "d_strand": 0.001,
                        "temperature": 5.7,
                        "materials": [{"material": COPPER_300, "fraction": 1.0}],
                    },
                    "dx": 0.034648435154495685,
                    "aspect_ratio": 1.2,
                },
                "mat_jacket": SS316_LN_MAG,
                "mat_ins": DUMMY_INSULATOR_MAG,
                "dx_jacket": 0.0030808556812487366,
                "dx_ins": 0.001,
            },
            "nx": 25,
            "ny": 6,
        },
        {
            "name_in_registry": "WindingPack",
            "name": "WindingPack",
            "conductor": {
                "name_in_registry": "SymmetricConductor",
                "name": "SymmetricConductor",
                "cable": {
                    "name_in_registry": "DummyRectangularCableLTS",
                    "name": "DummyRectangularCableLTS",
                    "n_sc_strand": 321,
                    "n_stab_strand": 476,
                    "d_cooling_channel": 0.01,
                    "void_fraction": 0.7,
                    "cos_theta": 0.97,
                    "sc_strand": {
                        "name_in_registry": "SuperconductingStrand",
                        "name": "Nb3Sn_strand",
                        "d_strand": 0.001,
                        "temperature": 5.7,
                        "materials": [
                            {"material": NB3SN_MAG, "fraction": 0.5},
                            {"material": COPPER_100, "fraction": 0.5},
                        ],
                    },
                    "stab_strand": {
                        "name_in_registry": "Strand",
                        "name": "Stabilizer",
                        "d_strand": 0.001,
                        "temperature": 5.7,
                        "materials": [{"material": COPPER_300, "fraction": 1.0}],
                    },
                    "dx": 0.034648435154495685,
                    "aspect_ratio": 1.2,
                },
                "mat_jacket": SS316_LN_MAG,
                "mat_ins": DUMMY_INSULATOR_MAG,
                "dx_jacket": 0.0030808556812487366,
                "dx_ins": 0.001,
            },
            "nx": 18,
            "ny": 1,
        },
    ],
}

strand_cls = SuperconductingStrand
strand = strand_cls.from_dict(case_tf_dict["WPs"][0]["conductor"]["cable"]["sc_strand"])

print(f"Strand erho: {strand.erho(temperature=5.7, B=10.0)}")
print(f"Strand Cp: {strand.Cp(temperature=5.7, B=10.0)}")
print(f"Strand rho: {strand.rho(temperature=5.7, B=10.0)}")
print(f"Strand E: {strand.E(temperature=5.7, B=10.0)}")

cable_cls = DummyRectangularCableLTS

cable = cable_cls.from_dict(case_tf_dict["WPs"][0]["conductor"]["cable"])

print(f"Cable erho: {cable.erho(temperature=5.7, B=10.0)}")
print(f"Cable Cp: {cable.Cp(temperature=5.7, B=10.0)}")
print(f"Cable rho: {cable.rho(temperature=5.7, B=10.0)}")
print(f"Cable E: {cable.E(temperature=5.7, B=10.0)}")


case_tf = create_case_tf_from_dict(case_tf_dict)

case_tf.plot(show=True, homogenized=False)

# Machine parameters (should match the original setup)
R0 = 8.6
B0 = 4.39
A = 2.8
n_TF = 16
ripple = 6e-3
# operational current per conductor
Iop = 70.0e3
# Safety factor to be considered on the allowable stress
safety_factor = 1.5 * 1.3

# Derived values
a = R0 / A
d = 1.82
Ri = R0 - a - d
Re = (R0 + a) * (1 / ripple) ** (1 / n_TF)
B_TF_i = 1.08 * (MU_0_2PI * n_TF * (B0 * R0 / MU_0_2PI / n_TF) / Ri)
pm = B_TF_i**2 / (2 * MU_0)
t_z = 0.5 * np.log(Re / Ri) * MU_0_4PI * n_TF * (B0 * R0 / MU_0_2PI / n_TF) ** 2
T_sc = 4.2
T_margin = 1.5
T_op = T_sc + T_margin
S_Y = 1e9 / safety_factor
n_cond = int(np.floor((B0 * R0 / MU_0_2PI / n_TF) / Iop))

# Layout and WP parameters
layout = "auto"
wp_reduction_factor = 0.75
min_gap_x = 2 * (R0 * 2 / 3 * 1e-2)  # 2 * dr_plasma_side
n_layers_reduction = 4

# Optimization parameters already defined earlier
bounds_cond_jacket = np.array([1e-5, 0.2])
bounds_dy_vault = np.array([0.1, 2])
max_niter = 100
err = 1e-6

# optimize number of stabilizer strands
sc_strand = case_tf.WPs[0].conductor.cable.sc_strand
Ic_sc = sc_strand.Ic(B=B_TF_i, temperature=T_op)
case_tf.WPs[0].conductor.cable.n_sc_strand = int(np.ceil(Iop / Ic_sc))

from bluemira.magnets.utils import delayed_exp_func

Tau_discharge = 20  # [s]
t_delay = 3  # [s]
t0 = 0  # [s]
hotspot_target_temperature = 250.0  # [K]

tf = Tau_discharge
T_for_hts = T_op
I_fun = delayed_exp_func(Iop, Tau_discharge, t_delay)
B_fun = delayed_exp_func(B_TF_i, Tau_discharge, t_delay)

print("cable")
print(case_tf.WPs[0].conductor.cable)

case_tf.WPs[0].conductor.cable.optimize_n_stab_ths(
    t0,
    tf,
    T_for_hts,
    hotspot_target_temperature,
    B_fun,
    I_fun,
    bounds=[1, 10000],
    show=True,
)

# Optimize case with structural constraints
case_tf.optimize_jacket_and_vault(
    pm=pm,
    fz=t_z,
    temperature=T_op,
    B=B_TF_i,
    allowable_sigma=S_Y,
    bounds_cond_jacket=bounds_cond_jacket,
    bounds_dy_vault=bounds_dy_vault,
    layout=layout,
    wp_reduction_factor=wp_reduction_factor,
    min_gap_x=min_gap_x,
    n_layers_reduction=n_layers_reduction,
    max_niter=max_niter,
    eps=err,
    n_conds=n_cond,
)

case_tf.plot_convergence()

show = True
homogenized = True
if show:
    scalex = np.array([2, 1])
    scaley = np.array([1, 1.2])

    ax = case_tf.plot(homogenized=homogenized)
    ax.set_aspect("equal")

    # Fix the x and y limits
    ax.set_xlim(-scalex[0] * case_tf.dx_i, scalex[1] * case_tf.dx_i)
    ax.set_ylim(scaley[0] * 0, scaley[1] * case_tf.Ri)

    deltax = [-case_tf.dx_i / 2, case_tf.dx_i / 2]

    ax.plot(
        [-scalex[0] * case_tf.dx_i, -case_tf.dx_i / 2], [case_tf.Ri, case_tf.Ri], "k:"
    )

    for i in range(len(case_tf.WPs)):
        ax.plot(
            [-scalex[0] * case_tf.dx_i, -case_tf.dx_i / 2],
            [case_tf.R_wp_i[i], case_tf.R_wp_i[i]],
            "k:",
        )

    ax.plot(
        [-scalex[0] * case_tf.dx_i, -case_tf.dx_i / 2],
        [case_tf.R_wp_k[-1], case_tf.R_wp_k[-1]],
        "k:",
    )
    ax.plot(
        [-scalex[0] * case_tf.dx_i, -case_tf.dx_i / 2], [case_tf.Rk, case_tf.Rk], "k:"
    )

    ax.set_title("Equatorial cross section of the TF WP")
    ax.set_xlabel("Toroidal direction [m]")
    ax.set_ylabel("Radial direction [m]")

    plt.show()

I_sc = case_tf.WPs[0].conductor.cable.sc_strand.Ic(B=B_TF_i, temperature=T_op)
I_max = I_sc * case_tf.WPs[0].conductor.cable.n_sc_strand
I_TF_max = I_max * case_tf.n_conductors
print(I_max)
print(I_TF_max)
I_TF = R0 * B0 / (MU_0_2PI * n_TF)
print(I_TF)
