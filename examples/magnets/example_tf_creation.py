# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Example script demonstrating the design of the TF coil xy cross section.
This involves the design and optimisation of each module: strand, cable,
conductor, winding pack and casing.
"""

from eurofusion_materials.library.magnet_branch_mats import (
    COPPER_100,
    COPPER_300,
    DUMMY_INSULATOR_MAG,
    NB3SN_MAG,
    SS316_LN_MAG,
)

from bluemira.magnets.tfcoil_designer import TFCoilXYDesigner

config = {
    "stabilising_strand": {
        "class": "Strand",
        "materials": [{"material": COPPER_300, "fraction": 1.0}],
        "params": {
            "d_strand": {"value": 1.0e-3, "unit": "m"},
            "operating_temperature": {"value": 5.7, "unit": "K"},
        },
    },
    "superconducting_strand": {
        "class": "SuperconductingStrand",
        "materials": [
            {"material": NB3SN_MAG, "fraction": 0.5},
            {"material": COPPER_100, "fraction": 0.5},
        ],
        "params": {
            "d_strand": {"value": 1.0e-3, "unit": "m"},
            "operating_temperature": {"value": 5.7, "unit": "K"},
        },
    },
    "cable": {
        "class": "RectangularCable",
        "n_sc_strand": 321,
        "n_stab_strand": 476,
        "params": {
            "d_cooling_channel": {"value": 0.01, "unit": "m"},
            "void_fraction": {"value": 0.7, "unit": ""},
            "cos_theta": {"value": 0.97, "unit": ""},
            "dx": {"value": 0.017324217577247843 * 2, "unit": "m"},
            "E": {"value": 0.1e9, "unit": ""},
        },
    },
    "conductor": {
        "class": "SymmetricConductor",
        "jacket_material": SS316_LN_MAG,
        "ins_material": DUMMY_INSULATOR_MAG,
        "params": {
            "dx_jacket": {"value": 0.0015404278406243683 * 2, "unit": "m"},
            # "dy_jacket": 0.0,
            "dx_ins": {"value": 0.0005 * 2, "unit": "m"},
            # "dy_ins": 0.0,
        },
    },
    "winding_pack": {
        "class": "WindingPack",
        "sets": 2,
        "nx": [25, 18],
        "ny": [6, 1],
    },
    "case": {
        "class": "TrapezoidalCaseTF",
        "material": SS316_LN_MAG,
        "params": {
            # "Ri": {"value": 3.708571428571428, "unit": "m"},
            # "Rk": {"value": 0, "unit": "m"},
            "theta_TF": {"value": 22.5, "unit": "deg"},
            "dy_ps": {"value": 0.028666666666666667 * 2, "unit": "m"},
            "dy_vault": {"value": 0.22647895819808084 * 2, "unit": "m"},
        },
    },
    "optimisation_params": {
        "t0": 0,
        "Tau_discharge": 20,
        "hotspot_target_temperature": 250,
        "layout": "auto",
        "wp_reduction_factor": 0.75,
        "n_layers_reduction": 4,
        "bounds_cond_jacket": (1e-5, 0.2),
        "bounds_dy_vault": (0.1, 2),
        "max_niter": 100,
        "eps": 1e-6,
    },
}

params = {
    # base
    "R0": {"value": 8.6, "unit": "m"},
    "B0": {"value": 4.39, "unit": "T"},
    "A": {"value": 2.8, "unit": "dimensionless"},
    "n_TF": {"value": 16, "unit": "dimensionless"},
    "ripple": {"value": 6e-3, "unit": "dimensionless"},
    "d": {"value": 1.82, "unit": "m"},
    "S_VV": {"value": 100e6, "unit": "dimensionless"},
    "safety_factor": {"value": 1.5 * 1.3, "unit": "dimensionless"},
    "B_ref": {"value": 15, "unit": "T"},
    # misc params
    "Iop": {"value": 70.0e3, "unit": "A"},
    "T_sc": {"value": 4.2, "unit": "K"},
    "T_margin": {"value": 1.5, "unit": "K"},
    "t_delay": {"value": 3, "unit": "s"},
    "strain": {"value": 0.0055, "unit": ""},
}

tf_coil_xy = TFCoilXYDesigner(params=params, build_config=config).execute()
tf_coil_xy.plot(show=True, homogenised=False)
tf_coil_xy.plot_convergence(show=True)
tf_coil_xy.plot_summary(100, show=True)
