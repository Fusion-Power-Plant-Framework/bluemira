# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021-2023 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh,
#                         J. Morris, D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.
"""Functions to optimise an EUDEMO radial build"""

from typing import Dict, TypeVar

from bluemira.base.parameter_frame import ParameterFrame
from bluemira.codes import plot_radial_build, systems_code_solver
from bluemira.codes.process._inputs import ProcessInputs

_PfT = TypeVar("_PfT", bound=ParameterFrame)


CONSTRAINT_EQS = [
    1,  # Beta Consistency
    # JUSTIFICATION: Consistency equations should always be on
    2,  # Global Power Balance Consistency
    # JUSTIFICATION: Consistency equations should always be on
    5,  # Density Upper Limit
    # JUSTIFICATION: Used to enforce Greenwald limit
    8,  # Neutron wall load upper limit
    # JUSTIFICATION: To keep component lifetime acceptable
    11,  # Radial Build Consistency
    # JUSTIFICATION: Consistency equations should always be on
    13,  # Burn time lower limit
    # JUSTIFICATION: Required minimum burn time
    15,  # L-H Power Threshold Limit
    # JUSTIFICATION: Required to be in H-mode
    16,  # Net electric power lower limit
    # JUSTIFICATION: Required to generate net electricity
    24,  # Beta Upper Limit
    # JUSTIFICATION: Limit for plasma stability
    # 25, # Max TF field
    # JUSTIFICATION: switch off
    26,  # Central solenoid EOF current density upper limit
    # JUSTIFICATION: enforce current limits on inductive current drive
    27,  # Central solenoid BOP current density upper limit
    # JUSTIFICATION: enforce current limits on inductive current drive
    30,  # Injection Power Upper Limit
    # JUSTIFICATION: Limit for plasma stability
    31,  # TF coil case stress upper limit
    # JUSTIFICATION: The support structure must hold
    32,  # TF WP steel jacket/conduit stress upper limit
    # JUSTIFICATION: The turn support structure must hold
    33,  # TF superconductor operating current / critical current density
    # JUSTIFICATION: A quench must be avoided
    34,  # Dump voltage upper limit
    # JUSTIFICATION: Quench protection constraint
    35,  # J_winding pack
    # JUSTIFICATION: Constraint of TF engineering desgin
    36,  # TF temperature marg
    # JUSTIFICATION: Constraint of TF engineering desgin
    60,  # OH coil temp margin
    # JUSTIFICATION: Constraint of CS engineering desgin
    62,  # taup/taueff ratio of particle to energy confinement times
    # JUSTIFICATION: Used to constrain helium fraction
    65,  # dump time by VV stresses
    # JUSTIFICATION: Quench protection constraint
    68,  # Pseparatrix Bt / q A R upper limit
    # JUSTIICATION: Divertor protection
    72,  # OH stress limit
    # JUSTIFICATION: CS coil structure must hold
    81,  # ne(0) > ne(ped) constraint
    # JUSTIFICATION: Prevents unrealistic density profiles
    90,  # CS fatigue constraints
    # JUSTIFICATION: Enforce number of cycles over lifetime
]

ITERATION_VARS = [
    2,  # Toroidal field on axis (T)
    3,  # Plasma major radius
    4,  # Volume averaged electron temperature (keV)
    5,  # Total Plasma Beta
    6,  # Electron density (/m3)
    9,  # f-value for density limit
    # 11, # heating power not used for current drive (MW)
    13,  # inboard TF coil thickness
    14,  # f-value for neutron wall load limit
    # 15, # F-value for volt-sec consistency (icc=12)
    16,  # Central solenoid thickness (m)
    18,  # Safety factor at 95% flux surface
    # TODO: Figure out how bounds are actually changed in this shit
    # 21, # F-value for minimum burn time (icc=13)
    25,  # F-value for net electric power (icc=16)
    29,  # central solenoid inboard radius (m)
    36,  # f-value for Beta Limit
    37,  # Central solenoid overall current density at end of flat-top (A/m2)
    38,  # f-value for central solenoid current at end-of-flattop
    39,  # f-value for central solenoid current at beginning of pulse
    41,  # ratio of central solenoid overall current density at beginning of pulse / end of flat-top
    # 42,  # gap between central solenoid and TF coil (m)
    44,  # fraction of the plasma current produced by non-inductive means
    # 46, # value for injection power
    48,  # f-value for Maxiumum TF Coil case (bucking) TRESCA stress
    49,  # f-value for Maxiumum TF Coil Conduit Tresca Stress
    50,  # f-value for TF coil operating current / critical current density ratio
    51,  # f-value for dump voltage
    52,  # Max voltage across TF coil during quench (kV)
    53,  # f-value for TF coil winding pack current density
    54,  # f-value for TF coil temperature margin
    56,  # fast discharge time for TF coil in event of quench (s)
    57,  # inboard TF coil case outer (non-plasma side) thickness (m)
    58,  # TF coil conduit case thickness (m)
    59,  # copper fraction of cable conductor (TF coils)
    60,  # Max TF coil current per turn [A]
    # 61,  # gap between inboard vacuum vessel and thermal shield (m)
    103,  # f-value for L-H Power Threshold
    106,  # f-value for central solenoid temperature margin
    109,  # Thermal alpha density / electron density
    110,  # f-falue for the He/energy confinement time ratio
    113,  # f-value for calculated minimum TF quench time
    117,  # f-value for upper limit on psepbqar, maximum Psep*Bt/qAR limits
    122,  # central solenoid steel fraction
    123,  # f-value for Tresca yield criterion in Central Solenoid
    135,  # Xenon Impurity Concentration
    154,  # f-value for ne(0) > ne(sep) (icc=81)
    167,  # f-value for constraint n_cycle > n_cycle_min
]

BOUNDS = {
    "2": {"u": "20.0"},  # bt
    "3": {"u": "13"},  # rmajor
    "4": {"u": "150.0"},  # te
    "9": {"u": "1.2"},  # fdene
    # "13": {"l": 1.4},  # tfcth
    "16": {"l": "0.1"},  # ohcth
    "18": {"l": "3.8"},  # q_95  (!!)
    "29": {"l": "0.1"},  # bore
    "38": {"u": "1.0"},  # fjohc
    "39": {"u": "1.0"},  # fjohc0
    "42": {"l": "0.05", "u": "0.1"},  # gapoh
    "50": {"u": "1.0"},  # fiooic
    "52": {"u": "10.0"},  # vdalw
    "58": {"l": "8.0e-3"},  # thwcndut
    "59": {"l": "0.50", "u": "0.94"},  # fcutfsu
    "60": {"l": "6.0e4", "u": "9.0e4"},  # cpttf
    "61": {"l": "0.02"},  # gapds
    "103": {"l": "1.1", "u": "1.2"},  # flhthresh  (!!)
    "154": {"u": "0.95"},  # fne0
}

EUDEMO_PROCESS_INPUTS = ProcessInputs(
    bounds=BOUNDS,
    icc=CONSTRAINT_EQS,
    ixc=ITERATION_VARS,
    flhthresh=1.2,
    bt=5.2652e00,
    rmajor=8.7438e00,
    te=1.2188e01,
    dene=7.8505e19,
    fdene=1.2,
    beta=3.2792e-02,
    tfcth=1.3825e00,
    bore=2.1200e00,
    fwalld=1.3291e-01,
    ohcth=6.6632e-01,
    fpnetel=1.0,
    fbetatry=4.8122e-01,
    coheof=1.6025e07,
    fjohc=6.2232e-01,
    fjohc0=5.5465e-01,
    fcohbop=9.1715e-01,
    fvsbrnni=4.8711e-01,
    fstrcase=1.0,
    fstrcond=7.6710e-01,
    fiooic=7.1982e-01,
    fvdump=1.0,
    vdalw=10.0,
    fjprot=1.0,
    ftmargtf=1.0,
    tdmptf=2.6933e01,
    thkcas=5.3165e-01,
    thwcndut=8.0e-3,
    fcutfsu=8.7819e-01,
    cpttf=6.5e4,
    ftmargoh=1.0,
    ralpne=6.8940e-02,
    ftaulimit=1.0,
    ftaucq=9.2751e-01,
    fpsepbqar=8.5689e-01,
    oh_steel_frac=7.6206e-01,
    foh_stress=5.1706e-01,
    fne0=5.9638e-01,
    fncycle=1.0,
    pinjalw=50.0,
    neped=0.678e20,
    nesep=0.2e20,
    rhopedn=0.94,
    rhopedt=0.94,
    q=3.8,
    m_s_limit=0.2,
    hfact=1.1,
    ishape=10,
    life_dpa=70.0,
    pheat=10,
    gapds=0.02,
    tbrnmn=7200.0,
    tburn=7200.0,
    n_cycle_min=20000,
    # fimp = [1.0, 0.1, *([0.0] * 10), 0.00044, 5e-05]
    # ipfloc =[2, 2, 3, 3]
    # ncls=[1, 1, 2, 2]
    # cptdin = [*([42200.0] * 4), *([43000.0] * 4)]
    # rjconpf=[1.1e7, 1.1e7, 6e6, 6e6, 8e6, 8e6, 8e6, 8e6]
    # zref=[3.6, 1.2, 1.0, 2.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
)


def radial_build(params: _PfT, build_config: Dict) -> _PfT:
    """
    Update parameters after a radial build is run/read/mocked using PROCESS.

    Parameters
    ----------
    params:
        Parameters on which to perform the solve (updated)
    build_config:
        Build configuration

    Returns
    -------
    Updated parameters following the solve.
    """
    run_mode = build_config.pop("run_mode", "mock")
    plot = build_config.pop("plot", False)
    if run_mode == "run":
        build_config["template_in_dat"] = EUDEMO_PROCESS_INPUTS.to_invariable()
    solver = systems_code_solver(params, build_config)
    new_params = solver.execute(run_mode)

    if plot:
        plot_radial_build(solver.read_directory)
    params.update_from_frame(new_params)
    return params
