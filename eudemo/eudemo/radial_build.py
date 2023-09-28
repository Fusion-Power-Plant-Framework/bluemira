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
from bluemira.codes.process._model_mapping import (
    AlphaPressureModel,
    AvailabilityModel,
    BetaLimitModel,
    BlanketModel,
    BootstrapCurrentScalingLaw,
    CSSuperconductorModel,
    ConfinementTimeScalingLaw,
    CostModel,
    CurrentDriveEfficiencyModel,
    DensityLimitModel,
    EPEDScalingModel,
    FISPACTSwitchModel,
    LHThreshholdScalingLaw,
    OperationModel,
    OutputCostsSwitch,
    PFConductorModel,
    PFSuperconductorModel,
    PROCESSOptimisationAlgorithm,
    PlasmaCurrentScalingLaw,
    PlasmaGeometryModel,
    PlasmaNullConfigurationModel,
    PlasmaPedestalModel,
    PlasmaProfileModel,
    PlasmaWallGapModel,
    PowerFlowModel,
    PrimaryPumpingModel,
    PulseTimingModel,
    SecondaryCycleModel,
    ShieldThermalHeatUse,
    SolenoidSwitchModel,
    TFCSTopologyModel,
    TFCasingGeometryModel,
    TFCoilConductorTechnology,
    TFCoilShapeModel,
    TFNuclearHeatingModel,
    TFSuperconductorModel,
    TFWindingPackGeometryModel,
    TFWindingPackTurnModel,
)
from bluemira.codes.process.template_builder import PROCESSTemplateBuilder

_PfT = TypeVar("_PfT", bound=ParameterFrame)


template_builder = PROCESSTemplateBuilder()
template_builder.set_optimisation_algorithm(PROCESSOptimisationAlgorithm.VMCON)
template_builder.set_optimisation_numerics(max_iterations=1000, tolerance=1e-8)

template_builder.set_minimisation_objective("rmajor")

for constraint in [
    "beta_consistency",
    "global_power_consistency",
    "radial_build_consistency",
    "confinement_ratio_lower_limit",
    "density_upper_limit",
    "density_profile_sanity",
    "beta_upper_limit",
    "NWL_upper_limit",
    "burn_time_lower_limit",
    "net_electric_lower_limit",
    "LH_threshhold_limit",
    "PsepBqAR_upper_limit",
    "Pinj_upper_limit",
    "TF_case_stress_upper_limit",
    "TF_jacket_stress_upper_limit",
    "TF_jcrit_ratio_upper_limit",
    "TF_dump_voltage_upper_limit",
    "TF_current_density_upper_limit",
    "TF_temp_margin_lower_limit",
    "CS_fatigue",
    "CS_stress_upper_limit",
    "CS_temp_margin_lower_limit",
    "CS_EOF_density_limit",
    "CS_BOP_density_limit",
]:
    template_builder.add_constraint(constraint)

# Variable vector values and bounds
template_builder.add_variable("bt", 5.5, lower_bound=2.0, upper_bound=20.0)
template_builder.add_variable("rmajor", 9.0, lower_bound=5.0, upper_bound=13.0)
template_builder.add_variable("te", 12.0, upper_bound=150.0)
template_builder.add_variable("beta", 3.14e-2)
template_builder.add_variable("dene", 7.85e19)
template_builder.add_variable("q", 3.8, lower_bound=3.8)
template_builder.add_variable("bore", 1.8, lower_bound=0.1)
template_builder.add_variable("ohcth", 0.66, lower_bound=0.1)
template_builder.add_variable("thwcndut", 8.0e-3, lower_bound=8.0e-3)
template_builder.add_variable("thkcas", 0.52, lower_bound=0.1)
template_builder.add_variable("tfcth", 1.0, lower_bound=0.2)
template_builder.add_variable("gapoh", 0.05, lower_bound=0.05, upper_bound=0.1)
template_builder.add_variable("coheof", 1.6e07)
template_builder.add_variable("oh_steel_frac", 0.76)
template_builder.add_variable("ralpne", 6.9e-02)
template_builder.add_variable("cpttf", 6.5e4, lower_bound=6.0e4, upper_bound=9.0e4)
template_builder.add_variable("tdmptf", 2.7e1, lower_bound=0.1)
template_builder.add_variable("vdalw", 10.0, upper_bound=10.0)
template_builder.add_variable("fimp(13)", 4.4e-4, lower_bound=0.0, upper_bound=0.1)

# Modified f-values and bounds w.r.t. defaults [0.001 < 0.5 < 1.0]
template_builder.add_variable("fdene", 1.2, upper_bound=1.2)
template_builder.add_variable("fne0", 0.6, upper_bound=0.95)
template_builder.add_variable("flhthresh", 1.15, lower_bound=1.1, upper_bound=1.2)
template_builder.add_variable("fcutfsu", 0.88, lower_bound=0.5, upper_bound=0.94)

# Modifying the initial variable vector to improve convergence
template_builder.add_variable("fpnetel", 1.0)
template_builder.add_variable("fncycle", 1.0)
template_builder.add_variable("fstrcase", 1.0)
template_builder.add_variable("ftmargtf", 1.0)
template_builder.add_variable("ftmargoh", 1.0)
template_builder.add_variable("ftaulimit", 1.0)
template_builder.add_variable("ftaucq", 0.93)
template_builder.add_variable("fpsepbqar", 1.0)
template_builder.add_variable("fvdump", 1.0)
template_builder.add_variable("fcohbop", 0.9)
template_builder.add_variable("fwalld", 0.13)
template_builder.add_variable("fstrcond", 0.77)
template_builder.add_variable("fiooic", 0.72)
template_builder.add_variable("fjprot", 1.0)
template_builder.add_variable("fpinj", 1.0)

# Set model switches
for model_choice in [
    BootstrapCurrentScalingLaw.SAUTER,
    ConfinementTimeScalingLaw.IPB98_Y2_H_MODE,
    PlasmaCurrentScalingLaw.ITER_REVISED,
    PlasmaProfileModel.CONSISTENT,
    PlasmaPedestalModel.PEDESTAL_GW,
    EPEDScalingModel.SAARELMA,
    BetaLimitModel.THERMAL,
    DensityLimitModel.GREENWALD,
    AlphaPressureModel.WARD,
    LHThreshholdScalingLaw.MARTIN_NOM,
    PlasmaNullConfigurationModel.SINGLE_NULL,
    PlasmaGeometryModel.CREATE_A_M_S,
    PlasmaWallGapModel.INPUT,
    PowerFlowModel.SIMPLE,
    PrimaryPumpingModel.PRESSURE_DROP,
    ShieldThermalHeatUse.LOW_GRADE_HEAT,
    SecondaryCycleModel.FIXED,
    BlanketModel.CCFE_HCPB,
    CurrentDriveEfficiencyModel.ECRH_UI_GAM,
    OperationModel.PULSED,
    PulseTimingModel.RAMP_RATE,
    PFConductorModel.SUPERCONDUCTING,
    PFSuperconductorModel.NBTI,
    SolenoidSwitchModel.SOLENOID,
    CSSuperconductorModel.NB3SN_WST,
    TFCasingGeometryModel.FLAT,
    TFCoilConductorTechnology.SC,
    TFCoilShapeModel.PRINCETON,
    TFCSTopologyModel.ITER,
    TFSuperconductorModel.NB3SN_WST,
    TFWindingPackGeometryModel,
    TFWindingPackTurnModel.INTEGER_TURN,
    FISPACTSwitchModel.OFF,
    TFNuclearHeatingModel.INPUT,
    CostModel.TETRA_1990,
    AvailabilityModel.INPUT,
    OutputCostsSwitch.NO,
]:
    template_builder.set_model(model_choice)


# Set fixed input values
template_builder.add_input_values(
    {
        # Profile parameterisation inputs
        "alphan": 1.0,
        "alphat": 1.45,
        "rhopedn": 0.94,
        "rhopedt": 0.94,
        "tbeta": 2.0,
        "teped": 5.5,
        "tesep": 0.1,
        "fgwped": 0.85,
        "neped": 0.678e20,
        "nesep": 0.2e20,
        "dnbeta": 3.0,
        "fkzohm": 1.0245,  # not used..?
        # Important stuff
        "pnetelin": 500.0,
        "tbrnmn": 7.2e3,
        "sig_tf_case_max": 5.8e8,
        "sig_tf_wp_max": 5.8e8,
        "alstroh": 6.6e8,
        "psepbqarmax": 9.2,
        "aspect": 3.1,
        "m_s_limit": 0.1,
        "q0": 1.0,
        "ssync": 0.6,
        "plasma_res_factor": 0.66,
        "gamma": 0.3,
        "hfact": 1.1,
        "lifedpa": 70.0,
        # Radial build inputs
        "tftsgap": 0.05,
        "d_vv_in": 0.3,
        "shldith": 0.3,
        "vvblgap": 0.02,
        "blnkith": 0.755,
        "scrapli": 0.225,
        "scraplo": 0.225,
        "blnkoth": 0.982,
        "d_vv_out": 0.3,
        "shldoth": 0.8,
        "ddwex": 0.15,
        "gapomin": 0.2,
        # Vertical build inputs
        "d_vv_top": 0.3,
        "vgap2": 0.05,
        "shldtth": 0.3,
        "divfix": 0.621,
        "d_vv_bot": 0.3,
        "pinjalw": 51.0,
        "gamma_ecrh": 0.3,
        "etaech": 0.4,
        "etath": 0.375,
        "etahtp": 0.87,
        "etaiso": 0.9,
        "vfshld": 0.6,
        "coreradius": 0.75,
        "coreradiationfraction": 0.6,
        "tdwell": 0.0,
        "tramp": 500.0,
        # CS / PF coil inputs
        "t_crack_vertical": 0.004,
        "fcuohsu": 0.7,
        "ohhghf": 0.9,
        "rpf2": -1.825,
        "cptdin": [4.22e4, 4.22e4, 4.22e4, 4.22e4, 4.3e4, 4.3e4, 4.3e4, 4.3e4],
        "ipfloc": [2, 2, 3, 3],
        "ncls": [1, 1, 2, 2],
        "ngrp": 4,
        "rjconpf": [1.1e7, 1.1e7, 6.0e6, 6.0e6, 8.0e6, 8.0e6, 8.0e6, 8.0e6],
        "zref": [3.6, 1.2, 1.0, 2.8, 1.0, 1.0, 1.0, 1.0],
        # TF coil inputs
        "n_tf": 16,
        "casthi": 0.06,
        "casths": 0.05,
        "ripmax": 0.06,
        "dhecoil": 0.01,
        "tftmp": 4.75,
        "thicndut": 0.002,
        "tinsft": 0.008,
        "tmargmin": 1.5,
        "vftf": 0.3,
        "n_pancake": 20,
        "n_layer": 10,
        "qnuc": 1.292e4,
        # Inputs we don't care about but must specify
        "cfactr": 0.75,  # Ha!
        "kappa": 1.848,  # Should be overwritten
        "triang": 0.5,  # Should be overwritten
        "walalw": 8.0,  # Should never get even close to this
    }
)

template = template_builder.make_inputs()


EUDEMO_PROCESS_INPUTS = ProcessInputs(
    tdmptf=2.6933e01,
    pinjalw=50.0,
    neped=0.678e20,
    nesep=0.2e20,
    rhopedn=0.94,
    rhopedt=0.94,
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
