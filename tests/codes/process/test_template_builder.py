# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Test PROCESS template builder
"""

import os
from pathlib import Path

import numpy as np
import pytest

from bluemira.base.constants import EPS
from bluemira.base.file import try_get_bluemira_private_data_root
from bluemira.codes.process.api import ENABLED, Impurities
from bluemira.codes.process.equation_variable_mapping import Constraint, Objective
from bluemira.codes.process.model_mapping import (
    AlphaJModel,
    AlphaPressureModel,
    AvailabilityModel,
    BetaLimitModel,
    BetaNormMaxModel,
    BootstrapCurrentScalingLaw,
    CSSuperconductorModel,
    ConfinementTimeScalingLaw,
    CostModel,
    CurrentDriveEfficiencyModel,
    DensityLimitModel,
    OperationModel,
    OutputCostsSwitch,
    PFSuperconductorModel,
    PROCESSOptimisationAlgorithm,
    PlasmaCurrentScalingLaw,
    PlasmaGeometryModel,
    PlasmaNullConfigurationModel,
    PlasmaPedestalModel,
    PlasmaProfileModel,
    PowerFlowModel,
    PrimaryPumpingModel,
    SecondaryCycleModel,
    ShieldThermalHeatUse,
    SolenoidSwitchModel,
    TFNuclearHeatingModel,
    TFSuperconductorModel,
    TFWindingPackTurnModel,
)
from bluemira.codes.process.template_builder import PROCESSTemplateBuilder
from bluemira.utilities.tools import compare_dicts


def extract_warning(caplog):
    result = [line for message in caplog.messages for line in message.split(os.linesep)]
    result = " ".join(result)
    caplog.clear()
    return result


class TestPROCESSTemplateBuilder:
    def test_no_error_on_nothing(self, caplog):
        t = PROCESSTemplateBuilder()
        _ = t.make_inputs()
        assert len(caplog.messages) == 0

    def test_warn_on_optimisation_with_no_objective(self, caplog):
        t = PROCESSTemplateBuilder()
        t.set_optimisation_algorithm(PROCESSOptimisationAlgorithm.VMCON)
        _ = t.make_inputs()
        assert len(caplog.messages) == 1
        warning = extract_warning(caplog)
        assert "You are running in optimisation mode, but" in warning

    @pytest.mark.parametrize(
        "objective",
        [Objective.FUSION_GAIN_PULSE_LENGTH, Objective.MAJOR_RADIUS_PULSE_LENGTH],
    )
    def test_error_on_maxmin_objective(self, objective):
        t = PROCESSTemplateBuilder()
        with pytest.raises(
            ValueError, match="can only be used as a minimisation objective"
        ):
            t.set_maximisation_objective(objective)

    @pytest.mark.parametrize("bad_name", ["spoon", "aaaaaaaaaaaaa"])
    def test_error_on_bad_itv_name(self, bad_name):
        t = PROCESSTemplateBuilder()
        with pytest.raises(ValueError, match="There is no iteration variable:"):
            t.add_variable(bad_name, 3.14159)

    @pytest.mark.parametrize("bad_name", ["spoon", "aaaaaaaaaaaaa"])
    def test_error_on_adjusting_bad_variable(self, bad_name):
        t = PROCESSTemplateBuilder()
        with pytest.raises(ValueError, match="There is no iteration variable:"):
            t.adjust_variable(bad_name, 3.14159)

    def test_warn_on_repeated_constraint(self, caplog):
        t = PROCESSTemplateBuilder()
        t.add_constraint(Constraint.BETA_CONSISTENCY)
        t.add_constraint(Constraint.BETA_CONSISTENCY)
        assert len(caplog.messages) == 1
        warning = extract_warning(caplog)
        assert "is already in" in warning

    def test_warn_on_repeated_itv(self, caplog):
        t = PROCESSTemplateBuilder()
        t.add_variable("dr_bore", 2.0)
        t.add_variable("dr_bore", 3.0)
        assert len(caplog.messages) == 1
        warning = extract_warning(caplog)
        assert "Iteration variable 'dr_bore' is already" in warning

    def test_warn_on_adjusting_nonexistent_variable(self, caplog):
        t = PROCESSTemplateBuilder()
        t.adjust_variable("dr_bore", 2.0)
        assert len(caplog.messages) == 1
        warning = extract_warning(caplog)
        assert "Iteration variable 'dr_bore' is not in" in warning
        assert "dr_bore" in t.variables

    def test_warn_on_missing_input_constraint(self, caplog):
        t = PROCESSTemplateBuilder()
        t.add_constraint(Constraint.NWL_UPPER_LIMIT)
        t.add_variable("aspect", 3.1)
        t.add_variable("bt", 5.0)
        t.add_variable("rmajor", 9.0)
        t.add_variable("te", 12.0)
        t.add_variable("dene", 8.0e19)
        _ = t.make_inputs()
        assert len(caplog.messages) == 1
        warning = extract_warning(caplog)
        assert "requires inputs 'pflux_fw_neutron_max_mw'" in warning

    def test_warn_on_missing_itv_constraint(self, caplog):
        t = PROCESSTemplateBuilder()
        t.add_constraint(Constraint.RADIAL_BUILD_CONSISTENCY)
        _ = t.make_inputs()
        assert len(caplog.messages) == 1
        assert "requires iteration" in extract_warning(caplog)

    def test_no_warn_on_missing_itv_constraint_but_as_input(self, caplog):
        t = PROCESSTemplateBuilder()
        t.add_constraint(Constraint.NWL_UPPER_LIMIT)
        t.add_variable("bt", 5.0)
        t.add_variable("rmajor", 9.0)
        t.add_variable("te", 12.0)
        t.add_variable("dene", 8.0e19)
        t.add_input_value("pflux_fw_neutron_max_mw", 8.0)
        t.add_input_value("aspect", 3.1)
        _ = t.make_inputs()
        assert len(caplog.messages) == 0

    def test_warn_on_missing_input_model(self, caplog):
        t = PROCESSTemplateBuilder()
        t.set_model(PlasmaGeometryModel.CREATE_A_M_S)
        _ = t.make_inputs()
        assert len(caplog.messages) == 1
        assert "requires inputs" in extract_warning(caplog)

    def test_automatic_fvalue_itv(self):
        t = PROCESSTemplateBuilder()
        t.set_minimisation_objective(Objective.MAJOR_RADIUS)
        t.add_constraint(Constraint.NET_ELEC_LOWER_LIMIT)
        assert "fp_plant_electric_net_required_mw" in t.variables

    def test_warn_on_overwrite_value(self, caplog):
        t = PROCESSTemplateBuilder()
        t.add_input_value("dummy", 1.0)
        t.add_input_value("dummy", 2.0)
        assert len(caplog.messages) == 1
        assert "Over-writing" in extract_warning(caplog)

    def test_warn_on_overwrite_model(self, caplog):
        t = PROCESSTemplateBuilder()
        t.set_model(PlasmaGeometryModel.CREATE_A_M_S)
        t.set_model(PlasmaGeometryModel.FIESTA_100)
        assert len(caplog.messages) == 1
        assert "Over-writing" in extract_warning(caplog)

    def test_impurity_shenanigans(self):
        t = PROCESSTemplateBuilder()
        t.add_impurity(Impurities.Xe, 0.5)
        assert t.fimp[12] == pytest.approx(0.5, rel=0, abs=EPS)
        t.add_variable("fimp(13)", 0.6)
        assert t.fimp[12] == pytest.approx(0.6, rel=0, abs=EPS)
        t.add_impurity(Impurities.Xe, 0.4)
        assert t.fimp[12] == pytest.approx(0.4, rel=0, abs=EPS)

    def test_input_appears_in_dat(self):
        t = PROCESSTemplateBuilder()
        t.add_input_value("dx_tf_wp_insulation", 1000.0)
        assert t.values["dx_tf_wp_insulation"] == pytest.approx(1000.0, rel=0, abs=EPS)
        data = t.make_inputs()
        assert data.to_invariable()["dx_tf_wp_insulation"]._value == pytest.approx(
            1000.0, rel=0, abs=EPS
        )

    def test_inputs_appear_in_dat(self):
        t = PROCESSTemplateBuilder()
        t.add_input_values({"dx_tf_wp_insulation": 1000.0, "dr_bore": 1000})
        assert t.values["dx_tf_wp_insulation"] == pytest.approx(1000.0, rel=0, abs=EPS)
        assert t.values["dr_bore"] == pytest.approx(1000.0, rel=0, abs=EPS)
        data = t.make_inputs().to_invariable()
        assert data["dx_tf_wp_insulation"]._value == pytest.approx(
            1000.0, rel=0, abs=EPS
        )
        assert data["dr_bore"]._value == pytest.approx(1000.0, rel=0, abs=EPS)


def read_indat(filename):
    from process.io.in_dat import InDat  # noqa: PLC0415

    naughties = ["runtitle", "pulsetimings"]
    data = InDat(filename=filename).data
    return {k: v for k, v in data.items() if k not in naughties}


@pytest.mark.private
@pytest.mark.skipif(not ENABLED, reason="PROCESS is not installed on the system.")
class TestInDatOneForOne:
    @classmethod
    def setup_class(cls):
        fp = Path(
            try_get_bluemira_private_data_root(),
            "process/DEMO_2023_TEMPLATE_TEST_IN.DAT",
        )

        cls.true_data = read_indat(fp)

        template_builder = PROCESSTemplateBuilder()
        template_builder.set_optimisation_algorithm(PROCESSOptimisationAlgorithm.VMCON)
        template_builder.set_optimisation_numerics(maxiter=1000, tolerance=1e-8)

        template_builder.set_minimisation_objective(Objective.MAJOR_RADIUS)

        for constraint in (
            Constraint.BETA_CONSISTENCY,
            Constraint.GLOBAL_POWER_CONSISTENCY,
            Constraint.DENSITY_UPPER_LIMIT,
            Constraint.NWL_UPPER_LIMIT,
            Constraint.RADIAL_BUILD_CONSISTENCY,
            Constraint.BURN_TIME_LOWER_LIMIT,
            Constraint.LH_THRESHHOLD_LIMIT,
            Constraint.NET_ELEC_LOWER_LIMIT,
            Constraint.BETA_UPPER_LIMIT,
            Constraint.CS_EOF_DENSITY_LIMIT,
            Constraint.CS_BOP_DENSITY_LIMIT,
            Constraint.PINJ_UPPER_LIMIT,
            Constraint.TF_CASE_STRESS_UPPER_LIMIT,
            Constraint.TF_JACKET_STRESS_UPPER_LIMIT,
            Constraint.TF_JCRIT_RATIO_UPPER_LIMIT,
            Constraint.TF_DUMP_VOLTAGE_UPPER_LIMIT,
            Constraint.TF_CURRENT_DENSITY_UPPER_LIMIT,
            Constraint.TF_T_MARGIN_LOWER_LIMIT,
            Constraint.CS_T_MARGIN_LOWER_LIMIT,
            Constraint.CONFINEMENT_RATIO_LOWER_LIMIT,
            Constraint.DUMP_TIME_LOWER_LIMIT,
            Constraint.PSEPB_QAR_UPPER_LIMIT,
            Constraint.CS_STRESS_UPPER_LIMIT,
            Constraint.DENSITY_PROFILE_CONSISTENCY,
            Constraint.CS_FATIGUE,
        ):
            template_builder.add_constraint(constraint)

        # Variable vector values and bounds
        template_builder.add_variable("bt", 5.3292, upper_bound=20.0)
        template_builder.add_variable("rmajor", 8.8901, upper_bound=13)
        template_builder.add_variable("te", 12.33, upper_bound=150.0)
        template_builder.add_variable("beta", 3.1421e-2)
        template_builder.add_variable("dene", 7.4321e19)
        template_builder.add_variable("q95", 3.5, lower_bound=3.5)
        template_builder.add_variable("p_hcd_primary_extra_heat_mw", 50.0)
        template_builder.add_variable("f_nd_alpha_electron", 6.8940e-02)
        template_builder.add_variable("dr_bore", 2.3322, lower_bound=0.1)
        template_builder.add_variable("dr_cs", 0.55242, lower_bound=0.1)
        template_builder.add_variable("dx_tf_turn_steel", 8.0e-3, lower_bound=8.0e-3)
        template_builder.add_variable("dr_tf_nose_case", 0.52465)
        template_builder.add_variable("dr_tf_inboard", 1.2080)
        template_builder.add_variable(
            "dr_cs_tf_gap", 0.05, lower_bound=0.05, upper_bound=0.1
        )
        template_builder.add_variable("dr_shld_vv_gap_inboard", 0.02, lower_bound=0.02)
        template_builder.add_variable("f_a_cs_steel", 0.57875)
        template_builder.add_variable("j_cs_flat_top_end", 2.0726e07)
        template_builder.add_variable(
            "c_tf_turn", 6.5e4, lower_bound=6.0e4, upper_bound=9.0e4
        )
        template_builder.add_variable("tdmptf", 2.5829e01)
        template_builder.add_variable("fimp(13)", 3.573e-04)

        # Some constraints require multiple f-values, but they are getting
        # ridding of those, so no fancy mechanics for now...
        template_builder.add_variable(
            "fcutfsu", 0.80884, lower_bound=0.5, upper_bound=0.94
        )
        template_builder.add_variable("f_j_cs_start_pulse_end_flat_top", 0.93176)
        template_builder.add_variable("f_c_plasma_non_inductive", 0.39566)
        template_builder.add_variable("fncycle", 1.0)

        # Modified f-values and bounds w.r.t. defaults
        template_builder.adjust_variable("fne0", 0.6, upper_bound=0.95)
        template_builder.adjust_variable("fdene", 1.2, upper_bound=1.2)
        template_builder.adjust_variable(
            "fl_h_threshold", 0.833, lower_bound=0.833, upper_bound=0.909
        )
        template_builder.adjust_variable("ft_burn", 1.0, upper_bound=1.0)

        # Modifying the initial variable vector to improve convergence
        template_builder.adjust_variable("fp_plant_electric_net_required_mw", 1.0)
        template_builder.adjust_variable("fstrcase", 1.0)
        template_builder.adjust_variable("ftmargtf", 1.0)
        template_builder.adjust_variable("ftmargoh", 1.0)
        template_builder.adjust_variable("falpha_energy_confinement", 1.0)
        template_builder.adjust_variable("fjohc", 0.57941, upper_bound=1.0)
        template_builder.adjust_variable("fjohc0", 0.53923, upper_bound=1.0)
        template_builder.adjust_variable("foh_stress", 1.0)
        template_builder.adjust_variable("fbeta_max", 0.48251)
        template_builder.adjust_variable("fpflux_fw_neutron_max_mw", 0.131)
        template_builder.adjust_variable("fmaxvvstress", 1.0)
        template_builder.adjust_variable("fpsepbqar", 1.0)
        template_builder.adjust_variable("fvdump", 1.0)
        template_builder.adjust_variable("fstrcond", 0.92007)
        template_builder.adjust_variable("fiooic", 0.63437, upper_bound=1.0)
        template_builder.adjust_variable("fjprot", 1.0)

        # Set model switches
        for model_choice in (
            BootstrapCurrentScalingLaw.SAUTER,
            ConfinementTimeScalingLaw.IPB98_Y2_H_MODE,
            PlasmaCurrentScalingLaw.ITER_REVISED,
            BetaNormMaxModel.WESSON,
            PlasmaProfileModel.WESSON,
            AlphaJModel.WESSON,
            PlasmaPedestalModel.PEDESTAL_GW,
            PlasmaNullConfigurationModel.SINGLE_NULL,
            BetaLimitModel.THERMAL,
            DensityLimitModel.GREENWALD,
            AlphaPressureModel.WARD,
            PlasmaGeometryModel.CREATE_A_M_S,
            PowerFlowModel.SIMPLE,
            ShieldThermalHeatUse.LOW_GRADE_HEAT,
            SecondaryCycleModel.INPUT,
            CurrentDriveEfficiencyModel.ECRH_UI_GAM,
            OperationModel.PULSED,
            PFSuperconductorModel.NBTI,
            SolenoidSwitchModel.SOLENOID,
            CSSuperconductorModel.NB3SN_WST,
            TFSuperconductorModel.NB3SN_WST,
            TFWindingPackTurnModel.INTEGER_TURN,
            PrimaryPumpingModel.PRESSURE_DROP_INPUT,
            TFNuclearHeatingModel.INPUT,
            CostModel.TETRA_1990,
            AvailabilityModel.INPUT,
            OutputCostsSwitch.NO,
        ):
            template_builder.set_model(model_choice)

        template_builder.add_impurity(Impurities.H, 1.0)
        template_builder.add_impurity(Impurities.He, 0.1)
        template_builder.add_impurity(Impurities.W, 5.0e-5)

        # Set fixed input values
        template_builder.add_input_values({
            # Undocumented danger stuff
            "i_blanket_type": 1,
            "lsa": 2,
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
            "beta_norm_max": 3.0,
            # Plasma impurity stuff
            "radius_plasma_core_norm": 0.75,
            "coreradiationfraction": 0.6,
            # Important stuff
            "p_plant_electric_net_required_mw": 500.0,
            "t_burn_min": 7.2e3,
            "sig_tf_case_max": 5.8e8,
            "sig_tf_wp_max": 5.8e8,
            "alstroh": 6.6e8,
            "psepbqarmax": 9.2,
            "aspect": 3.1,
            "m_s_limit": 0.1,
            "triang": 0.5,
            "q0": 1.0,
            "f_sync_reflect": 0.6,
            "plasma_res_factor": 0.66,
            "ejima_coeff": 0.3,
            "hfact": 1.1,
            "life_dpa": 70.0,
            # Radial build inputs
            "dr_tf_shld_gap": 0.05,
            "dr_vv_inboard": 0.3,
            "dr_shld_inboard": 0.3,
            "dr_shld_blkt_gap": 0.02,
            "dr_blkt_inboard": 0.755,
            "dr_fw_plasma_gap_inboard": 0.225,
            "dr_fw_plasma_gap_outboard": 0.225,
            "dr_blkt_outboard": 0.982,
            "dr_vv_outboard": 0.3,
            "dr_shld_outboard": 0.8,
            "dr_cryostat": 0.15,
            "gapomin": 0.2,
            # Vertical build inputs
            "dz_vv_upper": 0.3,
            "dz_shld_vv_gap": 0.05,
            "dz_shld_upper": 0.3,
            "dz_divertor": 0.621,
            "dz_vv_lower": 0.3,
            # HCD inputs
            "p_hcd_injected_max": 51.0,
            "eta_cd_norm_ecrh": 0.3,
            "eta_ecrh_injector_wall_plug": 0.4,
            "f_c_plasma_bootstrap_max": 0.99,
            # BOP inputs
            "eta_turbine": 0.375,
            "eta_coolant_pump_electric": 0.87,
            "etaiso": 0.9,
            "vfshld": 0.6,
            "t_between_pulse": 0.0,
            "t_precharge": 500.0,
            # CS / PF coil inputs
            "t_crack_vertical": 0.4e-3,
            "fcuohsu": 0.7,
            "f_z_cs_tf_internal": 0.9,
            "rpf2": -1.825,
            "c_pf_coil_turn_peak_input": [
                4.22e4,
                4.22e4,
                4.22e4,
                4.22e4,
                4.3e4,
                4.3e4,
                4.3e4,
                4.3e4,
            ],
            "i_pf_location": [2, 2, 3, 3],
            "n_pf_coils_in_group": [1, 1, 2, 2],
            "n_pf_coil_groups": 4,
            "j_pf_coil_wp_peak": [
                1.1e7,
                1.1e7,
                6.0e6,
                6.0e6,
                8.0e6,
                8.0e6,
                8.0e6,
                8.0e6,
            ],
            # TF coil inputs
            "n_tf_coils": 16,
            "dr_tf_plasma_case": 0.06,
            "dx_tf_side_case": 0.05,
            "ripmax": 0.6,
            "dia_tf_turn_coolant_channel": 0.01,
            "tftmp": 4.75,
            "dx_tf_turn_insulation": 2.0e-3,
            "dx_tf_wp_insulation": 0.008,
            # "dx_tf_wp_insertion_gap": 0.01,
            "tmargmin": 1.5,
            "f_a_tf_turn_cable_space_extra_void": 0.3,
            "n_pancake": 20,
            "n_layer": 10,
            "qnuc": 1.292e4,
            "vdalw": 10.0,
            # Inputs we don't care about but must specify
            "cfactr": 0.75,  # Ha!
            "kappa": 1.848,  # Should be overwritten
            "pflux_fw_neutron_max_mw": 8.0,  # Should never get even close to this
            "tlife": 40.0,
            "abktflnc": 15.0,
            "adivflnc": 20.0,
            # For sanity...
            "pflux_div_heat_load_max_mw": 10,
            "prn1": 0.4,
            "b_tf_inboard_max": 11.2,
            "fp_fusion_total_max_mw": 1.0,
            "fb_tf_inboard_max": 1.0,
            "ibkt_life": 1,
            "fkzohm": 1.0245,
            "dintrt": 0.0,
            "fcap0": 1.15,
            "fcap0cp": 1.06,
            "fcontng": 0.15,
            "fcr0": 0.065,
            "fkind": 1.0,
            "ifueltyp": 1,
            "discount_rate": 0.06,
            "bkt_life_csf": 1,
            "ucblvd": 280.0,
            "ucdiv": 5e5,
            "ucme": 3.0e8,
            # Suspicous stuff
            "zref": [3.6, 1.2, 1.0, 2.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            "fp_hcd_injected_total_mw": 1.0,
        })

        cls.template = template_builder.make_inputs().to_invariable()

    def test_indat_bounds_the_same(self):
        true_bounds = self.true_data.pop("bounds").get_value
        new_bounds = self.template.pop("bounds").get_value

        # Make everything floats for easier comparison
        for k in true_bounds:
            for kk in true_bounds[k]:
                true_bounds[k][kk] = float(true_bounds[k][kk])
        for k in new_bounds:
            for kk in new_bounds[k]:
                new_bounds[k][kk] = float(new_bounds[k][kk])
        assert compare_dicts(true_bounds, new_bounds)

    def test_indat_constraints(self):
        true_cons = self.true_data.pop("icc").get_value
        new_cons = self.template.pop("icc").get_value

        np.testing.assert_allclose(sorted(true_cons), sorted(new_cons))

    def test_indat_variables(self):
        true_vars = self.true_data.pop("ixc").get_value
        new_vars = self.template.pop("ixc").get_value

        np.testing.assert_allclose(sorted(true_vars), sorted(new_vars))

    def test_inputs_same(self):
        for k in self.true_data:
            if not isinstance(self.true_data[k].get_value, list | dict):
                assert np.allclose(
                    self.true_data[k].get_value, self.template[k].get_value
                )
            elif isinstance(self.true_data[k].get_value, dict):
                assert compare_dicts(
                    self.true_data[k].get_value,
                    self.template[k]._value,
                    almost_equal=True,
                )
            else:
                assert not set(self.true_data[k].get_value) - set(
                    self.template[k].get_value
                )

    def test_no_extra_inputs(self):
        vals = [k for k in self.template if k not in self.true_data]
        assert len(vals) == 0
