# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
from pathlib import Path

import pytest
from matplotlib.pyplot import Axes

from bluemira.base.error import BluemiraError
from bluemira.base.file import get_bluemira_path
from bluemira.equilibria.analysis import (
    EqAnalysis,
    MultiEqAnalysis,
    select_eq,
    select_multi_eqs,
)
from bluemira.equilibria.diagnostics import (
    DivLegsToPlot,
    EqDiagnosticOptions,
    EqPlotMask,
    EqSubplots,
    FixedOrFree,
    FluxSurfaceType,
    PsiPlotType,
)
from bluemira.equilibria.equilibrium import Equilibrium, FixedPlasmaEquilibrium
from bluemira.equilibria.flux_surfaces import CoreResults
from bluemira.equilibria.physics import EqSummary
from bluemira.geometry.coordinates import Coordinates

TEST_PATH = get_bluemira_path("equilibria/test_data", subfolder="tests")
# MAST-like-DN
masty_path = Path(TEST_PATH, "SH_test_file.json")
# DEMO-like-SN
single_demoish_path = Path(TEST_PATH, "eqref_OOB.json")
# DEMO-like-DN
double_demoish_path = Path(TEST_PATH, "DN-DEMO_eqref.json")


class TestEqAnalysis:
    """
    Tests for EqAnalysis.
    """

    @classmethod
    def setup_class(cls):
        cls.single_demoish_eq = select_eq(single_demoish_path, from_cocos=7)
        cls.single_demoish_eq_fixed = select_eq(
            single_demoish_path, from_cocos=7, fixed_or_free=FixedOrFree.FIXED
        )
        cls.double_demoish_eq = select_eq(double_demoish_path)
        cls.ref_free = cls.single_demoish_eq
        cls.ref_fixed = cls.single_demoish_eq
        cls.diag_ops_1 = EqDiagnosticOptions(
            psi_diff=PsiPlotType.PSI_ABS_DIFF,
            split_psi_plots=EqSubplots.XZ_COMPONENT_PSI,
            plot_mask=EqPlotMask.IN_LCFS,
        )
        cls.diag_ops_2 = EqDiagnosticOptions(
            psi_diff=PsiPlotType.PSI_ABS_DIFF,
            split_psi_plots=EqSubplots.XZ,
            plot_mask=EqPlotMask.IN_LCFS,
        )

    def test_select_eq(self):
        """Test correct class is returned."""
        assert isinstance(self.single_demoish_eq, Equilibrium)
        assert isinstance(self.single_demoish_eq_fixed, FixedPlasmaEquilibrium)

    def test_plotting(self):
        "Test plots return expected axes etc."
        # Compare free DN to free SN
        eq_analysis_1 = EqAnalysis(
            input_eq=self.double_demoish_eq,
            reference_eq=self.ref_free,
            diag_ops=self.diag_ops_1,
        )
        # Compare free DN to fixed SN
        eq_analysis_2 = EqAnalysis(
            input_eq=self.double_demoish_eq,
            reference_eq=self.ref_fixed,
            diag_ops=self.diag_ops_1,
        )
        # Compare free DN to free SN, with psi components splitting for plots
        eq_analysis_3 = EqAnalysis(
            input_eq=self.double_demoish_eq,
            reference_eq=self.ref_free,
            diag_ops=self.diag_ops_2,
        )

        # Target for legflux plotting.
        target = "lower_outer"
        target_coords = Coordinates({"x": [10, 11], "z": [-7.5, -7.5]})

        plot_1 = eq_analysis_1.plot()
        plot_1b = eq_analysis_2.plot()
        plot_2 = eq_analysis_1.plot_field()
        plot_2b = eq_analysis_2.plot_field()
        plot_3 = eq_analysis_1.plot_profiles()
        plot_3b = eq_analysis_2.plot_profiles()
        plot_4 = eq_analysis_1.plot_equilibria_with_profiles()
        plot_4b = eq_analysis_2.plot_equilibria_with_profiles()
        plot_5 = eq_analysis_1.plot_eq_core_mag_axis()
        plot_6 = eq_analysis_1.plot_compare_profiles()
        plot_6b = eq_analysis_2.plot_compare_profiles()
        plot_7_res, plot_7_ax = eq_analysis_1.plot_eq_core_analysis()
        plot_8 = eq_analysis_1.physics_info_table()
        plot_9 = eq_analysis_1.plot_compare_separatrix()
        plot_9b = eq_analysis_2.plot_compare_separatrix()
        plot_10 = eq_analysis_1.plot_compare_psi()
        plot_10b = eq_analysis_2.plot_compare_psi()
        plot_11 = eq_analysis_3.plot_compare_psi()
        plot_12 = eq_analysis_1.plot_target_flux(target, target_coords)

        assert isinstance(plot_1, Axes)
        assert isinstance(plot_2[0], Axes)
        assert len(plot_2) == 2
        assert isinstance(plot_3, Axes)
        assert isinstance(plot_4[0], Axes)
        assert len(plot_4) == 2
        assert isinstance(plot_5[0], Axes)
        assert isinstance(plot_6[0, 0], Axes)
        assert isinstance(plot_7_ax[0], Axes)
        assert isinstance(plot_7_res, CoreResults)
        assert len(plot_7_ax) == 18
        assert len(plot_7_res.__dict__.items()) == 17
        assert isinstance(plot_8, EqSummary)
        assert isinstance(plot_9, Axes)
        assert plot_10 is None
        assert plot_11 is None
        assert isinstance(plot_12, Axes)
        assert isinstance(plot_1b, Axes)
        assert isinstance(plot_2b[0], Axes)
        assert len(plot_2b) == 2
        assert isinstance(plot_3b, Axes)
        assert isinstance(plot_4b[0], Axes)
        assert len(plot_4b) == 2
        assert isinstance(plot_6b[0, 0], Axes)
        assert isinstance(plot_9b, Axes)
        assert plot_10b is None


class TestMultiEqAnalysis:
    """
    Tests for MultiEqAnalysis.
    """

    @classmethod
    def setup_class(cls):
        paths = [masty_path, double_demoish_path, single_demoish_path]
        equilibrium_names = ["Little DN", "Big DN", "Big SN"]
        cls.equilibria_dictionary = select_multi_eqs(
            paths, equilibrium_names=equilibrium_names, from_cocos=[3, 3, 7]
        )
        cls.multi_analysis = MultiEqAnalysis(equilibria_dict=cls.equilibria_dictionary)
        cls.pfb_masty = Coordinates({
            "x": [1.75, 1.75, 0.0, 0.0, 1.75],
            "z": [-1.75, 1.75, 1.75, -1.75, -1.75],
        })
        cls.pfb_demoish = Coordinates({
            "x": [14.5, 14.5, 5.75, 5.75, 14.5],
            "z": [-7.5, 7.5, 7.5, -7.5, -7.5],
        })

    def test_plotting(self):
        core_res, ax1 = self.multi_analysis.plot_core_physics()
        ax2 = self.multi_analysis.plot_compare_profiles()
        ax3 = self.multi_analysis.plot_compare_flux_surfaces()
        ax4 = self.multi_analysis.plot_compare_flux_surfaces(
            flux_surface=FluxSurfaceType.PSI_NORM, psi_norm=1.05
        )
        assert isinstance(core_res[0], CoreResults)
        assert isinstance(ax1[0], Axes)
        assert len(ax1) == 18
        assert isinstance(ax2[0], Axes)
        assert len(ax2) == 6
        assert isinstance(ax3, Axes)
        assert isinstance(ax4, Axes)

    @pytest.mark.parametrize("legs_to_plot", [DivLegsToPlot.ALL, DivLegsToPlot.LW])
    def test_div_info_plot(self, legs_to_plot):
        ax = self.multi_analysis.plot_divertor_length_angle(
            plasma_facing_boundary_list=[
                self.pfb_masty,
                self.pfb_demoish,
                self.pfb_demoish,
            ],
            legs_to_plot=legs_to_plot,
        )
        ax_num = 2 if legs_to_plot in DivLegsToPlot.PAIR else 4
        assert len(ax[0]) == ax_num
        assert len(ax[1]) == ax_num

    def test_div_info_plot_wrong_input(self, legs_to_plot=DivLegsToPlot.UP):
        with pytest.raises(BluemiraError):
            _ = self.multi_analysis.plot_divertor_length_angle(
                plasma_facing_boundary_list=[
                    self.pfb_masty,
                    self.pfb_demoish,
                    self.pfb_demoish,
                ],
                legs_to_plot=legs_to_plot,
            )
