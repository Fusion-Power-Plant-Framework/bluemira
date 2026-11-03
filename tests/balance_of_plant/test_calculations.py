# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
import numpy as np
import pytest

from bluemira.balance_of_plant.calculations import (
    H2O_pumping,
    He_pumping,
    cryo_power,
    superheated_rankine,
)


class TestCryoPower:
    def test_cryo_power_basic(self):
        """Test basic cryogenic power calculation with expected values"""
        result = cryo_power(
            s_tf=2.5,
            m_cold=300.0,
            nucl_heating=100000.0,
            e_pf_max=8e6,
            t_pulse=1.0,
            tf_current=10000.0,
            n_TF=4,
        )

        # Expected based on formula: qss = 4.3e-4 * m_cold + 2 * s_tf
        expected_qss = (4.3e-4 * 300) + (2 * 2.5)
        expected_qac = 8e3 / 1.0
        expected_qcl = 13.6e-3 * 4 * 10000

        # Total should be (1+fmisc)*(nucl_heating + qcl + qac + qss)
        total_expected = (1 + 0.45) * (
            100000 + expected_qcl + expected_qac + expected_qss
        )

        assert abs(result - total_expected) < 1e-6

    def test_cryo_power_with_zero_parameters(self):
        """Test cryo power with minimal parameters"""
        result = cryo_power(
            s_tf=0.0,
            m_cold=0.0,
            nucl_heating=0.0,
            e_pf_max=0.0,
            t_pulse=1.0,
            tf_current=0.0,
            n_TF=1,
        )

        # Should return 0 if all inputs are zero
        assert result == 0


class TestHePumping:
    def test_he_pumping_basic(self):
        """Test basic helium pumping power calculation"""
        p_in, p_out = 5e6, 3e6  # Pa
        t_in, t_out = 290.0, 310.0  # K
        blanket_power = 8e6  # W
        eta_isen, eta_el = 0.75, 0.9

        p_pump_is, p_pump_el = He_pumping(
            p_in, p_out, t_in, t_out, blanket_power, eta_isen, eta_el
        )

        # Expected calculation:
        d_temp = t_out - t_in
        t_comp_inlet = t_in / ((p_in / p_out) ** (2 / 5))
        f_pump = (t_comp_inlet / (eta_isen * d_temp)) * ((p_in / p_out) ** (2 / 5) - 1)

        # Validate calculation matches expected result
        assert abs(p_pump_is - f_pump * blanket_power / (1 - f_pump)) < 1e-6

    def test_he_pumping_invalid_input(self):
        """Test that invalid inputs are caught"""
        with pytest.raises(ValueError, match="All parameters must be non-negative"):
            He_pumping(-1e6, -2e6, -300.0, -400.0, 8e6, -0.5, 0.9)

    @pytest.mark.parametrize(
        ("t_in", "t_out", "blanket_power", "p_in", "p_out", "eta_isen", "eta_el"),
        [
            (400.0, 450.0, 8e6, 7e6, 4e6, 0.8, 0.9),
            (300.0, 350.0, 12e6, 7e6, 4e6, 0.8, 0.9),
            # Edge case small pressure drop
            (300.0, 450.0, 8e6, 2e7, 2e7 + 100, 0.75, 0.9),
        ],
    )
    def test_he_pumping_returns_positive_values(
        self, t_in, t_out, blanket_power, p_in, p_out, eta_isen, eta_el
    ):
        """Parameterized test for different temperature ranges"""

        p_pump_is, p_pump_el = He_pumping(
            p_in, p_out, t_in, t_out, blanket_power, eta_isen, eta_el
        )

        # Should still return reasonable values despite small pressure drop
        assert p_pump_is > 0
        assert p_pump_el > 0


class TestH2OPumping:
    @pytest.mark.parametrize(
        ("p_blanket", "f_pump", "eta_isen", "eta_el"),
        [(1e7, 0.05, 0.7, 0.8), (5e6, 1e-4, 0.8, 0.9)],
    )
    def test_h2o_pumping_out_vs_in(self, p_blanket, f_pump, eta_isen, eta_el):
        p_pump_is, p_pump_el = H2O_pumping(
            p_blanket=p_blanket, f_pump=f_pump, eta_isen=eta_isen, eta_el=eta_el
        )

        # Expected: f_pump /= eta_isen => 0.05 / 0.7 = ~0.0714
        expected_f_pump = f_pump / eta_isen

        # Calculate expected result:
        p_expected_is = (expected_f_pump * p_blanket) / (1 - expected_f_pump)

        assert abs(p_pump_is - p_expected_is) < 1e-6


class TestSuperheatedRankine:
    @pytest.mark.parametrize(
        ("blanket_power", "div_power", "bb_outlet_temp", "delta_t_turbine"),
        [(8e6, 2e6, 900, 150), (8e6, 2e6, 915, 150)],
    )
    def test_superheated_rankine_efficiency(
        self, blanket_power, div_power, bb_outlet_temp, delta_t_turbine
    ):
        """Test superheated Rankine cycle efficiency"""

        eta = superheated_rankine(
            blanket_power, div_power, bb_outlet_temp, delta_t_turbine
        )

        # Expected calculation:
        t_turb = bb_outlet_temp - delta_t_turbine
        f_lgh = div_power / (blanket_power + div_power)
        delta_eta = 0.339 * f_lgh

        expected_eta = 0.1802 * np.log(t_turb) - 0.7823 - delta_eta

        assert abs(eta - expected_eta) < 1e-6

    # def test_superheated_rankine_at_boundary(self):
    #     """Test at the boundary conditions"""

    #     result = superheated_rankine(
    #         blanket_power=8e6, div_power=2e6, bb_outlet_temp=915.0, delta_t_turbine=150
    #     )

    #     # Should be valid and return expected efficiency at upper limit
    #     assert abs(result - (0.1802 * np.log(765) - 0.7823)) < 1e-6

    def test_superheated_rankine_invalid_temperature(self, caplog):
        """Test that temperature out of valid range triggers warning"""

        result = superheated_rankine(
            blanket_power=8e6,
            div_power=2e6,
            bb_outlet_temp=500.0,
            delta_t_turbine=100,
        )

        assert "BoP turbine inlet temperature outside range of validity." in caplog.text
        # Should return reasonable efficiency despite warning
        assert abs(result - (0.1802 * np.log(400) - 0.7823)) < 1e-6

    def test_superheated_rankine_valid_temperature(self, caplog):
        """Test within valid temperature range"""

        result = superheated_rankine(
            blanket_power=8e6,
            div_power=2e6,
            bb_outlet_temp=900.0,
            delta_t_turbine=150,
        )

        assert (
            "BoP turbine inlet temperature outside range of validity." not in caplog.text
        )
        assert len(caplog.text) == 0


# Optional: Add integration tests if needed
@pytest.mark.skip(reason="Integration testing requires more data")
def test_cryo_power_integration():
    """Test cryo power calculation with realistic parameters"""
    # This would be tested with actual project-specific inputs
