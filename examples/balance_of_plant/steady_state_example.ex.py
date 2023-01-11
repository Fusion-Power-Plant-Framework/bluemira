# %% nbsphinx="hidden"
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

"""
Simple example of a 0-D steady-state balance of plant view.
"""

# %%
import matplotlib.pyplot as plt

from bluemira.balance_of_plant.steady_state import (
    BalanceOfPlantModel,
    H2OPumping,
    HePumping,
    NeutronPowerStrategy,
    ParasiticLoadStrategy,
    PredeterminedEfficiency,
    RadChargedPowerStrategy,
    SuperheatedRankine,
)

# %% [markdown]
#
# # Simple Example of a 0-D steady-state balance of plant.
#
# Let's set up a typical power balance model. We start by specifying some parameters we
# want to use.

# %%
# fmt: off
default_params = {
    'P_fus_DT': 1995,
    'P_fus_DD': 5,
    'P_rad': 400,
    'P_hcd_ss': 50,
    'P_hcd_ss_el': 150,
}
# fmt: on

# %% [markdown]
#
# We then weed to specify how we're going to treat the neutrons, radiation, and charged
# particle loads. We do this by specifying "strategies".

# %%
neutron_power_strat = NeutronPowerStrategy(
    f_blanket=0.9,
    f_divertor=0.05,
    f_vessel=0.04,
    f_other=0.01,
    energy_multiplication=1.35,
    decay_multiplication=1.0175,
)
rad_sep_strat = RadChargedPowerStrategy(
    f_core_rad_fw=0.9,
    f_sol_rad=0.75,
    f_sol_rad_fw=0.8,
    f_sol_ch_fw=0.8,
    f_fw_aux=0.09,
)

# %% [markdown]
#
# Now we specify how the in-vessel components are being cooled, to calculate pumping
# powers, and the balance of plant cycle design.

# %%
blanket_pump_strat = HePumping(
    8e6, 7.5e6, 300, 500, eta_isentropic=0.9, eta_electric=0.87
)
bop_cycle = SuperheatedRankine(bb_t_out=500 + 273.15, delta_t_turbine=20)
divertor_pump_strat = H2OPumping(f_pump=0.05, eta_isentropic=0.99, eta_electric=0.87)

# %% [markdown]
#
# Maybe we don't have any good models to estimate some of the other parasitic loads. We
# can set up a simple scaling with respect to a known reference point, by sub-classing
# from the ABC and specifying some calculation in the `calculate` method.

# %%


class EUDEMOReferenceParasiticLoadStrategy(ParasiticLoadStrategy):
    """
    One way of defining the parasitic loads w.r.t. a known reference point.
    """

    def __init__(self):
        self.p_fusion_ref = 2037
        self.p_cryo = 44
        self.p_mag = 44
        self.p_t_plant = 15.5
        self.p_other = 31

    def calculate(self, p_fusion):
        """
        Because we were told to do this. Nobody trusts models.
        """
        f_norm = p_fusion / self.p_fusion_ref
        p_mag = f_norm * self.p_mag
        p_cryo = f_norm * self.p_cryo
        p_t_plant = f_norm * self.p_t_plant
        p_other = f_norm * self.p_other
        return p_mag, p_cryo, p_t_plant, p_other


parasitic_load_strat = EUDEMOReferenceParasiticLoadStrategy()

# %% [markdown]
#
# Now, we put everything together and build it

# %%
HCPB_bop = BalanceOfPlantModel(
    default_params,
    rad_sep_strat=rad_sep_strat,
    neutron_strat=neutron_power_strat,
    blanket_pump_strat=blanket_pump_strat,
    divertor_pump_strat=divertor_pump_strat,
    bop_cycle_strat=bop_cycle,
    parasitic_load_strat=parasitic_load_strat,
)
HCPB_bop.build()

# %% [markdown]
#
# And we can take a look...

# %%
HCPB_bop.plot(title="HCPB blanket")


# %% [markdown]
#
# What about if we had a different blanket concept? The coolant is different, which means
# the pumping loads will differ, and the power cycle will also be different. It's likely
# that the energy multiplication is different too

# %%

neutron_power_strat = NeutronPowerStrategy(
    f_blanket=0.9,
    f_divertor=0.05,
    f_vessel=0.04,
    f_other=0.01,
    energy_multiplication=1.25,
    decay_multiplication=1.002,
)
blanket_pump_strat = H2OPumping(0.005, eta_isentropic=0.99, eta_electric=0.87)
bop_cycle = PredeterminedEfficiency(0.33)

WCLL_bop = BalanceOfPlantModel(
    default_params,
    rad_sep_strat=rad_sep_strat,
    neutron_strat=neutron_power_strat,
    blanket_pump_strat=blanket_pump_strat,
    divertor_pump_strat=divertor_pump_strat,
    bop_cycle_strat=bop_cycle,
    parasitic_load_strat=parasitic_load_strat,
)
WCLL_bop.build()
WCLL_bop.plot(title="WCLL blanket")

plt.show()
