# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: tags,-all
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% tags=["remove-cell"]
# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later

"""
Simple example of a 0-D steady-state balance of plant view.
"""

# %%
import matplotlib.pyplot as plt

from bluemira.balance_of_plant.steady_state import (
    BalanceOfPlantModel,
    BoPModelParams,
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
default_params = BoPModelParams.from_dict({
    "P_fus_DT": {"value": 1995e6, "unit": "W", "source": "example"},
    "P_fus_DD": {"value": 5e6, "unit": "W", "source": "example"},
    "P_rad": {"value": 400e6, "unit": "W", "source": "example"},
    "P_hcd_ss": {"value": 50e6, "unit": "W", "source": "example"},
    "P_hcd_ss_el": {"value": 150e6, "unit": "W", "source": "example"},
})

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
    8e6, 7.5e6, 573.15, 773.15, eta_isentropic=0.9, eta_electric=0.87
)
bop_cycle = SuperheatedRankine(bb_t_out=773.15, delta_t_turbine=20)
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
        self.p_fusion_ref = 2037e6
        self.p_cryo = 44e6
        self.p_mag = 44e6
        self.p_t_plant = 15.5e6
        self.p_other = 31e6

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
