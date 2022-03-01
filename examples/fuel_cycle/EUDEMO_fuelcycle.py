# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
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
A typical fuel cycle result for an EU-DEMO reference point
"""

from bluemira.base.parameter import ParameterFrame
from bluemira.display.auto_config import plot_defaults
from bluemira.fuel_cycle.analysis import FuelCycleAnalysis
from bluemira.fuel_cycle.cycle import EUDEMOFuelCycleModel
from bluemira.fuel_cycle.lifecycle import LifeCycle
from bluemira.fuel_cycle.timeline_tools import (
    GompertzLearningStrategy,
    LogNormalAvailabilityStrategy,
)
from bluemira.fuel_cycle.tools import (
    convert_flux_to_flow,
    n_DD_reactions,
    n_DT_reactions,
)
from bluemira.utilities.tools import set_random_seed

plot_defaults()

# First let's get a reactor configuration (EU-DEMO 2015) and make a LifeCycle object
p_fus_DT = 2037
p_fus_DD = 3


# Select the reactor, blanket, and flux scenario values to use:

baseline_not_critical = True
HCPB_not_WCLL = True
high_not_low = False

if baseline_not_critical:
    A_global = 0.3
    TBR = 1.05
    f_dir = 0.8
else:
    A_global = 0.2
    TBR = 1.02
    f_dir = 0.6


# fmt:off
lifecycle_config = ParameterFrame([
    ["A_global", "Global load factor", A_global, "dimensionless", "Not always used", "Input"],
    ["I_p", "Plasma current", 19.2, "MA", None, "Input"],
    ["bmd", "Blanket maintenance duration", 150, "days", "Full replacement intervention duration", "Input"],
    ["dmd", "Divertor maintenance duration", 90, "days", "Full replacement intervention duration", "Input"],
    ["t_pulse", "Pulse length", 7200, "s", "Includes ramp-up and ramp-down time", "Input"],
    ["t_cs_recharge", "CS recharge time", 600, "s", "Presently assumed to dictate minimum dwell period", "Input"],
    ["t_pumpdown", "Pump down duration of the vessel in between pulses", 599, "s",
     "Presently assumed to take less time than the CS recharge", "Input"],
    ["s_ramp_up", "Plasma current ramp-up rate", 0.1, "MA/s", None, "R. Wenninger"],
    ["s_ramp_down", "Plasma current ramp-down rate", 0.1, "MA/s", None, "R. Wenninger"],
    ["n_DT_reactions", "D-T fusion reaction rate", n_DT_reactions(p_fus_DT), "1/s", "At full power", "Input"],
    ["n_DD_reactions", "D-D fusion reaction rate", n_DD_reactions(p_fus_DD), "1/s", "At full power", "Input"],
    ["blk_1_dpa", "Starter blanket life limit (EUROfer)", 20, "dpa",
     "https://iopscience.iop.org/article/10.1088/1741-4326/57/9/092002/pdf", "Input"],
    ["blk_2_dpa", "Second blanket life limit (EUROfer)", 50, "dpa",
     "https://iopscience.iop.org/article/10.1088/1741-4326/57/9/092002/pdf", "Input"],
    ["div_dpa", "Divertor life limit (CuCrZr)", 5, "dpa",
     "https://iopscience.iop.org/article/10.1088/1741-4326/57/9/092002/pdf", "Input"],
    ["vv_dpa", "Vacuum vessel life limit (SS316-LN-IG)", 3.25, "dpa", "RCC-Mx or whatever it is called", "Input"],
    ["tf_fluence", "Insulation fluence limit for ITER equivalent to 10 MGy", 3.2e21, "1/m^2",
     "https://ieeexplore.ieee.org/document/6374236/", "Input"],
])
# fmt:on

lifecycle_inputs = {}

# We need to define some stragies to define the pseudo-random timelines

# Let's choose a LearningStrategy such that the operational availability grows over time
learning_strategy = GompertzLearningStrategy(
    learn_rate=1.0, min_op_availability=0.1, max_op_availability=0.5
)
# Let's choose an OperationalAvailabilityStrategy to determine how to distribute outages
availability_strategy = LogNormalAvailabilityStrategy(sigma=2.0)

lifecycle = LifeCycle(
    lifecycle_config, learning_strategy, availability_strategy, lifecycle_inputs
)

# We can use this LifeCycle to make pseudo-randomised timelines. Let's set a
# random seed number first to get repeatable results
set_random_seed(2358203947)

# Let's do 50 runs Monte Carlo
# NOTE: Make sure you have enough memory..!
n = 50
time_dicts = [lifecycle.make_timeline().to_dict() for _ in range(n)]

# Now let's set up a TFVSystem

# First we need to get some input parameters

# Some conversions from inventories to residence times
# (as discussed with Jonas Schwenzer, KIT)
m_exhaust_systems = 0.11
exhaust_influx = 3.28229059e-05
t_exh = m_exhaust_systems / exhaust_influx

vessel_outflux = 0.00032812

if high_not_low:
    fw_retention_flux = 1e20
else:
    fw_retention_flux = 1e19

# Numbers taken from KDI-2 report (R. Arrendondo, F. Franza, C. Moro)
if HCPB_not_WCLL:
    if high_not_low:
        eta_ivc = 0.33013436
        max_ivc_inventory = 1.11571993
        max_bb_inventory = 0.055  # F. Franza
    else:
        eta_ivc = 0.0972729
        max_ivc_inventory = 0.37786173
        max_bb_inventory = 0.026
else:
    if high_not_low:
        eta_ivc = 0.62037084
        max_ivc_inventory = 1.65440323
        max_bb_inventory = 0.18519
    else:
        eta_ivc = 0.13090036
        max_ivc_inventory = 0.64900007
        max_bb_inventory = 0.06314

fw_flow = convert_flux_to_flow(fw_retention_flux, 1400)


# Normalise sqrt sink factors, because T flows are off...
eta_norm = fw_flow / vessel_outflux


if eta_norm >= 1:
    eta_ivc /= eta_norm
else:
    eta_ivc /= eta_norm

m_dir_pump = 0.024
t_pump = m_dir_pump / vessel_outflux

# It was agreed that this 1.8 kg is needed steady-state, and is used as I_TFV_min
m_cryodistillation = 1.8

# fmt: off
tfv_config = ParameterFrame([
    ['TBR', 'Tritium breeding ratio', TBR, 'dimensionless', None, 'Input'],
    ['f_b', 'Burn-up fraction', 0.015, 'dimensionless', None, 'Input'],
    ['m_gas', 'Gas puff flow rate', 50, 'Pa m^3/s', 'To maintain detachment - no chance of fusion from gas injection', 'Discussions with Chris Day and Yannick Hörstenmeyer'],
    ['A_global', 'Load factor', A_global, 'dimensionless', None, 'Silent input'],
    ['r_learn', 'Learning rate', 1, 'dimensionless', None, 'Silent input'],
    ['t_pump', 'Time in DIR loop', t_pump, 's', 'Time between exit from plasma and entry into plasma through DIR loop', 'Discussions with Chris Day and Yannick Hörstenmeyer'],
    ['t_exh', 'Time in INDIR loop', t_exh, 's', 'Time between exit from plasma and entry into TFV systems INDIR', 'Input'],
    ['t_ters', 'Time from BB exit to TFV system', 6750, 's', None, 'Input'],
    ['t_freeze', 'Time taken to freeze pellets', 3600 / 2, 's', None, 'Discussions with Chris Day and Yannick Hörstenmeyer'],
    ['f_dir', 'Fraction of flow through DIR loop', f_dir, 'dimensionless', None, 'Discussions with Chris Day and Yannick Hörstenmeyer'],
    ['t_detrit', 'Time in detritiation system', 0, 's', None, 'Input'],
    ['f_detrit_split', 'Fraction of detritiation line tritium extracted', 0.9999, 'dimensionless', None, 'Input'],
    ['f_exh_split', 'Fraction of exhaust tritium extracted', 0.99, 'dimensionless', None, 'Input'],
    ['eta_fuel_pump', 'Efficiency of fuel line pump', 0.9, 'dimensionless', 'Pump which pumps down the fuelling lines', 'Input'],
    ['eta_f', 'Fuelling efficiency', 0.5, 'dimensionless', 'Efficiency of the fuelling lines prior to entry into the VV chamber', 'Input'],
    ['I_miv', 'Maximum in-vessel T inventory', max_ivc_inventory, 'kg', None, 'Input'],
    ['I_tfv_min', 'Minimum TFV inventory', m_cryodistillation, 'kg', 'Without which e.g. cryodistillation columns are not effective', "Discussions with Chris Day and Jonas Schwenzer (N.B. working assumptions only)"],
    ['I_tfv_max', 'Maximum TFV inventory', m_cryodistillation + 0.2, 'kg', "Account for T sequestration inside the T plant", "Discussions with Chris Day and Jonas Schwenzer (N.B. working assumptions only)"],
    ['I_mbb', 'Maximum BB T inventory', max_bb_inventory, 'kg', None, 'Input'],
    ['eta_iv', 'In-vessel bathtub parameter', eta_ivc, 'dimensionless', None, 'Input'],
    ['eta_bb', 'BB bathtub parameter', 0.995, 'dimensionless', None, 'Input'],
    ['eta_tfv', 'TFV bathtub parameter', 0.998, 'dimensionless', None, 'Input'],
    ['f_terscwps', 'TERS and CWPS cumulated factor', 0.9999, 'dimensionless', None, 'Input']
])
# fmt:on


# We can run a single model and look at a typical result
model = EUDEMOFuelCycleModel(tfv_config, {})
model.run(time_dicts[0])
model.plot()

# Now, let's run the fuel cycle model for all the timelines we generated
tfv_analysis = FuelCycleAnalysis(model)
tfv_analysis.run_model(time_dicts)

# And the distributions for the start-up inventory and doubling time:
tfv_analysis.plot()

# And finally, you can get the desired statistical results:

m_T_start_95 = tfv_analysis.get_startup_inventory("95th")
t_d_95 = tfv_analysis.get_doubling_time("95th")

m_T_start_mean = tfv_analysis.get_startup_inventory("mean")
t_d_mean = tfv_analysis.get_doubling_time("mean")

m_T_start_max = tfv_analysis.get_startup_inventory("max")
t_d_max = tfv_analysis.get_doubling_time("max")


print(f"The mean start-up inventory is: {m_T_start_mean:.2f} kg.")
print(f"The mean doubling time is: {t_d_mean:.2f} years.")
print(f"The 95th percentile start-up inventory is: {m_T_start_95:.2f} kg.")
print(f"The 95th percentile doubling time is: {t_d_95:.2f} years.")
print(f"The maximum start-up inventory is: {m_T_start_max:.2f} kg.")
print(f"The maximum doubling time is: {t_d_max:.2f} years.")
