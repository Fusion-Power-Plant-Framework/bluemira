# %%
# COPYRIGHT PLACEHOLDER

"""
Time evolution of production and consumption power loads in EUDEMO 2017
baseline power plant (indirect HCPB-BoP concept) using the Power Cycle
module.

Reference for load inputs:
--------------------------
S. Minucci, S. Panella, S. Ciattaglia, M.C. Falvo, A. Lampasi,
Electrical Loads and Power Systems for the DEMO Nuclear Fusion Project,
Energies. 13 (2020) 2269. https://doi.org/10.3390/en13092269.
"""

# Run with:
# python examples/power_cycle/EUDEMO17_example/eudemo17.ex.py

# %%
from bluemira.power_cycle.net.manager import PowerCycleManager

try:
    import kits_import
    from kits_for_examples import DisplayKit, PathKit, ScenarioKit

    kits_import.successfull_import()

except ImportError:
    kits_import.sys.exit(kits_import.failed_import())


# %% [markdown]
# # Build & plot Power Cycle Manager
#
# In this example we have the following groups initialized in the
# manager:
# - Heating & Current Drive (`loads_HCD.json` file);
# - Magnetics (`loads_MAG.json` file);
# - Balance-of-Plant (`loads_BOP.json` file);
# - Tritium, Fuelling & Vacuum (`loads_TFV.json` file);
# - Plasma Diagnostics & Control (`loads_PDC.json` file);
# - Maintenance (`loads_MNT.json` file);
# - Cryoplant & Cryodistribution (`loads_CRY.json` file);
# - Auxiliaries (`loads_AUX.json` file).
#


# %%
def build_manager_config_path():
    """
    Read Power Cycle manager configuration file.
    """
    manager_config_filename = "manager_config.json"
    manager_config_path = PathKit.path_from_crumbs(
        PathKit.examples_crumbs,
        PathKit.eudemo17_folder,
        manager_config_filename,
    )
    return manager_config_path


def build_manager(manager_config_path):
    """
    Create the Power Cycle manager.
    """
    scenario_config_path = ScenarioKit.build_scenario_config_path()
    manager = PowerCycleManager(scenario_config_path, manager_config_path)
    return manager


def plot_manager(manager):
    """
    Plot net loads computed by the Power Cycle manager.
    """
    ax = DisplayKit.prepare_plot("EUDEMO 2017 Baseline, Net Power Loads")
    ax, _ = manager.plot(ax=ax)
    ax = DisplayKit.finalize_plot(ax)


# %%
if __name__ == "__main__":
    # Build Power Cycle Manager
    manager_config_path = build_manager_config_path()
    manager = build_manager(manager_config_path)
    plot_manager(manager)
