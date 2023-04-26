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
try:
    import kits_import
    from kits_for_examples import ManagerKit, PathKit

    kits_import.successfull_import()

except ImportError:
    kits_import.sys.exit(kits_import.failed_import())


# %% [markdown]
# # Build & plot Power Cycle Manager (only with SSEN loads)
#
# In this example we have the following default groups initialized in
# the manager:
# - Magnetics (`loads_MAG.json` file);
# - Balance-of-Plant (`loads_BOP.json` file);
# - Tritium, Fuelling & Vacuum (`loads_TFV.json` file);
# - Plasma Diagnostics & Control (`loads_PDC.json` file);
# - Maintenance (`loads_MNT.json` file);
# - Cryoplant & Cryodistribution (`loads_CRY.json` file).
#
# Plus the following groups characterized only with SSEN loads:
# - Heating & Current Drive (`loads_HCD_onlySSEN.json` file);
# - Auxiliaries (`loads_AUX_onlySSEN.json` file).
#
# This allows us to visualize the same results presented in the
# reference article.
#


# %%
def build_ssen_manager_config_path():
    """
    Build manager configuration file path for SSEN-only plant case.
    """
    manager_config_filename = "manager_config_onlySSEN.json"
    manager_config_path = PathKit.build_eudemo_manager_config_path(
        manager_config_filename,
    )
    return manager_config_path


def plot_ssen_net_loads(ssen_manager):
    """
    Plot SSEN net loads computed by the Power Cycle manager.
    """
    title = "EUDEMO 2017 Baseline, SSEN Net Power Loads"
    return ManagerKit.plot_manager(title, ssen_manager)


# %% [markdown]
# # Analyze load in particular phase
#
# By studying a particular phase load, we can find which systems
# contribute most to the power recirculation of the plant.
#


# %%
def extract_phaseload_for_single_phase(pulseload, phase_label):
    """
    Extract 'PhaseLoad' from list that has a 'phase' attribute whose
    label matches 'phase_label'.
    """
    phaseload_of_single_phase = ManagerKit.extract_phaseload_for_single_phase(
        pulseload.phaseload_set,
        phase_label,
    )
    return phaseload_of_single_phase


# %% [markdown]
# # Build & plot Power Cycle Manager (including PPEN loads)
#
# In this example we have again the default groups initialized in
# the manager, plus the following groups with both SSEN and PPEN loads:
# - Heating & Current Drive (`loads_HCD_complete.json` file);
# - Auxiliaries (`loads_AUX_complete.json` file).
#


# %%
def build_complete_manager_config_path():
    """
    Build manager configuration file path for SSEN-only plant case.
    """
    manager_config_filename = "manager_config_complete.json"
    manager_config_path = PathKit.build_eudemo_manager_config_path(
        manager_config_filename,
    )
    return manager_config_path


def plot_complete_net_loads(complete_manager):
    """
    Plot complete net loads computed by the Power Cycle manager.
    """
    title = "EUDEMO 2017 Baseline, Complete Net Power Loads"
    return ManagerKit.plot_manager(title, complete_manager)


# %%
if __name__ == "__main__":
    # Build SSEN-only Power Cycle Manager
    ssen_manager_config_path = build_ssen_manager_config_path()
    ssen_manager = ManagerKit.build_manager(ssen_manager_config_path)
    plot_ssen_net_loads(ssen_manager)

    # Analyze active load in particular phase
    active_pulseload = ssen_manager.net_active
    ftt_active_phaseload = extract_phaseload_for_single_phase(
        active_pulseload,
        "ftt",
    )
    ManagerKit.plot_detailed_phaseload(ftt_active_phaseload)

    # Analyze reactive load in particular phase
    reactive_pulseload = ssen_manager.net_reactive
    ftt_reactive_phaseload = extract_phaseload_for_single_phase(
        reactive_pulseload,
        "ftt",
    )
    ManagerKit.plot_detailed_phaseload(ftt_reactive_phaseload)

    # Build complete Power Cycle Manager
    complete_manager_config_path = build_complete_manager_config_path()
    complete_manager = ManagerKit.build_manager(complete_manager_config_path)
    plot_complete_net_loads(complete_manager)
