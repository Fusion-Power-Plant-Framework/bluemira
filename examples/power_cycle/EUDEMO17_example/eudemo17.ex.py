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
# `python examples/power_cycle/EUDEMO17_example/eudemo17.ex.py`
#
# or start notebook with:
# `jupyter notebook`

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
# All values in these files come from the expected powers for each Plant
# Breakdown Structure item reported in the article (Table 2), in
# particular for the plant design for the HCPB blanket concept.
#
# Notice how the net loads plotted by `ssen_manager` match the
# difference between active/reactive loads to the grid reported in the
# article (800 MVAR and 640 MW; Section 4, last paragraph) and the SSEN
# consumption values reported in the article (~270 MVAR and 540 MW
# during flat-top; Figure 7a).


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
# # Build & plot Power Cycle Manager (only with SSEN loads & no BOP)
#
# If only partial results are desired, the manager can be initialized
# using a configuration file that does not include any systems of any
# groups.
#
# For example, if net loads should not include systems of the Balance-
# of-Plant group, the configuration file can omit those systems in the
# `systems` field of the `BOP` group. This can be useful for coupling
# `bluemira` to an external BOP model, for example.
#


# %%
def build_bopless_manager_config_path():
    """
    Build manager configuration file path for SSEN-only, no BOP systems
    plant case.
    """
    manager_config_filename = "manager_config_onlySSENnoBOP.json"
    manager_config_path = PathKit.build_eudemo_manager_config_path(
        manager_config_filename,
    )
    return manager_config_path


def plot_bopless_net_loads(bopless_manager):
    """
    Plot SSEN net loads (excluding BOP systems) computed by the Power
    Cycle manager.
    """
    title = "EUDEMO 2017 Baseline, SSEN (no BOP systems) Net Power Loads"
    return ManagerKit.plot_manager(title, bopless_manager)


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
# Notice that the article reports **peak** values for PPEN loads, which
# inevitably leads to a _negative_ net active power load by the power
# plant.
#
# This differentiation between cases could have been made in alternative
# ways. For example by defining different systems in a single input
# file (e.g. in `loads_HCD.json`) and defining different systems in it
# (e.g. splitting between ECH_SSEN and ECH_PPEN), and then only listing
# the desired systems in each relevant "manager_config" file.


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


# %% [markdown]
# # Export all manager versions Net Loads for the EU-DEMO 2017 Baseline
#
# The active and reactive net loads can be exported to text files using
# the `export_net_loads` method applied to each initilized version of
# the manager:
# - SSEN-only plant case (without PPEN loads);
# - no BOP, SSEN-only plant case;
# - complete plant case (with PPEN loads).
#


# %%
def build_export_file_path(export_file_filename):
    """
    Read Power Cycle manager configuration file.
    """
    export_file_crumbs = PathKit.path_from_crumbs(
        PathKit.examples_crumbs,
        PathKit.eudemo17_folder,
        export_file_filename,
    )
    return export_file_crumbs


# %%
if __name__ == "__main__":
    # Build SSEN-only Power Cycle Manager
    ssen_manager_config_path = build_ssen_manager_config_path()
    ssen_manager = ManagerKit.build_manager(ssen_manager_config_path)
    plot_ssen_net_loads(ssen_manager)

    # Build SSEN-only (no BOP systems) Power Cycle Manager
    bopless_manager_config_path = build_bopless_manager_config_path()
    bopless_manager = ManagerKit.build_manager(bopless_manager_config_path)
    plot_bopless_net_loads(bopless_manager)

    # Analyze active load in particular phase of SSEN-only case
    active_pulseload = ssen_manager.net_active
    ftt_active_phaseload = extract_phaseload_for_single_phase(
        active_pulseload,
        "ftt",
    )
    ManagerKit.plot_detailed_phaseload(ftt_active_phaseload)

    # Analyze reactive load in particular phase of SSEN-only case
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

    # Export net loads into text files
    filepath_ssen = build_export_file_path("exported_net_onlySSEN")
    filepath_bopless = build_export_file_path("exported_net_onlySSENnoBOP")
    filepath_complete = build_export_file_path("exported_net_complete")
    ssen_manager.export_net_loads(filepath_ssen)
    bopless_manager.export_net_loads(filepath_bopless)
    complete_manager.export_net_loads(filepath_complete)
