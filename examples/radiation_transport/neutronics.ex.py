from pathlib import Path
from typing import Tuple

import numpy as np

from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.tools import make_polygon
from bluemira.neutronics.make_materials import BlanketType
from bluemira.neutronics.params import (
    BreederTypeParameters,
    OpenMCSimulationRuntimeParameters,
    TokamakGeometry,
    TokamakOperationParameters,
)
from bluemira.neutronics.quick_tbr_heating import TBRHeatingSimulation

CROSS_SECTION_XML = str(Path("~/bluemira_openmc_data/cross_sections.xml").expanduser())


def get_preset_physical_properties(
    blanket_type: BlanketType,
) -> Tuple[BreederTypeParameters, TokamakGeometry]:
    """
    Works as a switch-case for choosing the tokamak geometry
        and blankets for a given blanket type.
    The allowed list of blanket types are specified in BlanketType.
    Currently, the blanket types with pre-populated data in this function are:
        {'wcll', 'dcll', 'hcpb'}
    """
    if not isinstance(blanket_type, BlanketType):
        raise KeyError(f"{blanket_type} is not an accepted blanket type.")
    breeder_materials = BreederTypeParameters(
        blanket_type=blanket_type,
        li_enrich_ao=60.0,  # atomic fraction percentage of lithium
    )

    # Geometry variables

    # Break down from here.
    # Paper inboard build ---
    # Nuclear analyses of solid breeder blanket options for DEMO:
    # Status,challenges and outlook,
    # Pereslavtsev, 2019
    #
    # 40.0,      # TF Coil inner
    # 20.0,      # gap                  from figures
    # 6.0,       # VV steel wall
    # 48.0,      # VV
    # 6.0,       # VV steel wall
    # 2.0,       # gap                  from figures
    # 35.0,      # Back Supporting Structure
    # 6.0,       # Back Wall and Gas Collectors   Back wall = 3.0
    # 35.0,      # breeder zone
    # 2.2        # fw and armour

    plasma_shape = {
        "minor_r": 288.3,
        "major_r": 893.8,
        "elong": 1.65,
        "shaf_shift": 0.0,
    }  # The shafranov shift of the plasma
    if blanket_type is BlanketType.WCLL:
        tokamak_geometry = TokamakGeometry(
            **plasma_shape,
            inb_fw_thick=2.7,
            inb_bz_thick=37.8,
            inb_mnfld_thick=43.5,
            inb_vv_thick=60.0,
            tf_thick=40.0,
            outb_fw_thick=2.7,
            outb_bz_thick=53.8,
            outb_mnfld_thick=42.9,
            outb_vv_thick=60.0,
        )
    elif blanket_type is BlanketType.DCLL:
        tokamak_geometry = TokamakGeometry(
            **plasma_shape,
            inb_fw_thick=2.2,
            inb_bz_thick=30.0,
            inb_mnfld_thick=17.8,
            inb_vv_thick=60.0,
            tf_thick=40.0,
            outb_fw_thick=2.2,
            outb_bz_thick=64.0,
            outb_mnfld_thick=24.8,
            outb_vv_thick=60.0,
        )
    elif blanket_type is BlanketType.HCPB:
        # HCPB Design Report, 26/07/2019
        tokamak_geometry = TokamakGeometry(
            **plasma_shape,
            inb_fw_thick=2.7,
            inb_bz_thick=46.0,
            inb_mnfld_thick=56.0,
            inb_vv_thick=60.0,
            tf_thick=40.0,
            outb_fw_thick=2.7,
            outb_bz_thick=46.0,
            outb_mnfld_thick=56.0,
            outb_vv_thick=60.0,
        )

    return breeder_materials, tokamak_geometry


# set up the variables to be used for the openmc simulation
# allowed blanket_type so far = {'WCLL', 'DCLL', 'HCPB'}
breeder_materials, tokamak_geometry = get_preset_physical_properties(BlanketType.WCLL)

runtime_variables = OpenMCSimulationRuntimeParameters(
    particles=16800,  # 16800 takes 5 seconds,  1000000 takes 280 seconds.
    batches=2,
    photon_transport=True,
    electron_treatment="ttb",
    run_mode="fixed source",
    openmc_write_summary=False,
    parametric_source=True,
    # only used if stochastic_volume_calculation is turned on.
    volume_calc_particles=int(4e8),
    cross_section_xml=CROSS_SECTION_XML,
)

operation_variable = TokamakOperationParameters(reactor_power_MW=1998.0)

# set up a DEMO-like reactor, and run OpenMC simualtion
tbr_heat_sim = TBRHeatingSimulation(
    runtime_variables, operation_variable, breeder_materials, tokamak_geometry
)
blanket_face = make_polygon(Coordinates(np.load("blanket_face.npy")))
divertor_face = make_polygon(Coordinates(np.load("divertor_face.npy")))
tbr_heat_sim.setup(
    blanket_face, divertor_face, R_0=900, A=3.10344, kappa=1.792, plot_geometry=True
)
tbr_heat_sim.run()
# get the TBR, component heating, first wall dpa, and photon heat flux
tbr_heat_sim.get_result(True)
# tbr_heat_sim.calculate_volume_stochastically()
# # don't do this because it takes a long time.
