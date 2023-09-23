"""Example of how to use the neutronics module"""
from pathlib import Path
from typing import Tuple

import numpy as np
import openmc

from bluemira.base.constants import raw_uc
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

CROSS_SECTION_XML = str(
    Path(
        "~/Others/cross_section_data/cross_section_data/cross_sections.xml"
    ).expanduser()
)


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
        enrichment_fraction_Li6=0.60,
    )

    # Geometry variables

    # Break down from here.
    # Paper inboard build ---
    # Nuclear analyses of solid breeder blanket options for DEMO:
    # Status,challenges and outlook,
    # Pereslavtsev, 2019
    #
    # 0.400,      # TF Coil inner
    # 0.200,      # gap                  from figures
    # 0.060,       # VV steel wall
    # 0.480,      # VV
    # 0.060,       # VV steel wall
    # 0.020,       # gap                  from figures
    # 0.350,      # Back Supporting Structure
    # 0.060,       # Back Wall and Gas Collectors   Back wall = 3.0
    # 0.350,      # breeder zone
    # 0.022        # fw and armour

    shared_poloidal_outline = {
        "major_r": 8.938,  # [m]
        "minor_r": 2.883,  # [m]
        "elong": 1.65,  # [dimensionless]
        "triang": 0.333,  # [m]
    }
    shared_building_geometry = {  # that are identical in all three types of reactors.
        "inb_gap": 0.2,  # [m]
        "inb_vv_thick": 0.6,  # [m]
        "tf_thick": 0.4,  # [m]
        "outb_vv_thick": 0.6,  # [m]
    }
    if blanket_type is BlanketType.WCLL:
        tokamak_geometry = TokamakGeometry(
            **shared_poloidal_outline,
            **shared_building_geometry,
            inb_fw_thick=0.027,  # [m]
            inb_bz_thick=0.378,  # [m]
            inb_mnfld_thick=0.435,  # [m]
            outb_fw_thick=0.027,  # [m]
            outb_bz_thick=0.538,  # [m]
            outb_mnfld_thick=0.429,  # [m]
        )
    elif blanket_type is BlanketType.DCLL:
        tokamak_geometry = TokamakGeometry(
            **shared_poloidal_outline,
            **shared_building_geometry,
            inb_fw_thick=0.022,  # [m]
            inb_bz_thick=0.300,  # [m]
            inb_mnfld_thick=0.178,  # [m]
            outb_fw_thick=0.022,  # [m]
            outb_bz_thick=0.640,  # [m]
            outb_mnfld_thick=0.248,  # [m]
        )
    elif blanket_type is BlanketType.HCPB:
        # HCPB Design Report, 26/07/2019
        tokamak_geometry = TokamakGeometry(
            **shared_poloidal_outline,
            **shared_building_geometry,
            inb_fw_thick=0.027,  # [m]
            inb_bz_thick=0.460,  # [m]
            inb_mnfld_thick=0.560,  # [m]
            outb_fw_thick=0.027,  # [m]
            outb_bz_thick=0.460,  # [m]
            outb_mnfld_thick=0.560,  # [m]
        )

    return breeder_materials, tokamak_geometry


# set up the variables to be used for the openmc simulation
# allowed blanket_type so far = {'WCLL', 'DCLL', 'HCPB'}
breeder_materials, tokamak_geometry = get_preset_physical_properties(BlanketType.DCLL)

runtime_variables = OpenMCSimulationRuntimeParameters(
    particles=100000,  # 16800 takes 5 seconds,  1000000 takes 280 seconds.
    batches=2,
    photon_transport=True,
    electron_treatment="ttb",
    run_mode=openmc.settings.RunMode.FIXED_SOURCE.value,
    openmc_write_summary=False,
    parametric_source=True,
    # only used if stochastic_volume_calculation is turned on.
    volume_calc_particles=int(4e8),
    cross_section_xml=CROSS_SECTION_XML,
)

operation_variable = TokamakOperationParameters(
    reactor_power=1998e6,  # [W]
    temperature=round(raw_uc(15.4, "keV", "K"), 5),
    peaking_factor=1.508,  # [dimensionless]
    shaf_shift=0.0,  # [m]
    vertical_shift=0.0,  # [m]
)

# set up a DEMO-like reactor, and run OpenMC simualtion
tbr_heat_sim = TBRHeatingSimulation(
    runtime_variables, operation_variable, breeder_materials, tokamak_geometry
)
blanket_wire = make_polygon(Coordinates(np.load("blanket_face.npy")))
divertor_wire = make_polygon(Coordinates(np.load("divertor_face.npy")))
tbr_heat_sim.setup(
    blanket_wire,
    divertor_wire,
    new_major_radius=9.00,  # [m]
    new_aspect_ratio=3.10344,  # [dimensionless]
    new_elong=1.792,  # [dimensionless]
    plot_geometry=True,
)
tbr_heat_sim.run()
# get the TBR, component heating, first wall dpa, and photon heat flux
results = tbr_heat_sim.get_result()

print(results)
# tbr_heat_sim.calculate_volume_stochastically()
# # don't do this because it takes a long time.
