# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Test script to make the CSG branch work."""

from pathlib import Path

import numpy as np

import bluemira.neutronics.make_geometry as mg
from bluemira.base.constants import raw_uc
from bluemira.geometry.coordinates import Coordinates
from bluemira.geometry.tools import make_polygon
from bluemira.neutronics.make_materials import BlanketType
from bluemira.neutronics.neutronics_axisymmetric import (
    PlasmaSourceParametersPPS,
    TBRHeatingSimulation,
    TokamakGeometry,
    create_and_export_materials,
    create_parametric_plasma_source,
    create_ring_source,
    setup_openmc,
)
from bluemira.neutronics.params import (
    OpenMCSimulationRuntimeParameters,
    PlasmaSourceParameters,
    get_preset_physical_properties,
)
from bluemira.neutronics.tallying import create_tallies
from bluemira.plasma_physics.reactions import n_DT_reactions

CROSS_SECTION_XML = str(
    Path(
        "~/Others/cross_section_data/cross_section_data/cross_sections.xml"
    ).expanduser()
)
breeder_materials, _tokamak_geometry = get_preset_physical_properties(BlanketType.HCPB)

runtime_variables = OpenMCSimulationRuntimeParameters(
    cross_section_xml=CROSS_SECTION_XML,
    particles=100000,  # 16800 takes 5 seconds,  1000000 takes 280 seconds.
    batches=2,
    photon_transport=True,
    electron_treatment="ttb",
    run_mode="fixed source",
    openmc_write_summary=False,
    parametric_source=True,
    # only used if stochastic_volume_calculation is turned on.
    volume_calc_particles=int(4e8),
)

_source_parameters = PlasmaSourceParameters(
    major_radius=8.938,  # [m]
    aspect_ratio=8.938 / 2.883,  # [m]
    elongation=1.65,  # [dimensionless]
    triangularity=0.333,  # [m]
    reactor_power=1998e6,  # [W]
    peaking_factor=1.508,  # [dimensionless]
    temperature=raw_uc(15.4, "keV", "K"),
    shaf_shift=0.0,  # [m]
    vertical_shift=0.0,  # [m]
)

tbr_heat_sim = TBRHeatingSimulation(
    runtime_variables,
    _source_parameters,
    breeder_materials,
    _tokamak_geometry,
)
# read in wires
tokamak_geometry = TokamakGeometry.from_si(_tokamak_geometry)
source_parameters = PlasmaSourceParametersPPS.from_si(_source_parameters)

blanket_wire = make_polygon(Coordinates(np.load("blanket_face.npy")))
divertor_wire = make_polygon(Coordinates(np.load("divertor_face.npy")))
# define machine new geometry
new_major_radius = 9.00  # [m]
new_aspect_ratio = 3.10344  # [dimensionless]
new_elong = 1.792  # [dimensionless]
plot_geometry = True

tbr_heat_sim.material_lib = create_and_export_materials(tbr_heat_sim.breeder_materials)
if tbr_heat_sim.runtime_variables.parametric_source:
    source = create_parametric_plasma_source(
        # tokamak geometry
        major_r=source_parameters.plasma_physics_units.major_radius,
        minor_r=source_parameters.plasma_physics_units.minor_radius,
        elongation=source_parameters.plasma_physics_units.elongation,
        triangularity=source_parameters.plasma_physics_units.triangularity,
        # plasma geometry
        peaking_factor=source_parameters.plasma_physics_units.peaking_factor,
        temperature=source_parameters.plasma_physics_units.temperature,
        radial_shift=source_parameters.plasma_physics_units.shaf_shift,
        vertical_shift=source_parameters.plasma_physics_units.vertical_shift,
        # plasma type
        mode="DT",
    )
else:
    source = create_ring_source(
        source_parameters.plasma_physics_units.major_radius,
        source_parameters.plasma_physics_units.shaf_shift,
    )
setup_openmc(source, tbr_heat_sim.runtime_variables)
blanket_points, div_points, num_inboard_points = mg.load_fw_points(
    source_parameters,
    blanket_wire,
    divertor_wire,
    raw_uc(new_major_radius, "m", "cm"),
    new_aspect_ratio,
    new_elong,
    True,
)  # TODO: improve here
tbr_heat_sim.cells, tbr_heat_sim.universe = mg.make_geometry(
    tokamak_geometry,
    blanket_points,
    div_points,
    num_inboard_points,
    tbr_heat_sim.material_lib,
)  # TODO: improve here

tbr_heat_sim.src_rate = n_DT_reactions(
    source_parameters.plasma_physics_units.reactor_power  # [MW]
    # TODO: when issue #2858 is fixed,
    # it will change the definition of n_DT_reactions from [MW] to [W].
    # in which which case, we use tbr_heat_sim.source_parameters.reactor_power.
)
create_tallies(tbr_heat_sim.cells, tbr_heat_sim.material_lib)  # TODO: improve here

tbr_heat_sim.run()
if False:
    tbr_heat_sim.calculate_volume_stochastically()
results = tbr_heat_sim.get_result()
print(results)
