# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Test script to make the CSG branch work."""

from pathlib import Path

from bluemira.base.constants import raw_uc
from bluemira.neutronics.blanket_data import (
    BlanketType,
    create_materials,
    get_preset_physical_properties,
)
from bluemira.neutronics.neutronics_axisymmetric import (
    NeutronicsReactor,
    NeutronicsReactorParameterFrame,
)
from bluemira.neutronics.openmc.solver import (
    OpenMCNeutronicsSolver,
    OpenMCNeutronicsSolverParams,
)
from bluemira.neutronics.openmc.sources import make_pps_source

build_config = {
    "cross_section_xml": Path(
        "~/Documents/BLUEPRINT/cross_section_data/cross_section_data/cross_sections.xml"
    ).expanduser(),
    "particles": 16800,  # 16800 takes 5 seconds,  1000000 takes 280 seconds.
    "batches": 3,
    "photon_transport": True,
    "electron_treatment": "ttb",
    "run_mode": "run_and_plot",
    "openmc_write_summary": False,
    "parametric_source": True,
    "plot_axis": "xz",
    "plot_pixel_per_metre": 100,
}

# Using built in breeding blanket properties
breeder_materials, tokamak_geometry = get_preset_physical_properties(BlanketType.HCPB)
material_library = create_materials(breeder_materials)

# Setup neutronics parameters
params = OpenMCNeutronicsSolverParams.from_dict({
    "major_radius": {"value": 8.938, "unit": "m"},
    "aspect_ratio": {"value": 8.938 / 2.8938, "unit": "m"},
    "elongation": {"value": 1.65, "unit": ""},
    "triangularity": {"value": 0.333, "unit": ""},
    "reactor_power": {"value": 1998, "unit": "MW"},
    "peaking_factor": {"value": 1.508, "unit": ""},
    "temperature": {"value": raw_uc(15.4, "keV", "K"), "unit": "K"},
    "shaf_shift": {"value": 0, "unit": "m"},
    "vertical_shift": {"value": 0, "unit": "m"},
})

# setup csg parameters
csg_params = NeutronicsReactorParameterFrame.from_dict({
    "inboard_fw_tk": {"value": tokamak_geometry.inb_fw_thick, "unit": "m"},
    "inboard_breeding_tk": {"value": tokamak_geometry.inb_bz_thick, "unit": "m"},
    "outboard_fw_tk": {"value": tokamak_geometry.outb_fw_thick, "unit": "m"},
    "outboard_breeding_tk": {"value": tokamak_geometry.outb_bz_thick, "unit": "m"},
    "blanket_io_cut": {"value": params.major_radius.value, "unit": "m"},
    "tf_inner_radius": {"value": 2, "unit": "m"},
    "tf_outer_radius": {"value": 4, "unit": "m"},
    "divertor_surface_tk": {"value": 0.1, "unit": "m"},
    "blanket_surface_tk": {"value": 0.01, "unit": "m"},
    "blk_ib_manifold": {"value": 0.02, "unit": "m"},
    "blk_ob_manifold": {"value": 0.2, "unit": "m"},
})


solver = OpenMCNeutronicsSolver(
    params,
    NeutronicsReactor(csg_params, None, None, None, material_library),
    source=make_pps_source,
    build_config=build_config,
)

print(solver.execute())
