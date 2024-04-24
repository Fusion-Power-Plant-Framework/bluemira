# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Test script to make the CSG branch work."""

import json
import sys
from pathlib import Path
from time import time

import numpy as np
import openmc

from bluemira.base.constants import raw_uc
from bluemira.display import plot_2d, plot_3d, show_cad  # noqa: F401
from bluemira.geometry.coordinates import vector_intersect
from bluemira.geometry.tools import deserialise_shape, make_polygon
from bluemira.neutronics import result_presentation as present
from bluemira.neutronics.execution import (
    PlasmaSourceSimulation,
    Plotting,
    VolumeCalculation,
)
from bluemira.neutronics.full_tokamak import SingleNullTokamak
from bluemira.neutronics.make_materials import BlanketType
from bluemira.neutronics.neutronics_axisymmetric import (
    PlasmaSourceParametersPPS,
    create_materials,
)
from bluemira.neutronics.params import (
    BlanketLayers,
    OpenMCSimulationRuntimeParameters,
    PlasmaSourceParameters,
    TokamakDimensions,
    get_preset_physical_properties,
)
from bluemira.plasma_physics.reactions import n_DT_reactions

# Parameters initialization
CROSS_SECTION_XML = str(
    Path(
        "~/Others/cross_section_data/cross_section_data/cross_sections.xml"
    ).expanduser()
)
print(getattr(openmc.config, "config", openmc.config))
_breeder_materials, _tokamak_geometry = get_preset_physical_properties(BlanketType.HCPB)

# TODO: move most of the following parameters into PlasmaSourceSimulation.__init__?
runtime_variables = OpenMCSimulationRuntimeParameters(
    cross_section_xml=CROSS_SECTION_XML,
    particles=16800,  # 16800 takes 5 seconds,  1000000 takes 280 seconds.
    batches=3,
    photon_transport=True,
    electron_treatment="ttb",
    run_mode="dummy",  # no longer needed
    openmc_write_summary=False,
    parametric_source=True,
    # only used if stochastic_volume_calculation is turned on.
    volume_calc_particles=int(4e8),  # TODO: obsolete
)

_source_parameters = PlasmaSourceParameters(
    major_radius=8.938,  # [m]
    aspect_ratio=8.938 / 2.8938,  # [m]
    elongation=1.65,  # [dimensionless]
    triangularity=0.333,  # [dimensionless]
    reactor_power=1998e6,  # [W]
    peaking_factor=1.508,  # [dimensionless]
    temperature=raw_uc(15.4, "keV", "K"),
    shaf_shift=0.0,  # [m]
    vertical_shift=0.0,  # [m]
)

tokamak_dimensions = TokamakDimensions.from_tokamak_geometry_base(
    _tokamak_geometry, _source_parameters.major_radius, 0.1
)
tokamak_dimensions.inboard.manifold = 0.02
tokamak_dimensions.outboard.manifold = 0.2

source_parameters = PlasmaSourceParametersPPS.from_si(_source_parameters)
mat_lib = create_materials(_breeder_materials)

# Loading data
with open("data/inner_boundary") as j:
    inner_boundary = deserialise_shape(json.load(j))
with open("data/outer_boundary") as j:
    outer_boundary = deserialise_shape(json.load(j))
    # TODO: need to add method of scaling BluemiraWire (issue #3038 /
    # TODO: raise new issue about needing method to scale BluemiraWire)
with open("data/divertor_face.correct.json") as j:
    divertor_bmwire = deserialise_shape(json.load(j))
fw_panel_bp_list = [
    np.load("data/fw_panels_10_0.1.npy"),
    np.load("data/fw_panels_25_0.1.npy"),
    np.load("data/fw_panels_25_0.3.npy"),
    np.load("data/fw_panels_50_0.3.npy"),
    np.load("data/fw_panels_50_0.5.npy"),
]
panel_breakpoint_t = fw_panel_bp_list[0].T
# MANUAL FIX of the coordinates, because the data we're given is not perfect.
panel_breakpoint_t[0] = vector_intersect(
    panel_breakpoint_t[0],
    panel_breakpoint_t[1],
    divertor_bmwire.edges[0].start_point()[::2].flatten(),
    divertor_bmwire.edges[0].end_point()[::2].flatten(),
)
panel_breakpoint_t[-1] = vector_intersect(
    panel_breakpoint_t[-2],
    panel_breakpoint_t[-1],
    divertor_bmwire.edges[-1].start_point()[::2].flatten(),
    divertor_bmwire.edges[-1].end_point()[::2].flatten(),
)

last_point = divertor_bmwire.edges[
    -1
].end_point()  # gonna need to extend it by 1 unit vector.

blanket_panels_bmwire = make_polygon(
    np.insert(panel_breakpoint_t, 1, 0, axis=1).T,
    label="blanket panels",
    closed=False,
)

if __name__ == "__main__":  # begin computation
    START_TIME = time()

    def elapsed(text, start_time=START_TIME):  # noqa: D103
        new_time = time()
        print(f"t={new_time - start_time:9.6f}s: " + text)
        return new_time

    elapsed("Before creating pre-cells")

    generator = SingleNullTokamak(panel_breakpoint_t, divertor_bmwire, outer_boundary)
    generator.make_pre_cell_arrays(preserve_volume=True, snap_to_horizontal_angle=45)
    mat_dict = {
        BlanketLayers.Surface.name: mat_lib.outb_sf_mat,
        BlanketLayers.FirstWall.name: mat_lib.outb_fw_mat,
        BlanketLayers.BreedingZone.name: mat_lib.outb_bz_mat,
        BlanketLayers.Manifold.name: mat_lib.outb_mani_mat,
        BlanketLayers.VacuumVessel.name: mat_lib.outb_vv_mat,
        # TODO: make these two Divertor names into Enum
        "Divertor": mat_lib.div_fw_mat,
        "DivertorSurface": mat_lib.div_fw_mat,
        "CentralSolenoid": mat_lib.tf_coil_mat,
        "TFCoil": mat_lib.tf_coil_mat,
    }
    blanket_cell_array, div_cell_array, tf_coils, cs, plasma, void = (
        generator.make_cell_arrays(mat_dict, tokamak_dimensions, control_id=True)
    )

    cells = generator.cell_array.cells
    # plot_2d([*[pc.outline for pc in generator.pre_cell_array.blanket],
    #         *[dpc.outline for dpc in generator.pre_cell_array.divertor]
    # ])

    # using openmc
    if sys.argv[1].upper() == "PLOT":
        with Plotting(
            runtime_variables.cross_section_xml,
            cells,
            mat_lib,
        ) as plotting:
            plotting.run(
                [
                    generator.data.outer_boundary.bounding_box.x_max * 2.1,
                    generator.data.outer_boundary.bounding_box.z_max * 3.1,
                ],
                100,
            )

    elif sys.argv[1].upper() == "VOLUME":

        def format_cell_volume(cell: openmc.Cell):
            """Write the volume in the format that openmc usually prints it as."""
            return f"   Cell {cell.id}: {cell.volume} cm^3"

        with VolumeCalculation(
            runtime_variables.cross_section_xml, cells, mat_lib
        ) as vol_calc:
            all_ext_vertices = generator.get_coordinates_from_pre_cell_arrays(
                generator.pre_cell_array.blanket, generator.pre_cell_array.divertor
            )
            z_min = all_ext_vertices[:, -1].min()
            z_max = all_ext_vertices[:, -1].max()
            r_max = max(abs(all_ext_vertices[:, 0]))
            r_min = -r_max

            min_xyz = (r_min, r_min, z_min)
            max_xyz = (r_max, r_max, z_max)

            vol_calc.run(int(4e8), min_xyz, max_xyz)
            print("Compare this with the analytically obtained volumes:")
            for cell in cells:
                print(format_cell_volume(cell))

    elif sys.argv[1].upper() == "SOURCE":
        with PlasmaSourceSimulation(
            runtime_variables.cross_section_xml, cells, mat_lib, debug_mode=False
        ) as pss:
            pss.run(source_parameters, runtime_variables, blanket_cell_array, mat_dict)
            src_rate = n_DT_reactions(
                source_parameters.plasma_physics_units.reactor_power  # [MW]
                # TODO: when issue #2858 is fixed,
                # it will change the definition of n_DT_reactions from [MW] to [W].
                # in which which case, we use source_parameters.reactor_power.
            )
            results = present.OpenMCResult.from_run(
                pss.universe, src_rate, pss.statepoint_file
            )
            print(results)
