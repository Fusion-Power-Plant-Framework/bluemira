# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""Test script to make the CSG branch work."""

import json
from pathlib import Path

import numpy as np
from numpy import typing as npt
import matplotlib.pyplot as plt

import bluemira.neutronics.make_geometry as mg
from bluemira.base.constants import raw_uc
from bluemira.display import plot_2d, show_cad, plot_3d # noqa: F401
from bluemira.geometry.coordinates import Coordinates, vector_intersect
from bluemira.geometry.tools import deserialize_shape, make_polygon  # make_circle_arc_3P
from bluemira.geometry.wire import BluemiraWire
from bluemira.neutronics.make_csg import (ThicknessFractions,
                                        fill_xz_to_3d,
                                        split_blanket_into_precells,)
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
    triangularity=0.333,  # [dimensionless]
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
thickness_fractions = ThicknessFractions.from_TokamakGeometry(tokamak_geometry)
# blanket_points, div_points, num_inboard_points = mg.load_fw_points(
#     source_parameters,
#     make_polygon(Coordinates(np.load("blanket_face.npy"))),
#     make_polygon(Coordinates(np.load("divertor_face.npy"))),
#     new_major_radius=raw_uc(9.0, "m", "cm"),
#     new_aspect_ratio=3.10344,  # [dimensionless]
#     new_elong=1.792,  # [dimensionless]
#     save_plots=True,
# data loading begins here.
from time import time
def elapsed(start_time=time()):
    return str(time()-start_time)+"s"
print(elapsed())
with open("data/inner_boundary") as j:
    inner_boundary = deserialize_shape(json.load(j))
with open("data/outer_boundary") as j:
    outer_boundary = deserialize_shape(json.load(j))
with open("data/divertor_face.correct.json") as j:
    divertor_bmwire = deserialize_shape(json.load(j))
fw_panel_bp_list = [
    np.load("data/fw_panels_10_0.1.npy"),
    np.load("data/fw_panels_25_0.1.npy"),
    np.load("data/fw_panels_25_0.3.npy"),
    np.load("data/fw_panels_50_0.3.npy"),
    np.load("data/fw_panels_50_0.5.npy"),
]
panel_breakpoint_T = fw_panel_bp_list[1].T
# TODO: MANUAL FIX begins here
panel_breakpoint_T[0] = vector_intersect(panel_breakpoint_T[0], panel_breakpoint_T[1], divertor_bmwire.edges[0].start_point()[::2].flatten(), divertor_bmwire.edges[0].end_point()[::2].flatten())
panel_breakpoint_T[-1]= vector_intersect(panel_breakpoint_T[-2],panel_breakpoint_T[-1],divertor_bmwire.edges[-1].start_point()[::2].flatten(), divertor_bmwire.edges[-1].end_point()[::2].flatten())
# TODO: MANUALLY determine where on the outer boundary we want to set the Z points.

last_point = divertor_bmwire.edges[-1].end_point()

def polygon_from2D(xarray: npt.NDArray[float], zarray: npt.NDArray[float], **kwargs):
    """Create BluemiraWire from  x coordinates and z coordinates.
    Parameters
    ----------
    xarray: np.ndarray of shape (N,)
    zarray: np.ndarray of shape (N,)

    Returns
    -------
    BluemiraWire:
        wire made of straight lines joint by vertices specified by the input parameters.
    """
    return make_polygon(fill_xz_to_3d([xarray, zarray]), **kwargs)

blanket_panels_bmwire = polygon_from2D(
                    *panel_breakpoint_T.T,
                    label="blanket panels",
                    closed=False,
                )

print(elapsed())
precell_list = split_blanket_into_precells(outer_boundary, panel_breakpoint_T,
                                            ending_cut=last_point.xz.flatten())
print(elapsed())
plot_2d([c.outline for c in precell_list])
import sys; sys.exit()

tbr_heat_sim.cells, tbr_heat_sim.universe = mg.make_neutronics_geometry(
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
