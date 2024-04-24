# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""OpenMC designer"""

import json
from collections.abc import Callable
from dataclasses import dataclass, fields
from enum import Enum, auto
from pathlib import Path

import numpy as np
import openmc

from bluemira.base.constants import raw_uc
from bluemira.base.designer import Designer
from bluemira.base.look_and_feel import bluemira_debug
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.base.tools import _timing
from bluemira.geometry.coordinates import vector_intersect
from bluemira.geometry.tools import deserialise_shape
from bluemira.geometry.wire import BluemiraWire
from bluemira.neutronics.full_tokamak import SingleNullTokamak
from bluemira.neutronics.make_csg import BlanketCellArray
from bluemira.neutronics.neutronics_axisymmetric import create_materials
from bluemira.neutronics.params import (
    BlanketLayers,
    OpenMCSimulationRuntimeParameters,
    PlasmaSourceParameters,
    TokamakDimensions,
    get_preset_physical_properties,
)
from bluemira.neutronics.result_presentation import OpenMCResult
from bluemira.neutronics.tallying import _create_tallies_from_filters, filter_new_cells
from bluemira.plasma_physics.reactions import n_DT_reactions
from bluemira.radiation_transport.error import NeutronicsError


def some_function_on_blanket_wire(*args):
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
    return panel_breakpoint_t, outer_boundary, divertor_bmwire


class OpenMCRunModes(Enum):
    RUN = "fixed source"
    RUN_AND_PLOT = auto()
    PLOT = "plot"
    VOLUME = "volume"


@dataclass
class OpenMCNeutronicsDesignerParams(ParameterFrame):
    """

    Parameters
    ----------
    major_radius:
        Major radius of the machine
    aspect_ratio:
        aspect ratio of the machine
    elongation:
        elongation of the plasma
    triangularity:
        triangularity of the plasma
    reactor_power:
        total reactor (thermal) power when operating at 100%
    peaking_factor:
        (max. heat flux on fw)/(avg. heat flux on fw)
    temperature:
        plasma temperature (assumed to be uniform throughout the plasma)
    shaf_shift:
        Shafranov shift
        shift of the centre of flux surfaces, i.e.
        mean(min radius, max radius) of the LCFS,
        towards the outboard radial direction.
    vertical_shift:
        how far (upwards) in the z direction is the centre of the plasma
        shifted compared to the geometric center of the poloidal cross-section.
    """

    major_radius: Parameter[float]  # [m]
    aspect_ratio: Parameter[float]  # [dimensionless]
    elongation: Parameter[float]  # [dimensionless]
    triangularity: Parameter[float]  # [dimensionless]
    reactor_power: Parameter[float]  # [W]
    peaking_factor: Parameter[float]  # [dimensionless]
    temperature: Parameter[float]  # [K]
    shaf_shift: Parameter[float]  # [m]
    vertical_shift: Parameter[float]  # [m]


# TODO add designer return type Designer[...]
class OpenMCNeutronicsDesigner(Designer):
    param_cls = OpenMCNeutronicsDesignerParams
    params: OpenMCNeutronicsDesignerParams

    def __init__(
        self,
        params: dict | ParameterFrame,
        blanket_wire: BluemiraWire,
        divertor_wire: BluemiraWire,
        vv_wire: BluemiraWire,
        source: Callable[[PlasmaSourceParameters], openmc.Source],
        build_config: dict | None = None,
    ):
        super().__init__(params, build_config)

        try:
            OpenMCRunModes[self.build_config[self.KEY_RUN_MODE].upper()]
        except KeyError:
            raise NeutronicsError("Run mode not recognised")

        self.runtime_params = OpenMCSimulationRuntimeParameters(**{
            k.name: self.build_config[k.name]
            for k in fields(OpenMCSimulationRuntimeParameters)
        })
        _breeder_materials, _tokamak_geometry = get_preset_physical_properties(
            self.build_config["blanket_type"]
        )

        self.source_params = PlasmaSourceParameters.from_parameterframe(self.params)
        self.source = source

        self.tokamak_dimensions = TokamakDimensions.from_tokamak_geometry_base(
            _tokamak_geometry, self.source_params.major_radius, 0.1
        )
        self.tokamak_dimensions.inboard.manifold = 0.02  # why modified?
        self.tokamak_dimensions.outboard.manifold = 0.2

        self.mat_lib = create_materials(_breeder_materials)

        panel_breakpoint_t, outer_boundary, divertor_wire = (
            some_function_on_blanket_wire(blanket_wire)
        )

        self.generator = SingleNullTokamak(
            panel_breakpoint_t, divertor_wire, outer_boundary
        )
        self.generator.make_pre_cell_arrays(
            preserve_volume=True, snap_to_horizontal_angle=45
        )
        self.mat_dict = {
            BlanketLayers.Surface.name: self.mat_lib.outb_sf_mat,
            BlanketLayers.FirstWall.name: self.mat_lib.outb_fw_mat,
            BlanketLayers.BreedingZone.name: self.mat_lib.outb_bz_mat,
            BlanketLayers.Manifold.name: self.mat_lib.outb_mani_mat,
            BlanketLayers.VacuumVessel.name: self.mat_lib.outb_vv_mat,
            # TODO: make these two Divertor names into Enum
            "Divertor": self.mat_lib.div_fw_mat,
            "DivertorSurface": self.mat_lib.div_fw_mat,
        }
        self.blanket_cell_array, div_cell_array, plasma, air = (
            self.generator.make_cell_arrays(
                self.mat_dict, self.tokamak_dimensions, control_id=True
            )
        )

        self.cells = self.generator.cell_array.cells

    def _openmc_setup(self, run_mode, cross_section_xml, *, debug_mode: bool = False):
        self.files_created = set()
        from openmc.config import config

        config["cross_sections"] = cross_section_xml

        self.settings = openmc.Settings(run_mode=run_mode, output={"summary": False})
        self.universe = openmc.Universe(cells=self.cells)
        self.geometry = openmc.Geometry(self.universe)
        self.settings.verbosity = 10 if debug_mode else 6

    def _run(
        self, run_mode, cross_section_xml, *args, debug_mode: bool = False, **kwargs
    ):
        """Complete the run"""
        # self._openmc_setup(run_mode, cross_section_xml, debug_mode=debug_mode)
        _timing(openmc.run, "Executed in", "Running OpenMC", debug_info_str=False)(
            *args, debug_mode, **kwargs
        )

    def _cleanup(self, *, debug_mode: bool = False):
        """Remove files generated during the run (mainly .xml files.)"""
        if debug_mode:
            bluemira_debug("No files removed as debug mode is turned on.")
            return  # skip this entire method if we want to keep the files.

        base_path = Path.cwd()
        removed_files, failed_to_remove_files = [], []
        for file_name in self.files_created:
            if (f := Path(base_path, file_name)).exists():
                f.unlink()
                removed_files.append(file_name)
            else:
                failed_to_remove_files.append(file_name)

        if removed_files:
            bluemira_debug(f"Removed files {removed_files}")
        if failed_to_remove_files:
            bluemira_debug(
                f"Attempted to remove files {failed_to_remove_files} but "
                "they don't exists."
            )

        self.files_created = set()  # clear the set

    @property
    def source(self) -> Callable[[PlasmaSourceParameters], openmc.Source]:
        return self._source

    @source.setter
    def source(self, value: Callable[[PlasmaSourceParameters], openmc.Source]):
        self._source = value

    def run(self) -> OpenMCResult:
        self._openmc_setup(
            OpenMCRunModes.RUN.value, self.build_config["cross_section_xml"]
        )

        self.settings.particles = self.runtime_params.particles
        self.settings.source = self.source(self.source_params)
        self.settings.batches = int(self.runtime_params.batches)
        self.settings.photon_transport = self.runtime_params.photon_transport
        self.settings.electron_treatment = self.runtime_params.electron_treatment
        self._set_tallies(self.blanket_cell_array, self.mat_dict)
        self.statepoint_file = f"statepoint.{self.settings.batches}.h5"
        self.files_created.add(self.statepoint_file)
        self.files_created.add("tallies.out")
        self.settings.export_to_xml()
        self.files_created.add("settings.xml")
        self.geometry.export_to_xml()
        self.files_created.add("geometry.xml")
        self.mat_lib.export()
        self.files_created.add("materials.xml")
        self._run(OpenMCRunModes.RUN.value, self.build_config["cross_section_xml"])
        # src_rate [MW]
        # TODO: when issue #2858 is fixed,
        # it will change the definition of n_DT_reactions from [MW] to [W].
        # in which which case, we use source_parameters.reactor_power.
        return OpenMCResult.from_run(
            self.universe,
            n_DT_reactions(self.source_params.plasma_physics_units.reactor_power),
            self.statepoint_file,
        )

    def run_and_plot(self) -> OpenMCResult:
        self.plot()
        return self.run()

    def plot(self):
        plot_width_0 = self.generator.data.outer_boundary.bounding_box.x_max * 2.1
        plot_width_1 = self.generator.data.outer_boundary.bounding_box.z_max * 3.1
        pixel_per_metre = 100
        self.plot = openmc.Plot()
        self.plot.basis = "xz"
        self.plot.pixels = [
            int(plot_width_0 * pixel_per_metre),
            int(plot_width_1 * pixel_per_metre),
        ]
        self.plot.width = raw_uc([plot_width_0, plot_width_1], "m", "cm")
        self.plot_list = openmc.Plots([self.plot])

        self.plot_list.export_to_xml()
        self.files_created.add("plots.xml")
        self._run(OpenMCRunModes.PLOT.value, self.build_config["cross_section_xml"])

    def _set_tallies(
        self, blanket_cell_array: BlanketCellArray, bodge_material_dict: dict
    ):
        filter_list = filter_new_cells(bodge_material_dict, blanket_cell_array)
        _create_tallies_from_filters(*filter_list)
        self.files_created.add("tallies.xml")

    def volume(self):
        """Run Monte Carlo to get the volume"""
        all_ext_vertices = self.generator.get_coordinates_from_pre_cell_arrays(
            self.generator.pre_cell_array.blanket, self.generator.pre_cell_array.divertor
        )
        z_min = all_ext_vertices[:, -1].min()
        z_max = all_ext_vertices[:, -1].max()
        r_max = max(abs(all_ext_vertices[:, 0]))
        r_min = -r_max

        min_xyz = (r_min, r_min, z_min)
        max_xyz = (r_max, r_max, z_max)

        self.settings.volume_calculations = openmc.VolumeCalculation(
            self.cells,
            self.runtime_params.volume_calc_particles,
            raw_uc(min_xyz, "m", "cm"),
            raw_uc(max_xyz, "m", "cm"),
        )
        self.files_created.add("summary.h5")
        self.files_created.add("statepoint.1.h5")  # single batch
        self._run(OpenMCRunModes.VOLUME.value, self.build_config["cross_section_xml"])

        return {cell.id: raw_uc(cell.volume, "cm^3", "m^3") for cell in self.cells}
