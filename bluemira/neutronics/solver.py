# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""OpenMC designer"""

import json
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, fields
from enum import auto
from operator import attrgetter
from pathlib import Path

import numpy as np
import openmc

from bluemira.base.constants import raw_uc
from bluemira.base.look_and_feel import bluemira_debug
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.base.tools import _timing
from bluemira.codes.interface import (
    BaseRunMode,
    CodesSetup,
    CodesSolver,
    CodesTask,
    CodesTeardown,
)
from bluemira.geometry.coordinates import vector_intersect
from bluemira.geometry.tools import deserialise_shape
from bluemira.geometry.wire import BluemiraWire
from bluemira.neutronics.make_csg import BlanketCellArray
from bluemira.neutronics.make_materials import create_materials
from bluemira.neutronics.neutronics_axisymmetric import SingleNullTokamak
from bluemira.neutronics.output import OpenMCResult
from bluemira.neutronics.params import (
    BlanketLayers,
    OpenMCSimulationRuntimeParameters,
    PlasmaSourceParameters,
    TokamakDimensions,
    get_preset_physical_properties,
)
from bluemira.neutronics.tallying import _create_tallies_from_filters, filter_new_cells
from bluemira.plasma_physics.reactions import n_DT_reactions


def some_function_on_blanket_wire(*_args):
    # Loading data
    with open("data/inner_boundary") as j:
        deserialise_shape(json.load(j))
    with open("data/outer_boundary") as j:
        outer_boundary = deserialise_shape(json.load(j))
        # TODO: need to add method of scaling BluemiraWire (issue #3038 /
        # TODO: raise new issue about needing method to scale BluemiraWire)
    with open("data/divertor_face.correct.json") as j:
        divertor_bmwire = deserialise_shape(json.load(j))
    with open("data/vv_bndry_outer.json") as j:
        vacuum_vessel_bmwire = deserialise_shape(json.load(j))

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
    return panel_breakpoint_t, outer_boundary, divertor_bmwire, vacuum_vessel_bmwire


class OpenMCRunModes(BaseRunMode):
    RUN = "fixed source"
    RUN_AND_PLOT = auto()
    PLOT = "plot"
    VOLUME = "volume"


OPENMC_NAME = "OpenMC"


@dataclass
class OpenMCNeutronicsSolverParams(ParameterFrame):
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


class Setup(CodesSetup):
    def __init__(
        self,
        out_path: str,
        codes_name: str,
        cross_section_xml: str,
        cells,
        source,
        blanket_cell_array,
        generator,
        outer_wire,
        mat_lib,
    ):
        super().__init__(None, codes_name)

        self.out_path = out_path
        self.cells = cells
        self.cross_section_xml = cross_section_xml
        self.source = source
        self.blanket_cell_array = blanket_cell_array
        self.generator = generator
        self.outer_wire = outer_wire
        self.mat_lib = mat_lib
        self.matlist = attrgetter(
            "outb_sf_mat",
            "outb_fw_mat",
            "outb_bz_mat",
            "outb_mani_mat",
            "outb_vv_mat",
            "div_fw_mat",
            "div_fw_mat",
        )

    @contextmanager
    def _base_setup(self, run_mode, *, debug: bool = False):
        from openmc.config import config  # noqa: PLC0415

        self.files_created = set()
        folder = run_mode.name.lower()
        cwd = Path(self.out_path, folder)
        cwd.mkdir(parents=True, exist_ok=True)

        config["cross_sections"] = self.cross_section_xml

        self.settings = openmc.Settings(
            run_mode=run_mode.value, output={"summary": False}
        )
        self.universe = openmc.Universe(cells=self.cells)
        self.geometry = openmc.Geometry(self.universe)
        self.settings.verbosity = 10 if debug else 6
        try:
            yield
        finally:
            for obj, pth in (
                (self.settings, Path(self.out_path, folder, "settings.xml")),
                (self.geometry, Path(self.out_path, folder, "geometry.xml")),
                (self.mat_lib, Path(self.out_path, folder, "materials.xml")),
            ):
                obj.export_to_xml(pth)
                self.files_created.add(pth)

    def _set_tallies(
        self, run_mode, blanket_cell_array: BlanketCellArray, material_list
    ):
        filter_list = filter_new_cells(material_list, blanket_cell_array)
        _create_tallies_from_filters(
            *filter_list, out_path=Path(self.out_path, run_mode.name.lower())
        )
        self.files_created.add(Path(self.out_path, run_mode.name.lower(), "tallies.xml"))

    def run(self, run_mode, runtime_params, source_params, *, debug: bool = False):
        with self._base_setup(run_mode, debug=debug):
            self.settings.particles = runtime_params.particles
            self.settings.source = self.source(source_params)
            self.settings.batches = int(runtime_params.batches)
            self.settings.photon_transport = runtime_params.photon_transport
            self.settings.electron_treatment = runtime_params.electron_treatment

            self._set_tallies(
                run_mode, self.blanket_cell_array, self.matlist(self.mat_lib)
            )
        self.files_created.add(f"statepoint.{runtime_params.batches}.h5")
        self.files_created.add("tallies.out")

    def plot(self, run_mode, runtime_params, _source_params, *, debug: bool = False):
        with self._base_setup(run_mode, debug=debug):
            plot_width_0 = self.outer_wire.bounding_box.x_max * 2.1
            plot_width_1 = self.outer_wire.bounding_box.z_max * 3.1
            plot = openmc.Plot()
            plot.basis = runtime_params.plot_axis
            plot.pixels = [
                int(plot_width_0 * runtime_params.plot_pixel_per_metre),
                int(plot_width_1 * runtime_params.plot_pixel_per_metre),
            ]
            plot.width = raw_uc([plot_width_0, plot_width_1], "m", "cm")

            plot_pth = Path(self.out_path, run_mode.name.lower(), "plots.xml")
            openmc.Plots([plot]).export_to_xml(plot_pth)
            self.files_created.add(plot_pth)

    def volume(self, run_mode, runtime_params, _source_params, *, debug: bool = False):
        z_max, z_min, r_max, r_min = self.generator.pre_cell_arrays.bounding_box()

        min_xyz = (r_min, r_min, z_min)
        max_xyz = (r_max, r_max, z_max)

        with self._base_setup(run_mode, debug=debug):
            self.settings.volume_calculations = openmc.VolumeCalculation(
                self.cells,
                runtime_params.particles,
                raw_uc(min_xyz, "m", "cm"),
                raw_uc(max_xyz, "m", "cm"),
            )
        self.files_created.add(Path(self.out_path, run_mode.name.lower(), "summary.h5"))
        # single batch
        self.files_created.add(
            Path(self.out_path, run_mode.name.lower(), "statepoint.1.h5")
        )


class Run(CodesTask):
    def __init__(self, out_path: Path, codes_name: str):
        super().__init__(None, codes_name)

        self.out_path = out_path

    def _run(self, run_mode, *, debug: bool = False):
        """Run openmc"""
        folder = run_mode.name.lower()
        cwd = Path(self.out_path, folder)
        cwd.mkdir(parents=True, exist_ok=True)
        _timing(openmc.run, "Executed in", "Running OpenMC", debug_info_str=False)(
            output=debug,
            threads=None,
            geometry_debug=False,
            restart_file=None,
            tracks=False,
            cwd=cwd,
            openmc_exec="openmc",
            mpi_args=None,
            event_based=False,
            path_input=None,
        )

    def run(self, run_mode, *, debug: bool = False):
        self._run(run_mode, debug=debug)

    def plot(self, run_mode, *, debug: bool = False):
        self._run(run_mode, debug=debug)

    def volume(self, run_mode, *, debug: bool = False):
        self._run(run_mode, debug=debug)


class Teardown(CodesTeardown):
    def __init__(
        self,
        cells,
        out_path: str,
        codes_name: str,
    ):
        super().__init__(None, codes_name)

        self.out_path = out_path
        self.cells = cells

    @staticmethod
    def _cleanup(files_created, *, delete_files: bool = False):
        """Remove files generated during the run (mainly .xml files.)"""
        if not delete_files:
            bluemira_debug("No files removed as debug mode is turned on.")
            return  # skip this entire method if we want to keep the files.

        removed_files, failed_to_remove_files = [], []
        for file_name in files_created:
            if (f := file_name).exists():
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

    def run(self, universe, files_created, source_params, statepoint_file):
        result = OpenMCResult.from_run(
            universe,
            n_DT_reactions(source_params.plasma_physics_units.reactor_power),
            statepoint_file,
        )
        self._cleanup(files_created)
        return result

    def plot(self, _universe, files_created, *_args):
        self._cleanup(files_created)

    def volume(
        self, _universe, files_created, _source_params, _statepoint_file
    ) -> dict[int, float]:
        self._cleanup(files_created)
        return {
            cell.id: raw_uc(
                np.nan if cell.volume is None else cell.volume, "cm^3", "m^3"
            )
            for cell in self.cells
        }


class OpenMCNeutronicsSolver(CodesSolver):
    name: str = OPENMC_NAME
    param_cls: type[OpenMCNeutronicsSolverParams] = OpenMCNeutronicsSolverParams
    params: OpenMCNeutronicsSolverParams
    run_mode_cls: type[OpenMCRunModes] = OpenMCRunModes
    setup_cls: type[Setup] = Setup
    run_cls: type[Run] = Run
    teardown_cls: type[Teardown] = Teardown

    def __init__(
        self,
        params: dict | ParameterFrame,
        blanket_wire: BluemiraWire,
        divertor_wire: BluemiraWire,
        vv_wire: BluemiraWire,
        source: Callable[[PlasmaSourceParameters], openmc.Source],
        build_config: dict | None = None,
    ):
        self.params = self.param_cls.from_frame(params)
        self.build_config = build_config

        self.out_path = self.build_config.get("neutronics_output_path", Path.cwd())

        _breeder_materials, _tokamak_geometry = get_preset_physical_properties(
            self.build_config["blanket_type"]
        )

        self.source = source

        self.tokamak_dimensions = TokamakDimensions.from_tokamak_geometry_base(
            _tokamak_geometry, self.params.major_radius.value, 0.1, 2, 4
        )
        self.tokamak_dimensions.inboard.manifold = 0.02  # why modified?
        self.tokamak_dimensions.outboard.manifold = 0.2

        self.mat_lib = create_materials(_breeder_materials)

        self.mat_dict = {
            BlanketLayers.Surface.name: self.mat_lib.outb_sf_mat,
            BlanketLayers.FirstWall.name: self.mat_lib.outb_fw_mat,
            BlanketLayers.BreedingZone.name: self.mat_lib.outb_bz_mat,
            BlanketLayers.Manifold.name: self.mat_lib.outb_mani_mat,
            BlanketLayers.VacuumVessel.name: self.mat_lib.outb_vv_mat,
            # TODO: make these two Divertor names into Enum
            "Divertor": self.mat_lib.div_fw_mat,
            "DivertorSurface": self.mat_lib.div_fw_mat,
            "CentralSolenoid": self.mat_lib.tf_coil_mat,
            "TFCoil": self.mat_lib.tf_coil_mat,
        }

        panel_breakpoint_t, outer_boundary, divertor_wire, self.vacuum_vessel_wire = (
            some_function_on_blanket_wire(blanket_wire, vv_wire, divertor_wire)
        )

        self.generator = SingleNullTokamak(
            panel_breakpoint_t, divertor_wire, outer_boundary, self.vacuum_vessel_wire
        )

        self.generator.make_pre_cell_arrays(snap_to_horizontal_angle=45)
        self.cell_arrays = self.generator.make_cell_arrays(
            self.mat_lib, self.tokamak_dimensions, control_id=True
        )

    @property
    def source(self) -> Callable[[PlasmaSourceParameters], openmc.Source]:
        return self._source

    @source.setter
    def source(self, value: Callable[[PlasmaSourceParameters], openmc.Source]):
        self._source = value

    def execute(self, *, debug=False):
        """Execute the setup, run, and teardown tasks, in order."""
        run_mode = self.build_config["run_mode"]
        if isinstance(run_mode, str):
            run_mode = self.run_mode_cls.from_string(run_mode)

        source_params = PlasmaSourceParameters.from_parameterframe(self.params)
        runtime_params = OpenMCSimulationRuntimeParameters(**{
            k.name: self.build_config[k.name]
            for k in fields(OpenMCSimulationRuntimeParameters)
            if k.name in self.build_config
        })
        if run_mode is OpenMCRunModes.RUN_AND_PLOT:
            for run_mode in (OpenMCRunModes.PLOT, OpenMCRunModes.RUN):
                result = self._single_run(
                    run_mode, source_params, runtime_params, debug=debug
                )
            return result
        return self._single_run(run_mode, source_params, runtime_params, debug=debug)

    def _single_run(
        self,
        run_mode: OpenMCRunModes,
        source_params: PlasmaSourceParameters,
        runtime_params: OpenMCSimulationRuntimeParameters,
        *,
        debug=False,
    ):
        self._setup = self.setup_cls(
            self.out_path,
            self.name,
            str(self.build_config["cross_section_xml"]),
            self.cell_arrays.cells,
            self.source,
            self.cell_arrays.blanket,
            self.generator,
            self.vacuum_vessel_wire,
            self.mat_lib,
        )
        self._run = self.run_cls(self.out_path, self.name)
        self._teardown = self.teardown_cls(
            self.cell_arrays.cells, self.out_path, self.name
        )

        result = None
        if setup := self._get_execution_method(self._setup, run_mode):
            result = setup(run_mode, runtime_params, source_params, debug=debug)
        if run := self._get_execution_method(self._run, run_mode):
            result = run(run_mode, debug=debug)
        if teardown := self._get_execution_method(self._teardown, run_mode):
            result = teardown(
                self._setup.universe,
                self._setup.files_created,
                source_params,
                Path(
                    self.out_path,
                    run_mode.name.lower(),
                    f"statepoint.{runtime_params.batches}.h5",
                ),
            )
        return result
