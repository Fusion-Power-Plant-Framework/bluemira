# SPDX-FileCopyrightText: 2021-present M. Coleman, J. Cook, F. Franza
# SPDX-FileCopyrightText: 2021-present I.A. Maione, S. McIntosh
# SPDX-FileCopyrightText: 2021-present J. Morris, D. Short
#
# SPDX-License-Identifier: LGPL-2.1-or-later
"""OpenMC designer"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields
from enum import auto
from operator import attrgetter
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

import numpy as np
import openmc

from bluemira.base.constants import raw_uc
from bluemira.base.parameter_frame import ParameterFrame, make_parameter_frame
from bluemira.base.tools import _timing
from bluemira.codes.interface import (
    BaseRunMode,
    CodesSetup,
    CodesSolver,
    CodesTask,
    CodesTeardown,
)
from bluemira.codes.openmc.make_csg import (
    BluemiraNeutronicsCSG,
    CellStage,
    make_cell_arrays,
)
from bluemira.codes.openmc.material import MaterialsLibrary
from bluemira.codes.openmc.output import (
    NeutronicsOutputParams,
    OpenMCCSGResult,
    OpenMCDAGMCResult,
)
from bluemira.codes.openmc.params import (
    OpenMCNeutronicsSolverParams,
    PlasmaSourceParameters,
)
from bluemira.codes.openmc.tallying import csg_filter_cells, dagmc_tallys
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.radiation_transport.neutronics.neutronics_axisymmetric import (
    NeutronicsReactor,
)

if TYPE_CHECKING:
    from matproplib.conditions import OperationalConditions

    from bluemira.radiation_transport.neutronics.neutronics_axisymmetric import (
        NeutronicsReactor,
    )


class OpenMCRunModes(BaseRunMode):
    """OpenMC run modes"""

    RUN = openmc.settings.RunMode.FIXED_SOURCE.value
    RUN_AND_PLOT = auto()
    PLOT = openmc.settings.RunMode.PLOT.value
    VOLUME = openmc.settings.RunMode.VOLUME.value


OPENMC_NAME = "OpenMC"


@dataclass
class OpenMCSimulationRuntimeParameters:
    """Parameters used in the actual simulation

    Parameters
    ----------
    particles:
        Number of neutrons emitted by the plasma source per batch.
    batches:
        How many batches to simulate.
    photon_transport:
        Whether to simulate the transport of photons (i.e. gamma-rays created) or not.
    electron_treatment:
        The way in which OpenMC handles secondary charged particles.
        'thick-target bremsstrahlung' or 'local energy deposition'
        'thick-target bremsstrahlung' accounts for the energy carried away by
        bremsstrahlung photons and deposited elsewhere, whereas 'local energy
        deposition' assumes electrons deposit all energies locally.
        (the latter is expected to be computationally faster.)
    run_mode:
        see below for details:
        https://docs.openmc.org/en/stable/usersguide/settings.html#run-modes
    openmc_write_summary:
        whether openmc should write a 'summary.h5' file or not.
    cross_section_xml:
        Where the xml file for cross-section is stored locally.
    """

    particles: int  # number of particles used in the neutronics simulation
    cross_section_xml: str | Path
    batches: int = 2
    photon_transport: bool = True
    # Bremsstrahlung only matters for very thin objects
    electron_treatment: Literal["ttb", "led"] = "led"
    run_mode: str = OpenMCRunModes.RUN.value
    openmc_write_summary: bool = False
    plot_axis: str = "xz"
    plot_pixel_per_metre: int = 100
    rel_max_lost_particles: float = 1e-6
    max_lost_particles: int = 10


# Signature for a function that creates an OpenMC neutron source
NeutronSourceCreator: TypeAlias = Callable[
    [Equilibrium, PlasmaSourceParameters], tuple[openmc.Source, float, float]
]


@dataclass
class PlotConfig:
    width: list[float]
    pixels: list[int]
    basis: str = "xz"
    colour_by: str = "cell"
    show_overlaps: bool = True


@dataclass
class SourceInfo:
    rate: float
    triton_rate: float


@dataclass
class FigureData:
    axis: Any
    path: Path


class OpenMCBaseSetup(CodesSetup, ABC):
    """Setup task for OpenMC solver"""

    tally_mats: openmc.Materials
    tally_geom: openmc.Geometry | CellStage

    def __init__(
        self,
        codes_name: str,
        cross_section_xml: str,
        eq: Equilibrium,
        source,
        materials,
    ):
        super().__init__(None, codes_name)

        self.cross_section_xml = cross_section_xml
        self.eq = eq
        self.source = source
        self._source_rate = 1.0
        self._source_triton_rate = 1.0
        self.materials = materials

    def _base_setup(
        self,
        run_mode,
        rel_max_lost_particles,
        max_lost_particles,
        *,
        debug: bool = False,
    ):
        from openmc.config import config  # noqa: PLC0415

        config["cross_sections"] = self.cross_section_xml

        settings = openmc.Settings(
            run_mode=run_mode.value,
            output={"summary": False},
            rel_max_lost_particles=rel_max_lost_particles,
            max_lost_particles=int(max_lost_particles),
        )

        self.universe, self.geometry = self._create_geometry()

        settings.verbosity = 10 if debug else 6

        return settings

    def _create_model(
        self, settings: openmc.Settings, tallies: openmc.Tallies | None = None
    ) -> openmc.Model:
        model = openmc.Model(geometry=self.geometry, tallies=tallies, settings=settings)
        if isinstance(self.materials, MaterialsLibrary):
            model.materials = self.materials.get_all_materials()
        else:
            model.materials = self.materials
        return model

    @abstractmethod
    def _create_geometry(self) -> tuple[openmc.Universe, openmc.geometry]: ...

    def _create_tallies(self, tally_function: TALLY_FUNCTION_TYPE) -> openmc.Tallies:
        tallies_list = []
        for name, scores, filters in tally_function(self.tally_mats, self.tally_geom):
            tally = openmc.Tally(name=name)
            tally.scores = [scores]
            if filters is not None:
                tally.filters = filters
            tallies_list.append(tally)

        return openmc.Tallies(tallies_list)

    @abstractmethod
    def plot(
        self,
        run_mode,
        runtime_params,
        eq,
        source_params,
        tally_function,
        *,
        debug: bool = False,
    ): ...

    @abstractmethod
    def volume(
        self,
        run_mode,
        runtime_params,
        eq,
        source_params,
        tally_function,
        *,
        debug: bool = False,
    ): ...

    def run(
        self,
        run_mode,
        runtime_params,
        eq,
        source_params,
        tally_function,
        *,
        debug: bool = False,
    ) -> tuple[openmc.Model, SourceInfo]:
        """Run stage for setup openmc"""
        settings = self._base_setup(
            run_mode,
            runtime_params.rel_max_lost_particles,
            runtime_params.max_lost_particles,
            debug=debug,
        )
        settings.particles = runtime_params.particles
        settings.source, source_rate, source_t_rate = self.source(eq, source_params)
        settings.batches = int(runtime_params.batches)
        settings.photon_transport = runtime_params.photon_transport
        settings.electron_treatment = runtime_params.electron_treatment

        tallies = self._create_tallies(tally_function)

        return (
            self._create_model(settings, tallies),
            SourceInfo(source_rate, source_t_rate),
        )

    def _plot(
        self,
        run_mode,
        runtime_params,
        bounding_box: Sequence[float] | None = None,
        *,
        debug: bool = False,
    ) -> tuple[openmc.Model, PlotConfig]:
        """Plot stage for setup openmc"""
        settings = self._base_setup(run_mode, debug=debug)
        if bounding_box is None:
            r_max, _y_max, z_max = self.universe.bounding_box.upper_right
        else:
            z_max, _z_min, r_max, _r_min = bounding_box
        plot_width_0 = r_max * 2.1
        plot_width_1 = z_max * 3.1
        basis = runtime_params.plot_axis
        pixels = [
            int(plot_width_0 * runtime_params.plot_pixel_per_metre),
            int(plot_width_1 * runtime_params.plot_pixel_per_metre),
        ]
        width = raw_uc([plot_width_0, plot_width_1], "m", "cm")

        return (
            self._create_model(settings),
            PlotConfig(width, pixels, basis),
        )

    def _volume(
        self, run_mode, runtime_params, domain, bounding_box, *, debug: bool = False
    ) -> tuple[openmc.Model, None]:
        """Stochastic volume stage for setup openmc"""
        z_max, z_min, r_max, r_min = bounding_box

        min_xyz = (r_min, r_min, z_min)
        max_xyz = (r_max, r_max, z_max)
        settings = self._base_setup(run_mode, debug=debug)
        settings.volume_calculations = openmc.VolumeCalculation(
            domain,
            runtime_params.particles,
            raw_uc(min_xyz, "m", "cm"),
            raw_uc(max_xyz, "m", "cm"),
        )
        return self._create_model(settings), None


class OpenMCCSGSetup(OpenMCBaseSetup):
    def __init__(
        self,
        codes_name: str,
        cross_section_xml: str,
        eq: Equilibrium,
        source,
        materials,
        cell_arrays,
        pre_cell_model,
    ):
        super().__init__(codes_name, cross_section_xml, eq, source, materials)
        self.cell_arrays = cell_arrays
        self.pre_cell_model = pre_cell_model
        self.mat_list = attrgetter(
            "outb_sf_mat",
            "outb_fw_mat",
            "outb_bz_mat",
            "outb_mani_mat",
            "outb_vv_mat",
            "divertor_mat",
            "div_fw_mat",
            "tf_coil_mat",
        )

    @property
    def tally_mats(self) -> list[openmc.Material]:
        return self.mat_list(self.materials)

    @property
    def tally_geom(self) -> CellStage:
        return self.cell_arrays

    def _create_geometry(self):
        universe = openmc.Universe(cells=self.cell_arrays.cells)
        geometry = openmc.Geometry(universe)
        return universe, geometry

    def plot(
        self,
        run_mode,
        runtime_params,
        *_args,
        debug: bool = False,
    ) -> tuple[openmc.Model, PlotConfig]:
        return self._plot(
            run_mode, runtime_params, self.pre_cell_model.bounding_box, debug=debug
        )

    def volume(
        self,
        run_mode,
        runtime_params,
        *_args,
        debug: bool = False,
    ) -> tuple[openmc.Model, None]:
        return self._volume(
            run_mode,
            runtime_params,
            self.cell_arrays.cells,
            self.pre_cell_model.bounding_box,
            debug=debug,
        )


class OpenMCDAGSetup(OpenMCBaseSetup):
    def __init__(
        self,
        codes_name: str,
        cross_section_xml: str,
        eq: Equilibrium,
        source,
        materials,
        dag_model_path: Path,
    ):
        super().__init__(codes_name, cross_section_xml, eq, source, materials)
        self.dag_model_path = dag_model_path

    def _create_geometry(self):
        universe = openmc.DAGMCUniverse(
            filename=self.dag_model_path.as_posix(),
            auto_geom_ids=True,
        ).bounded_universe()
        geometry = openmc.Geometry(universe)
        return universe, geometry

    @property
    def tally_mats(self) -> list[openmc.Material]:
        return self.materials

    @property
    def tally_geom(self) -> openmc.Geometry:
        return self.geometry

    def plot(
        self,
        run_mode,
        runtime_params,
        *_args,
        debug: bool = False,
    ) -> tuple[openmc.Model, PlotConfig]:
        return self._plot(run_mode, runtime_params, debug=debug)

    def volume(
        self,
        run_mode,
        runtime_params,
        *_args,
        debug: bool = False,
    ) -> tuple[openmc.Model, None]:
        return self._volume(
            run_mode,
            runtime_params,
            self.universe,
            self.universe.bounding_box,
            debug=debug,
        )


class OpenMCRun(CodesTask):
    """Run task for OpenMC solver"""

    def __init__(self, out_path: Path, codes_name: str):
        super().__init__(None, codes_name)

        self.out_path = out_path

    def _run(self, run_mode, function, **kwargs):
        """Run openmc"""
        folder = run_mode.name.lower()
        return _timing(
            function,
            "Executed in",
            f"Running OpenMC in {folder} mode",
            debug_info_str=False,
        )(
            openmc_exec="openmc",
            **kwargs,
        )

    def run(
        self, run_mode, model: openmc.Model, _config: SourceInfo, *, debug: bool = False
    ):
        """Run stage for run task"""
        folder = run_mode.name.lower()
        cwd = Path(self.out_path, folder)
        cwd.mkdir(parents=True, exist_ok=True)
        return self._run(
            run_mode,
            model.run,
            cwd=cwd,
            path=cwd,
            output=debug,
            geometry_debug=False,
            restart_file=None,
            tracks=False,
            mpi_args=None,
            event_based=False,
            threads=None,
        )

    def plot(
        self, run_mode, model: openmc.Model, config: PlotConfig, *, debug: bool = False
    ):
        """Plot stage for run task"""
        folder = run_mode.name.lower()
        cwd = Path(self.out_path, folder)
        cwd.mkdir(parents=True, exist_ok=True)
        return FigureData(
            self._run(
                run_mode,
                model.plot,
                width=config.width,
                pixels=config.pixels,
                basis=config.basis,
                color_by=config.colour_by,
                show_overlaps=config.show_overlaps,
            ),
            cwd / "geometry.png",
        )

    def volume(
        self, run_mode, model: openmc.Model, _config: None, *, debug: bool = False
    ):
        """Stochastic volume stage for run task"""
        folder = run_mode.name.lower()
        cwd = Path(self.out_path, folder)
        cwd.mkdir(parents=True, exist_ok=True)
        return self._run(
            run_mode,
            model.calculate_volumes,
            cwd=cwd,
            path=cwd,
            output=debug,
            mpi_args=None,
        )


class OpenMCCSGTeardown(CodesTeardown):
    """Teardown task for OpenMC solver"""

    def __init__(
        self,
        cell_arrays: CellStage,
        pre_cell_model: NeutronicsReactor,
        out_path: str,
        codes_name: str,
    ):
        super().__init__(None, codes_name)

        self.out_path = out_path
        self.cell_arrays = cell_arrays
        self.pre_cell_model = pre_cell_model

    def run(
        self,
        universe,
        source_info: SourceInfo,
        statepoint_file,
    ):
        """Run stage for Teardown task"""
        result = OpenMCCSGResult.from_run(
            universe,
            self.cell_arrays,
            source_info.rate,
            source_info.triton_rate,
            Path(statepoint_file),
        )
        output_params = NeutronicsOutputParams.from_openmc_csg_result(result)

        return result, output_params

    def plot(self, _universe, _source_info, fig: FigureData, **kwargs):
        """Plot stage for Teardown task"""
        fig.axis.get_figure().savefig(fig.path)
        return fig.axis

    def volume(
        self,
        _universe,
        _source_params,
        _statepoint_file,
    ) -> dict[int, float]:
        """Stochastic volume stage for teardown task"""
        return {
            cell.id: raw_uc(
                np.nan if cell.volume is None else cell.volume, "cm^3", "m^3"
            )
            for cell in self.cell_arrays.cells
        }


class OpenMCDAGTeardown(CodesTeardown):
    def __init__(self, out_path: str, codes_name: str):
        super().__init__(None, codes_name)

        self.out_path = out_path

    def run(
        self,
        universe,
        source_info: SourceInfo,
        statepoint_file,
    ):
        """Run stage for Teardown task"""
        result = OpenMCDAGMCResult.from_run(
            universe, source_info.rate, source_info.triton_rate, Path(statepoint_file)
        )
        output_params = NeutronicsOutputParams.from_openmc_dag_result(result)

        return result, output_params

    def plot(
        self,
        _universe,
        _source_info,
        fig: FigureData,
        **_kwargs,
    ):
        """Plot stage for Teardown task"""
        fig.axis.get_figure().save_fig(fig.path)
        return fig.axis

    def volume(
        self,
        _universe,
        _source_params,
        _statepoint_file,
    ) -> dict[int, float]:
        """Stochastic volume stage for teardown task"""
        raise NotImplementedError


TALLY_FUNCTION_TYPE = Callable[
    [list[openmc.Material], CellStage | openmc.Geometry],
    tuple[
        str,
        str,
        list[openmc.CellFilter | openmc.MaterialFilter | openmc.ParticleFilter],
    ],
]


class OpenMCNeutronicsSolver(CodesSolver, ABC):
    """OpenMC 2D neutronics solver"""

    params: OpenMCNeutronicsSolverParams
    setup_cls: type[OpenMCBaseSetup]
    teardown_cls: type[CodesTeardown]

    name: str = OPENMC_NAME
    param_cls: type[OpenMCNeutronicsSolverParams] = OpenMCNeutronicsSolverParams
    run_mode_cls: type[OpenMCRunModes] = OpenMCRunModes
    run_cls: type[OpenMCRun] = OpenMCRun

    def __init__(
        self,
        params: dict | ParameterFrame,
        build_config: dict,
        eq: Equilibrium,
        source: NeutronSourceCreator,
    ):
        self.params = make_parameter_frame(params, self.param_cls)
        self.build_config = build_config

        self.out_path = self.build_config.get("neutronics_output_path", Path.cwd())

        self.eq = eq
        self.source = source

    @property
    def source(self) -> NeutronSourceCreator:
        """Source term for OpenMC"""
        return self._source

    @source.setter
    def source(self, value: NeutronSourceCreator):
        self._source = value

    @property
    def tally_function(self) -> TALLY_FUNCTION_TYPE:
        """Function used to set up tallies"""
        return self._tally_function

    @tally_function.setter
    def tally_function(self, value: TALLY_FUNCTION_TYPE):
        self._tally_function = value

    def execute(self, run_mode, *, debug=False) -> OpenMCCSGResult | dict[int, float]:
        """Execute the setup, run, and teardown tasks, in order."""
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
    ) -> OpenMCCSGResult | dict[int, float]:
        result = None
        if setup := self._get_execution_method(self._setup, run_mode):
            model, config = setup(
                run_mode,
                runtime_params,
                self.eq,
                source_params,
                self.tally_function,
                debug=debug,
            )
        if run := self._get_execution_method(self._run, run_mode):
            result = run(run_mode, model, config, debug=debug)
        if teardown := self._get_execution_method(self._teardown, run_mode):
            result = teardown(self._setup.universe, config, result)
        return result


class OpenMCCSGNeutronicsSolver(OpenMCNeutronicsSolver):
    setup_cls: type[OpenMCCSGSetup] = OpenMCCSGSetup
    teardown_cls: type[CodesTeardown] = OpenMCCSGTeardown

    def __init__(
        self,
        params: dict | ParameterFrame,
        build_config: dict,
        eq: Equilibrium,
        source: NeutronSourceCreator,
        neutronics_model: NeutronicsReactor,
        op_cond: OperationalConditions,
        tally_function: TALLY_FUNCTION_TYPE | None = None,
    ):
        super().__init__(
            params,
            build_config,
            eq,
            source,
        )
        self.neutronics_model = neutronics_model
        self.materials = MaterialsLibrary.from_neutronics_materials(
            self.neutronics_model.material_library, op_cond
        )

        self.cell_arrays = make_cell_arrays(
            self.neutronics_model,
            BluemiraNeutronicsCSG(),
            self.materials,
            control_id=True,
        )

        self.tally_function = (
            csg_filter_cells if tally_function is None else tally_function
        )

    def _single_run(
        self,
        run_mode: OpenMCRunModes,
        source_params: PlasmaSourceParameters,
        runtime_params: OpenMCSimulationRuntimeParameters,
        *,
        debug=False,
    ) -> OpenMCCSGResult | dict[int, float]:
        self._setup = self.setup_cls(
            self.name,
            str(self.build_config["cross_section_xml"]),
            self.eq,
            self.source,
            self.materials,
            self.cell_arrays,
            self.neutronics_model,
        )

        self._run = self.run_cls(self.out_path, self.name)
        self._teardown = self.teardown_cls(
            self.cell_arrays,
            self.neutronics_model,
            self.out_path,
            self.name,
        )

        return super()._single_run(run_mode, source_params, runtime_params, debug=debug)


class OpenMCDAGMCNeutronicsSolver(OpenMCNeutronicsSolver):
    setup_cls: type[OpenMCDAGSetup] = OpenMCDAGSetup
    teardown_cls: type[CodesTeardown] = OpenMCDAGTeardown

    def __init__(
        self,
        params: dict | ParameterFrame,
        build_config: dict,
        eq: Equilibrium,
        source: NeutronSourceCreator,
        dagmc_model_path: Path,
        materials,
        tally_function: TALLY_FUNCTION_TYPE | None = None,
    ):
        super().__init__(
            params,
            build_config,
            eq,
            source,
        )
        self.dagmc_model_path = dagmc_model_path
        self.materials = materials

        self.tally_function = dagmc_tallys if tally_function is None else tally_function

    def _single_run(
        self,
        run_mode: OpenMCRunModes,
        source_params: PlasmaSourceParameters,
        runtime_params: OpenMCSimulationRuntimeParameters,
        *,
        debug=False,
    ) -> OpenMCCSGResult | dict[int, float]:
        self._setup = self.setup_cls(
            self.name,
            str(self.build_config["cross_section_xml"]),
            self.eq,
            self.source,
            self.materials,
            self.dagmc_model_path,
        )

        self._run = self.run_cls(self.out_path, self.name)
        self._teardown = self.teardown_cls(
            self.out_path,
            self.name,
        )
        return super()._single_run(run_mode, source_params, runtime_params, debug=debug)
