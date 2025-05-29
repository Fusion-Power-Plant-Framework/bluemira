"""Optimised Reactor Example"""

import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import openmc
import vtk
from vtkmodules.util import numpy_support

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.designer import Designer
from bluemira.base.logs import set_log_level
from bluemira.base.look_and_feel import bluemira_print
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.base.reactor import ComponentManager, Reactor
from bluemira.builders.plasma import Plasma, PlasmaBuilder
from bluemira.builders.tools import apply_component_display_options
from bluemira.display import show_cad
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.display.plotter import PlotOptions, plot_2d
from bluemira.equilibria.coils import Coil, CoilSet
from bluemira.equilibria.diagnostics import PicardDiagnostic, PicardDiagnosticOptions
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.optimisation.constraints import (
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
)
from bluemira.equilibria.optimisation.problem import (
    UnconstrainedTikhonovCurrentGradientCOP,
)
from bluemira.equilibria.profiles import CustomProfile
from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.equilibria.solve import PicardIterator
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.optimisation import optimise_geometry
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.geometry.tools import (
    distance_to,
    interpolate_bspline,
    make_polygon,
    offset_wire,
    revolve_shape,
    sweep_shape,
)
from bluemira.geometry.wire import BluemiraWire
from bluemira.materials.cache import establish_material_cache, get_cached_material
from bluemira.optimisation import Algorithm


def lcfs_parameterisation(R_0, A):
    return JohnerLCFS({
        "r_0": {"value": R_0},
        "z_0": {"value": 0.0},
        "a": {"value": R_0 / A},
        "kappa_u": {"value": 1.6},
        "kappa_l": {"value": 1.9},
        "delta_u": {"value": 0.4},
        "delta_l": {"value": 0.4},
        "phi_u_neg": {"value": 0.0},
        "phi_u_pos": {"value": 0.0},
        "phi_l_neg": {"value": 45.0},
        "phi_l_pos": {"value": 30.0},
    })


def ref_eq(R_0, A) -> Equilibrium:  # noqa: D103
    x = [5.4, 14.0, 17.75, 17.75, 14.0, 7.0, 2.77, 2.77, 2.77, 2.77, 2.77]
    z = [9.26, 7.9, 2.5, -2.5, -7.9, -10.5, 7.07, 4.08, -0.4, -4.88, -7.86]
    dx = [0.6, 0.7, 0.5, 0.5, 0.7, 1.0, 0.4, 0.4, 0.4, 0.4, 0.4]
    dz = [0.6, 0.7, 0.5, 0.5, 0.7, 1.0, 2.99 / 2, 2.99 / 2, 5.97 / 2, 2.99 / 2, 2.99 / 2]

    coils = []
    j = 1
    for i, (xi, zi, dxi, dzi) in enumerate(zip(x, z, dx, dz, strict=False)):
        if j > 6:  # noqa: PLR2004
            j = 1
        ctype = "PF" if i < 6 else "CS"  # noqa: PLR2004
        coil = Coil(
            xi,
            zi,
            current=0,
            dx=dxi,
            dz=dzi,
            ctype=ctype,
            name=f"{ctype}_{j}",
        )
        coils.append(coil)
        j += 1

    coilset = CoilSet(*coils)
    coilset.assign_material("CS", j_max=16.5e6, b_max=12.5)
    coilset.assign_material("PF", j_max=12.5e6, b_max=11.0)

    # Later on, we will optimise the PF coil positions, but for the CS coils we can fix sizes
    # and mesh them already.

    cs = coilset.get_coiltype("CS")
    cs.fix_sizes()
    cs.discretisation = 0.3

    B_0 = 4.8901  # T
    I_p = 19.07e6  # A

    grid = Grid(3.0, 13.0, -10.0, 10.0, 65, 65)

    profiles = CustomProfile(
        np.array([
            86856,
            86506,
            84731,
            80784,
            74159,
            64576,
            52030,
            36918,
            20314,
            4807,
            0.0,
        ]),
        -np.array([
            0.125,
            0.124,
            0.122,
            0.116,
            0.106,
            0.093,
            0.074,
            0.053,
            0.029,
            0.007,
            0.0,
        ]),
        R_0=R_0,
        B_0=B_0,
        I_p=I_p,
    )

    eq = Equilibrium(coilset, grid, profiles, psi=None)

    lcfs = (
        lcfs_parameterisation(R_0, A).create_shape().discretise(byedges=True, ndiscr=50)
    )

    x_bdry, z_bdry = lcfs.x, lcfs.z
    arg_inner = np.argmin(x_bdry)

    isoflux = IsofluxConstraint(
        x_bdry,
        z_bdry,
        x_bdry[arg_inner],
        z_bdry[arg_inner],
        tolerance=0.5,  # Difficult to choose...
        constraint_value=0.0,  # Difficult to choose...
    )

    xp_idx = np.argmin(z_bdry)
    x_point = FieldNullConstraint(
        x_bdry[xp_idx],
        z_bdry[xp_idx],
        tolerance=1e-4,  # [T]
    )
    current_opt_problem = UnconstrainedTikhonovCurrentGradientCOP(
        coilset, eq, MagneticConstraintSet([isoflux, x_point]), gamma=1e-7
    )
    diagnostic_plotting = PicardDiagnosticOptions(plot=PicardDiagnostic.EQ)
    program = PicardIterator(
        eq,
        current_opt_problem,
        fixed_coils=True,
        relaxation=0.2,
        diagnostic_plotting=diagnostic_plotting,
    )
    program()

    plt.show()

    return eq


@dataclass
class OptimisedReactorParams(ParameterFrame):
    """All parameters for the OptimisedReactor."""

    # machine parameters
    n_TF: Parameter[int]
    R_0: Parameter[float]
    A: Parameter[float]
    # gaps
    g_p_bb: Parameter[float]
    g_bb_tf_min: Parameter[float]
    # thicknesses
    tk_bb: Parameter[float]
    tk_tf: Parameter[float]


class BBBuilder(Builder):
    BB = "BB"

    param_cls: type[OptimisedReactorParams] = OptimisedReactorParams
    params: OptimisedReactorParams

    def __init__(
        self,
        params: OptimisedReactorParams,
        lcfs_wire: BluemiraWire,
        material_name: str,
    ):
        super().__init__(params, {"material": {self.BB: material_name}})
        self.lcfs_wire = lcfs_wire

    def build(self) -> Component:
        inner_bb = offset_wire(self.lcfs_wire, self.params.g_p_bb.value, ndiscr=100)
        inner_bb = interpolate_bspline(inner_bb.vertexes, closed=True)
        outer_bb = offset_wire(inner_bb, self.params.tk_bb.value, ndiscr=100)
        outer_bb = interpolate_bspline(outer_bb.vertexes, closed=True)
        bb_xz = BluemiraFace([outer_bb, inner_bb])
        bb = revolve_shape(bb_xz, degree=360 / self.params.n_TF.value)
        mat = self.get_material(self.BB)
        pc_xz = PhysicalComponent(self.BB, bb_xz, mat)
        pc_xyz = PhysicalComponent(self.BB, bb, mat)
        apply_component_display_options(pc_xyz, color=BLUE_PALETTE["BB"][0])
        return self.component_tree(xz=[pc_xz], xy=[], xyz=[pc_xyz])


class BB(ComponentManager):
    def xz_face(self) -> BluemiraFace:
        return self.component().get_component("xz").get_component(BBBuilder.BB).shape


class TFDesigner(Designer[OptimisedReactorParams]):
    param_cls: type[OptimisedReactorParams] = OptimisedReactorParams
    params: OptimisedReactorParams

    def __init__(self, params, bb_xz_face: BluemiraFace):
        super().__init__(params, {})
        self.bb_xz_face = bb_xz_face

    def run(self) -> PrincetonD:
        bluemira_print("Optimising TF coil centreline...")
        gap = self.params.g_bb_tf_min.value + self.params.tk_tf.value
        p = PrincetonD({
            "x1": {"value": self.bb_xz_face.bounding_box.x_min - 2 * gap},
            "x2": {"value": self.bb_xz_face.bounding_box.x_max + 2 * gap},
        })

        initial_cl = p.create_shape()

        show_cad([self.bb_xz_face, initial_cl])

        distance_constraint = {
            "f_constraint": self._constrain_distance,
            "tolerance": np.array([1e-6]),
        }
        optimisation_result = optimise_geometry(
            algorithm=Algorithm.SLSQP,
            keep_history=True,
            opt_conditions={"max_eval": 500, "ftol_rel": 1e-6},
            geom=p,
            f_objective=lambda g: g.create_shape().length,
            ineq_constraints=[distance_constraint],
        )

        # plotting
        geom = PrincetonD()
        ax = plot_2d(self.bb_xz_face, show=False)
        ax = plot_2d(
            initial_cl,
            options=PlotOptions(wire_options={"color": "green", "linewidth": 0.8}),
            ax=ax,
            show=False,
        )
        for i, (x, _) in enumerate(optimisation_result.history):
            geom.variables.set_values_from_norm(x)
            wire = geom.create_shape()
            wire_options = {
                "alpha": 0.5 + ((i + 1) / len(optimisation_result.history)) / 2,
                "color": "red",
                "linewidth": 0.1,
            }
            ax = plot_2d(
                wire, options=PlotOptions(wire_options=wire_options), ax=ax, show=False
            )
        plot_2d(optimisation_result.geom.create_shape(), ax=ax, show=True)

        return optimisation_result.geom

    def _constrain_distance(self, geom: PrincetonD) -> float:
        bb_ob_wire = self.bb_xz_face.boundary[0]
        min_dist = self.params.g_bb_tf_min.value + self.params.tk_tf.value
        r = min_dist - distance_to(geom.create_shape(), bb_ob_wire)[0]
        g = r
        if r > 0:
            g = math.exp(10 * r) - 1
        return g


class TFBuilder(Builder):
    TF = "TF"

    param_cls: type[OptimisedReactorParams] = OptimisedReactorParams
    params: OptimisedReactorParams

    def __init__(
        self,
        params: OptimisedReactorParams,
        centerline: PrincetonD,
        material_name: str,
    ):
        super().__init__(params, {"material": {self.TF: material_name}})
        self.centerline = centerline

    def build(self) -> Component:
        p = self.centerline

        x2 = p.variables.x2
        dx = self.params.tk_tf.value / 2
        dy = self.params.tk_tf.value / 2
        profile = make_polygon(
            [
                [p.variables.x2 - dx, -dy, 0],
                [x2 + dx, -dy, 0],
                [x2 + dx, dy, 0],
                [x2 - dx, dy, 0],
            ],
            closed=True,
        )

        tf_cl = p.create_shape()

        tf_sweep = sweep_shape(profile, tf_cl)
        pc_xyz = PhysicalComponent(self.TF, tf_sweep, self.get_material(self.TF))
        apply_component_display_options(pc_xyz, color=BLUE_PALETTE["TF"][0])
        return self.component_tree(xz=[], xy=[], xyz=[pc_xyz])


class OptimisedReactor(Reactor):  # noqa: D101
    plasma: Plasma
    bb: BB
    tf: ComponentManager

    def __init__(self, reactor_params: OptimisedReactorParams):
        """Initialise the optimised reactor."""
        self.params = reactor_params
        super().__init__("OptimisedReactor", n_sectors=reactor_params.n_TF.value)
        establish_material_cache([
            Path(__file__).parent / "materials_data" / "materials.json",
            Path(__file__).parent / "materials_data" / "mixtures.json",
        ])

    def build_plasma(self) -> Plasma:
        _rf_eq = ref_eq(self.params.R_0.value, self.params.A.value)

        lcfs_wire = lcfs_parameterisation(
            self.params.R_0.value, self.params.A.value
        ).create_shape()
        self.plasma = Plasma(PlasmaBuilder(self.params, {}, lcfs_wire).build())

    def build_bb(self, mat_name: str) -> None:
        lcfs = self.plasma.lcfs()
        self.bb = BB(BBBuilder(self.params, lcfs, mat_name).build())

    def build_tf_coils(self, mat_name: str) -> None:
        bb_face = self.bb.xz_face()
        centerline = TFDesigner(self.params, bb_face).run()
        self.tf = ComponentManager(TFBuilder(self.params, centerline, mat_name).build())


set_log_level("INFO")

build = True
show = True
save = True
run_openmc = True
extract_results = True

r = OptimisedReactor(
    OptimisedReactorParams(
        n_TF=Parameter("n_TF", 16, "dimensionless", "Number of TF coils"),
        R_0=Parameter("R_0", 9, "m", "Major radius of the plasma"),
        A=Parameter("A", 3, "dimensionless", "Aspect ratio of the plasma"),
        g_p_bb=Parameter(
            "g_p_bb", 0.3, "m", "Gap between the plasma and the vacuum vessel"
        ),
        g_bb_tf_min=Parameter(
            "g_bb_tf_min",
            0.5,
            "m",
            "Minimum gap between the vacuum vessel and the TF coils",
        ),
        tk_bb=Parameter("tk_bb", 0.4, "m", "Thickness of the vacuum vessel"),
        tk_tf=Parameter("tk_tf", 0.4, "m", "Thickness of the TF coil WP"),
    )
)

if build:
    r.build_plasma()
    r.build_bb("Homogenised_HCPB_2015_v3_BZ")
    r.build_tf_coils("Toroidal_Field_Coil_2015")

if show:
    r.show_cad(construction_params={"n_sectors": 8})

if save:
    r.save_cad(
        cad_format="dagmc",
        construction_params={
            "without_components": [r.plasma],
        },
    )


def numpy_to_vtk(data, output_name, scaling=(1, 1, 1)):
    data_type = vtk.VTK_FLOAT
    shape = data.shape

    flat_data_array = data.flatten()
    vtk_data = numpy_support.numpy_to_vtk(
        num_array=flat_data_array, deep=True, array_type=data_type
    )
    vtk_data.SetName(output_name)

    half_x = int(0.5 * scaling[0] * (shape[0] - 1))
    half_y = int(0.5 * scaling[1] * (shape[1] - 1))
    half_z = int(0.5 * scaling[2] * (shape[2] - 1))

    img = vtk.vtkImageData()
    img.GetPointData().SetScalars(vtk_data)
    img.SetSpacing(scaling[0], scaling[1], scaling[2])
    img.SetDimensions(shape[0], shape[1], shape[2])
    img.SetOrigin(-half_x, -half_y, -half_z)

    # Save the VTK file
    writer = vtk.vtkXMLImageDataWriter()
    writer.SetFileName(f"{output_name}.vti")
    writer.SetInputData(img)
    writer.Write()


par = Path(__file__).parent
omc_output_path = par / "omc"
dag_model_path = par / "OptimisedReactor.h5m"
meta_data_path = par / "OptimisedReactor.meta.json"

# Used in extract_results too
n_batches = 5
dagmc_univ = openmc.DAGMCUniverse(
    filename=dag_model_path.as_posix(),
    auto_geom_ids=True,
).bounded_universe()

if run_openmc:
    # load model materials
    with open(meta_data_path) as meta_file:
        bom = json.load(meta_file)["bom"]
    openmc_mats = [
        get_cached_material(mat_name).to_openmc_material(temperature=294)
        for mat_name in bom
    ]

    # load DAG model
    geometry = openmc.Geometry(dagmc_univ)

    # source and settings
    major_radius = r.params.R_0.value * 100
    aspect_ratio = r.params.A.value
    minor_radius = major_radius / aspect_ratio
    bluemira_print(
        "Plasma parameters: "
        f"Major radius: {major_radius:.1f} cm, "
        f"minor radius: {minor_radius:.1f} cm, "
        f"aspect ratio: {aspect_ratio:.2f}"
    )

    # based on https://github.com/fusion-energy/magnetic_fusion_openmc_dagmc_paramak_example
    radius = openmc.stats.Discrete(
        [major_radius - minor_radius, major_radius + minor_radius], [1, 1]
    )
    z_values = openmc.stats.Discrete([-minor_radius, minor_radius], [1, 1])
    angle = openmc.stats.Uniform(a=0.0, b=math.radians(360))
    my_source = openmc.IndependentSource(
        space=openmc.stats.CylindricalIndependent(
            r=radius, phi=angle, z=z_values, origin=(0.0, 0.0, 0.0)
        ),
        angle=openmc.stats.Isotropic(),
        energy=openmc.stats.muir(e0=14080000.0, m_rat=5.0, kt=20000.0),
    )

    settings = openmc.Settings()
    settings.batches = n_batches
    settings.particles = 10000
    settings.inactive = 0
    settings.run_mode = "fixed source"
    settings.source = my_source
    settings.output = {"path": omc_output_path.as_posix()}

    # TALLIES

    # record the heat deposited in entire geometry
    heating_cell_tally = openmc.Tally(name="heating")
    heating_cell_tally.scores = ["heating"]

    # record the total TBR
    tbr_cell_tally = openmc.Tally(name="tbr")
    tbr_cell_tally.scores = ["(n,Xt)"]

    # mesh that covers the geometry
    mesh = openmc.RegularMesh.from_domain(geometry, dimension=(100, 100, 100))
    mesh_filter = openmc.MeshFilter(mesh)

    # mesh tally using the previously created mesh and records heating on the mesh
    heating_mesh_tally = openmc.Tally(name="heating_on_mesh")
    heating_mesh_tally.filters = [mesh_filter]
    heating_mesh_tally.scores = ["heating"]

    # mesh tally using the previously created mesh and records TBR on the mesh
    tbr_mesh_tally = openmc.Tally(name="tbr_on_mesh")
    tbr_mesh_tally.filters = [mesh_filter]
    tbr_mesh_tally.scores = ["(n,Xt)"]

    tallies = openmc.Tallies([
        tbr_cell_tally,
        tbr_mesh_tally,
        heating_cell_tally,
        heating_mesh_tally,
    ])

    model = openmc.Model(
        materials=openmc_mats,
        geometry=geometry,
        tallies=tallies,
        settings=settings,
    )
    model.export_to_model_xml()
    sp_path = model.run()
    print(f"OpenMC completed, results saved to {sp_path}")
else:
    sp_path = omc_output_path / f"statepoint.{n_batches}.h5"
    sp_path = omc_output_path / "statepoint.10.h5"

if extract_results:
    sp = openmc.StatePoint(sp_path.as_posix())

    tbr_cell_tally = sp.get_tally(name="tbr")
    tbr_mesh_tally = sp.get_tally(name="tbr_on_mesh")
    heating_cell_tally = sp.get_tally(name="heating")
    heating_mesh_tally = sp.get_tally(name="heating_on_mesh")

    bluemira_print(f"The reactor has a TBR of {tbr_cell_tally.mean.sum()}")
    bluemira_print(f"Standard deviation on the TBR is {tbr_cell_tally.std_dev.sum()}")

    bluemira_print(
        f"The heating of {heating_cell_tally.mean.sum() / 1e6} MeV per source particle is deposited"
    )
    bluemira_print(
        f"Standard deviation on the heating tally is {heating_cell_tally.std_dev.sum()}"
    )

    mesh = tbr_mesh_tally.find_filter(openmc.MeshFilter).mesh
    mesh.write_data_to_vtk(
        filename="tbr_mesh_mean.vtk",
        datasets={
            "mean": tbr_mesh_tally.mean
        },  # the first "mean" is the name of the data set label inside the vtk file
    )

    model_w = dagmc_univ.bounding_box.width
    # not working for some reason
    # tbr_mesh_mean = tbr_mesh_tally.mean
    # tbr_mesh_mean = tbr_mesh_mean.reshape(100, 100, 100)
    # scaling = tuple(
    #     round(t / c) for c, t in zip(tbr_mesh_mean.shape, model_w, strict=False)
    # )
    # numpy_to_vtk(tbr_mesh_mean, "tbr_mesh_mean", scaling)

    heating_mesh_mean = heating_mesh_tally.mean.reshape(100, 100, 100)
    scaling = tuple(
        round(t / c) for c, t in zip(heating_mesh_mean.shape, model_w, strict=False)
    )
    numpy_to_vtk(heating_mesh_mean, "heating_mesh_mean", scaling)
