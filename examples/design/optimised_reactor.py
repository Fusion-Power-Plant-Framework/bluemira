from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bluemira.base.builder import Builder
from bluemira.base.components import Component, PhysicalComponent
from bluemira.base.designer import Designer
from bluemira.base.logs import set_log_level
from bluemira.base.look_and_feel import bluemira_print
from bluemira.base.parameter_frame import Parameter, ParameterFrame
from bluemira.base.reactor import ComponentManager, Reactor
from bluemira.builders.plasma import Plasma, PlasmaBuilder
from bluemira.builders.tools import apply_component_display_options
from bluemira.display.palettes import BLUE_PALETTE
from bluemira.equilibria.coils import Coil, CoilSet
from bluemira.equilibria.diagnostics import PicardDiagnostic, PicardDiagnosticOptions
from bluemira.equilibria.equilibrium import Equilibrium
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.optimisation.constraints import (
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
)
from bluemira.equilibria.optimisation.problem._tikhonov import (
    UnconstrainedTikhonovCurrentGradientCOP,
)
from bluemira.equilibria.profiles import CustomProfile
from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.equilibria.solve import PicardIterator
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.geometry.tools import (
    interpolate_bspline,
    make_polygon,
    offset_wire,
    revolve_shape,
    sweep_shape,
)
from bluemira.geometry.wire import BluemiraWire
from bluemira.materials.cache import establish_material_cache


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
    diagnostic_plotting = PicardDiagnosticOptions(plot=PicardDiagnostic.NO_PLOT)
    program = PicardIterator(
        eq,
        current_opt_problem,
        fixed_coils=True,
        relaxation=0.2,
        diagnostic_plotting=diagnostic_plotting,
    )
    program()

    return eq


@dataclass
class OptimisedReactorParams(ParameterFrame):
    """All parameters for the OptimisedReactor."""

    # machine parameters
    n_TF: Parameter[int]
    R_0: Parameter[float]
    A: Parameter[float]
    # gaps
    g_p_vv: Parameter[float]
    g_vv_tf: Parameter[float]
    # thicknesses
    tk_vv: Parameter[float]
    tk_tf: Parameter[float]


class VVBuilder(Builder):
    VV = "VV"

    param_cls: type[OptimisedReactorParams] = OptimisedReactorParams
    params: OptimisedReactorParams

    def __init__(self, params: OptimisedReactorParams, lcfs_wire: BluemiraWire):
        super().__init__(params, {"material": {self.VV: "SS316-LN"}})
        self.lcfs_wire = lcfs_wire

    def build(self) -> Component:
        inner_vv = offset_wire(self.lcfs_wire, self.params.g_p_vv.value, ndiscr=100)
        inner_vv = interpolate_bspline(inner_vv.vertexes, closed=True)
        outer_vv = offset_wire(inner_vv, self.params.tk_vv.value, ndiscr=100)
        outer_vv = interpolate_bspline(outer_vv.vertexes, closed=True)
        vv_xz = BluemiraFace([outer_vv, inner_vv])
        vv = revolve_shape(vv_xz, degree=360 / self.params.n_TF.value)
        mat = self.get_material(self.VV)
        pc_xz = PhysicalComponent(self.VV, vv_xz, mat)
        pc_xyz = PhysicalComponent(self.VV, vv, mat)
        apply_component_display_options(pc_xyz, color=BLUE_PALETTE["VV"][0])
        return self.component_tree(xz=[pc_xz], xy=[], xyz=[pc_xyz])


class VV(ComponentManager):
    def xz_face(self) -> BluemiraFace:
        return (
            self.component()
            .get_component("xz")
            .get_component(VVBuilder.VV)
            .shape.boundary[0]
        )


class TFDesigner(Designer):
    pass


class TFBuilder(Builder):
    TF = "TF"

    param_cls: type[OptimisedReactorParams] = OptimisedReactorParams
    params: OptimisedReactorParams

    def __init__(self, params: OptimisedReactorParams, vv_xz_face: BluemiraFace):
        super().__init__(params, {"material": {self.TF: "Toroidal_Field_Coil_2015"}})
        self.vv_xz_face = vv_xz_face

    def build(self) -> Component:
        p = PrincetonD({
            "x1": {
                "value": self.vv_xz_face.bounding_box.x_min - self.params.g_vv_tf.value
            },
            "x2": {
                "value": self.vv_xz_face.bounding_box.x_max + self.params.g_vv_tf.value
            },
        })
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
    vv: VV
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
        # don't use, just show the solve.
        _rf_eq = ref_eq(self.params.R_0.value, self.params.A.value)

        lcfs_wire = lcfs_parameterisation(
            self.params.R_0.value, self.params.A.value
        ).create_shape()
        self.plasma = Plasma(PlasmaBuilder(self.params, {}, lcfs_wire).build())

    def build_vv(self) -> None:
        lcfs = self.plasma.lcfs()
        self.vv = VV(VVBuilder(self.params, lcfs).build())

    def build_tf_coils(self) -> None:
        vv_face = self.vv.xz_face()
        self.tf = ComponentManager(TFBuilder(self.params, vv_face).build())


set_log_level("INFO")


r = OptimisedReactor(
    OptimisedReactorParams(
        n_TF=Parameter("n_TF", 16, "dimensionless", "Number of TF coils"),
        R_0=Parameter("R_0", 8.938, "m", "Major radius of the plasma"),
        A=Parameter("A", 3.1, "dimensionless", "Aspect ratio of the plasma"),
        g_p_vv=Parameter(
            "g_p_vv", 0.5, "m", "Gap between the plasma and the vacuum vessel"
        ),
        g_vv_tf=Parameter(
            "g_vv_tf", 0.5, "m", "Gap between the vacuum vessel and the TF coils"
        ),
        tk_vv=Parameter("tk_vv", 0.5, "m", "Thickness of the vacuum vessel"),
        tk_tf=Parameter("tk_tf", 0.5, "m", "Thickness of the TF coil WP"),
    )
)

r.build_plasma()
r.build_vv()
r.build_tf_coils()

major_rad = r.params.R_0.value
aspect_ratio = r.params.A.value
minor_rad = major_rad / aspect_ratio
bluemira_print(
    "Plasma parameters: "
    f"Major radius: {major_rad:.2f} m, "
    f"minor radius: {minor_rad:.2f} m, "
    f"aspect ratio: {aspect_ratio:.2f}"
)

# exit()
r.show_cad(construction_params={"n_sectors": 8})
r.save_cad(
    cad_format="dagmc",
    construction_params={
        "without_components": [r.plasma],
    },
)
