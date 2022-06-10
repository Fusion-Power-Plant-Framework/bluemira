# bluemira is an integrated inter-disciplinary design tool for future fusion
# reactors. It incorporates several modules, some of which rely on other
# codes, to carry out a range of typical conceptual fusion reactor design
# activities.
#
# Copyright (C) 2021 M. Coleman, J. Cook, F. Franza, I.A. Maione, S. McIntosh, J. Morris,
#                    D. Short
#
# bluemira is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# bluemira is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with bluemira; if not, see <https://www.gnu.org/licenses/>.

"""
EU-DEMO specific builder for PF coils
"""

from typing import List, Optional, Type

import numpy as np

import bluemira.utilities.plot_tools as bm_plot_tools
from bluemira.base.builder import BuildConfig, Builder
from bluemira.base.components import Component
from bluemira.base.config import Configuration
from bluemira.base.error import BuilderError
from bluemira.base.look_and_feel import bluemira_print, bluemira_warn
from bluemira.builders.pf_coils import PFCoilBuilder
from bluemira.equilibria.coils import Coil, CoilSet
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.opt_constraints import (
    CoilFieldConstraints,
    CoilForceConstraints,
    FieldNullConstraint,
    IsofluxConstraint,
    MagneticConstraintSet,
)
from bluemira.equilibria.opt_problems import (
    BreakdownCOP,
    OutboardBreakdownZoneStrategy,
    PulsedNestedPositionCOP,
    TikhonovCurrentCOP,
)
from bluemira.equilibria.profiles import BetaIpProfile
from bluemira.equilibria.run import OptimisedPulsedCoilsetDesign, PulsedCoilsetDesign
from bluemira.equilibria.shapes import JohnerLCFS
from bluemira.equilibria.solve import DudsonConvergence
from bluemira.geometry.constants import VERY_BIG
from bluemira.geometry.face import BluemiraFace
from bluemira.geometry.tools import (
    boolean_cut,
    distance_to,
    make_polygon,
    offset_wire,
    split_wire,
)
from bluemira.geometry.wire import BluemiraWire
from bluemira.magnetostatics.baseclass import SourceGroup
from bluemira.magnetostatics.circular_arc import CircularArcCurrentSource
from bluemira.utilities.optimiser import Optimiser
from bluemira.utilities.positioning import PathInterpolator, PositionMapper


class PFCoilsComponent(Component):
    """
    Poloidal field coils component, with a solver for the magnetic field from all of the
    PF coils.

    Parameters
    ----------
    name: str
        Name of the component
    parent: Optional[Component] = None
        Parent component
    children: Optional[List[Component]] = None
        List of child components
    field_solver: Optional[CurrentSource]
        Magnetic field solver
    """

    def __init__(
        self,
        name: str,
        parent: Optional[Component] = None,
        children: Optional[List[Component]] = None,
        field_solver=None,
    ):
        super().__init__(name, parent=parent, children=children)
        self._field_solver = field_solver

    def field(self, x, y, z):
        """
        Calculate the magnetic field due to the TF coils at a set of points.

        Parameters
        ----------
        x: Union[float, np.array]
            The x coordinate(s) of the points at which to calculate the field
        y: Union[float, np.array]
            The y coordinate(s) of the points at which to calculate the field
        z: Union[float, np.array]
            The z coordinate(s) of the points at which to calculate the field
        Returns
        -------
        field: np.array
            The magnetic field vector {Bx, By, Bz} in [T]
        """
        return self._field_solver.field(x, y, z)


class PFCoilsBuilder(Builder):
    """
    Builder for the PF Coils.
    """

    _required_params: List[str] = [
        "tk_pf_insulation",
        "tk_pf_casing",
        "tk_cs_insulation",
        "tk_cs_casing",
        "r_pf_corner",
        "r_cs_corner",
        "r_cs_in",
        "tk_cs",
        "tk_sol_ib",
        "tk_sol_ob",
        "g_cs_mod",
        "R_0",
        "A",
        "I_p",
        "B_0",
        "beta_p",
        "l_i",
        "C_Ejima",
        "tau_flattop",
        "v_burn",
        "kappa",
        "delta",
        "n_CS",
        "n_PF",
        "CS_jmax",
        "PF_jmax",
        "CS_bmax",
        "PF_bmax",
        "F_pf_zmax",
        "F_cs_ztotmax",
        "F_cs_sepmax",
        "B_premag_stray_max",
    ]
    _required_config: List[str] = []
    _params: Configuration
    _param_class: Type[CoilSet]
    _default_runmode: str = "read"
    _problem_class: Type[PulsedCoilsetDesign] = OptimisedPulsedCoilsetDesign

    def __init__(
        self,
        params,
        build_config: BuildConfig,
        tf_coil_boundary: Optional[BluemiraWire] = None,
        keep_out_zones: Optional[List[BluemiraFace]] = None,
    ):
        super().__init__(
            params,
            build_config,
            tf_coil_boundary=tf_coil_boundary,
            keep_out_zones=keep_out_zones,
        )

    def _extract_config(self, build_config: BuildConfig):
        super()._extract_config(build_config)

        if self._runmode.name.lower() == "read":
            if build_config.get("eqdsk_path") is None:
                raise BuilderError(
                    "Must supply eqdsk_path in build_config when using 'read' mode."
                )
            self._eqdsk_path = build_config["eqdsk_path"]

        elif self._runmode.name.lower() == "run":
            # TODO: Process build_config properly here
            pass

        return super()._extract_config(build_config)

    def reinitialise(
        self,
        params,
        tf_coil_boundary: Optional[BluemiraWire] = None,
        keep_out_zones: Optional[List[BluemiraFace]] = None,
        **kwargs,
    ) -> None:
        """
        Initialise the state of this builder ready for a new run.

        Parameters
        ----------
        params: Dict[str, Any]
            The parameterisation containing at least the required params for this
            Builder.
        """
        super().reinitialise(params, **kwargs)

        self._reset_params(params)
        self._coilset = None
        self._tf_coil_boundary = tf_coil_boundary
        self._keep_out_zones = keep_out_zones

    def run(self):
        """
        Build PF coils from a design optimisation problem.
        """
        # Make initial CoilSet
        coilset = make_coilset(
            self._tf_coil_boundary,
            R_0=self._params.R_0.value,
            kappa=self._params.kappa.value,
            delta=self._params.delta.value,
            r_cs=self._params.r_cs_in.value + self._params.tk_cs.value / 2,
            tk_cs=self._params.tk_cs.value / 2,
            g_cs=self._params.g_cs_mod.value,
            tk_cs_ins=self._params.tk_cs_insulation.value,
            tk_cs_cas=self._params.tk_cs_casing.value,
            n_CS=self._params.n_CS.value,
            n_PF=self._params.n_PF.value,
            CS_jmax=self._params.CS_jmax.value,
            CS_bmax=self._params.CS_bmax.value,
            PF_jmax=self._params.PF_jmax.value,
            PF_bmax=self._params.PF_bmax.value,
        )

        # Get an offset from the TF that corresponds to a PF coil half-width of a
        # current equal to Ip
        offset_value = 0.5 * np.sqrt(self._params.I_p.value / self._params.PF_jmax.value)
        pf_coil_path = make_pf_coil_path(self._tf_coil_boundary, offset_value)
        pf_coil_names = coilset.get_PF_names()
        pf_coils = [coilset.coils[name] for name in pf_coil_names]
        position_mapper = make_coil_mapper(pf_coil_path, self._keep_out_zones, pf_coils)

        grid = make_grid(
            self._params.R_0.value, self._params.A.value, self._params.kappa.value
        )

        # TODO: Make a CustomProfile from flux functions coming from PLASMOD and fixed
        # boundary optimisation
        profiles = BetaIpProfile(
            self._params.beta_p.value,
            self._params.I_p.value * 1e6,
            self._params.R_0.value,
            self._params.B_0.value,
        )

        # Make Constraints - a tomar por culo que lo hago todo aqui y que se jodan
        kappa = self._params.kappa.value
        kappa_ul_tweak = 0.05
        kappa_u = (1 - kappa_ul_tweak) * kappa
        kappa_l = (1 + kappa_ul_tweak) * kappa
        lcfs_parameterisation = JohnerLCFS(
            {
                "r_0": {"value": self._params.R_0.value},
                "z_0": {"value": 0.0},
                "a": {"value": self._params.R_0.value / self._params.A.value},
                "kappa_u": {"value": kappa_u},
                "kappa_l": {"value": kappa_l},
                "delta_u": {"value": self._params.delta.value},
                "delta_l": {"value": self._params.delta.value},
                "phi_u_neg": {"value": 0.0},
                "phi_u_pos": {"value": 0.0},
                "phi_l_neg": {"value": 45.0},
                "phi_l_pos": {"value": 30.0},
            }
        )
        lcfs = lcfs_parameterisation.create_shape().discretize(byedges=True, ndiscr=50)
        x_lcfs, z_lcfs = lcfs.x, lcfs.z
        arg_inner = np.argmin(x_lcfs)
        arg_xp = np.argmin(z_lcfs)

        isoflux = IsofluxConstraint(x_lcfs, z_lcfs, x_lcfs[arg_inner], z_lcfs[arg_inner])
        x_point = FieldNullConstraint(x_lcfs[arg_xp], z_lcfs[arg_xp], tolerance=1e-4)
        current_opt_constraints = [
            x_point,
            CoilFieldConstraints(coilset, coilset.get_max_fields(), tolerance=1e-6),
            CoilForceConstraints(
                coilset,
                self._params.F_pf_zmax.value,
                self._params.F_cs_ztotmax.value,
                self._params.F_cs_sepmax.value,
                tolerance=1e-3,
            ),
        ]

        equilibrium_constraints = MagneticConstraintSet([isoflux])

        self._design_problem = self._problem_class(
            self._params,
            coilset,
            position_mapper,
            grid,
            current_opt_constraints,
            equilibrium_constraints,
            profiles=profiles,
            breakdown_strategy_cls=OutboardBreakdownZoneStrategy,
            breakdown_problem_cls=BreakdownCOP,
            breakdown_optimiser=Optimiser(
                "COBYLA", opt_conditions={"max_eval": 5000, "ftol_rel": 1e-10}
            ),
            breakdown_settings={"B_stray_con_tol": 1e-8, "n_B_stray_points": 20},
            equilibrium_problem_cls=TikhonovCurrentCOP,
            equilibrium_optimiser=Optimiser(
                "SLSQP", opt_conditions={"max_eval": 1000, "ftol_rel": 1e-6}
            ),
            equilibrium_convergence=DudsonConvergence(1e-3),
            equilibrium_settings={"gamma": 0.0, "relaxation": 0.1},
            position_problem_cls=PulsedNestedPositionCOP,
            position_optimiser=Optimiser(
                "COBYLA", opt_conditions={"max_eval": 100, "ftol_rel": 1e-4}
            ),
            limiter=None,
        )
        bluemira_print(
            f"Solving design problem: {self._design_problem.__class__.__name__}"
        )
        self._coilset = self._design_problem.optimise_positions()

    def read(self, **kwargs):
        """
        Build PF coils from a equilibrium file.
        """
        self._coilset = CoilSet.from_eqdsk(self._eqdsk_path)

    def mock(self, coilset):
        """
        Build PF coils from a CoilSet.
        """
        self._coilset = coilset

    def build(self, label: str = "PF Coils", **kwargs) -> PFCoilsComponent:
        """
        Build the PF Coils component.

        Returns
        -------
        component: PFCoilsComponent
            The Component built by this builder.
        """
        super().build(**kwargs)

        self.sub_components = []
        for coil in self._coilset.coils.values():
            if coil.ctype == "PF":
                r_corner = self.params.r_pf_corner
                tk_ins = self.params.tk_pf_insulation
                tk_cas = self.params.tk_pf_casing
            elif coil.ctype == "CS":
                r_corner = self.params.r_cs_corner
                tk_ins = self.params.tk_cs_insulation
                tk_cas = self.params.tk_cs_casing
            else:
                raise BuilderError(f"Unrecognised coil type {coil.ctype}.")

            sub_comp = PFCoilBuilder(coil, r_corner, tk_ins, tk_cas, coil.ctype)
            self.sub_components.append(sub_comp)

        field_solver = self._make_field_solver()
        component = PFCoilsComponent(self.name, field_solver=field_solver)

        component.add_child(self.build_xz())
        component.add_child(self.build_xy())
        component.add_child(self.build_xyz())
        return component

    def build_xy(self):
        """
        Build the x-y components of the PF coils.

        Returns
        -------
        component: Component
            The component grouping the results in the xy plane.
        """
        xy_comps = []
        comp: PFCoilBuilder
        for comp in self.sub_components:
            xy_comps.append(comp.build_xy())
        component = Component("xy", children=xy_comps)
        bm_plot_tools.set_component_view(component, "xy")
        return component

    def build_xz(self):
        """
        Build the x-z components of the PF coils.

        Returns
        -------
        component: Component
            The component grouping the results in the xz plane.
        """
        xz_comps = []
        comp: PFCoilBuilder
        for comp in self.sub_components:
            xz_comps.append(comp.build_xz())
        component = Component("xz", children=xz_comps)
        bm_plot_tools.set_component_view(component, "xz")
        return component

    def build_xyz(self, degree: float = 360.0):
        """
        Build the x-y-z components of the PF coils.

        Parameters
        ----------
        degree: float
            The angle [°] around which to build the components, by default 360.0.

        Returns
        -------
        component: Component
            The component grouping the results in 3D (xyz).
        """
        xyz_comps = []
        comp: PFCoilBuilder
        for comp in self.sub_components:
            xyz_comps.append(comp.build_xyz(degree=degree))
        component = Component("xyz", children=xyz_comps)
        return component

    def _make_field_solver(self):
        """
        Make a magnetostatics solver for the field from the PF coils.
        """
        sources = []
        for coil in self._coilset.coils.values():
            sources.append(
                CircularArcCurrentSource(
                    [0, 0, coil.z],
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    coil.dx,
                    coil.dz,
                    coil.x,
                    2 * np.pi,
                    coil.current,
                )
            )

        field_solver = SourceGroup(sources)
        return field_solver


def make_grid(R_0, A, kappa, scale_x=1.6, scale_z=1.7, nx=65, nz=65):
    """
    Make a finite difference Grid for an Equilibrium.

    Parameters
    ----------
    R_0: float
        Major radius
    A: float
        Aspect ratio
    kappa: float
        Elongation
    scale_x: float
        Scaling factor to "grow" the grid away from the plasma in the x direction
    scale_z: float
        Scaling factor to "grow" the grid away from the plasma in the z direction
    nx: int
        Grid discretisation in the x direction
    nz: int
        Grid discretisation in the z direction

    Returns
    -------
    grid: Grid
        Finite difference grid for an Equilibrium
    """
    x_min, x_max = R_0 - scale_x * (R_0 / A), R_0 + scale_x * (R_0 / A)
    z_min, z_max = -scale_z * (kappa * R_0 / A), scale_z * (kappa * R_0 / A)
    return Grid(x_min, x_max, z_min, z_max, nx, nz)


def make_coil_mapper(track, exclusion_zones, coils):
    """
    Break a track down into individual interpolator segments, incorporating exclusion
    zones and mapping them to coils.

    Parameters
    ----------
    track: BluemiraWire
        Full length interpolator track for PF coils
    exclusion_zones: List[BluemiraFace]
        List of exclusion zones
    coils: List[Coil]
        List of coils

    Returns
    -------
    mapper: PositionMapper
        Position mapper for coil position interpolation
    """
    # Break down the track into subsegments
    if exclusion_zones:
        segments = boolean_cut(track, exclusion_zones)
    else:
        segments = [track]

    # Sort the coils into the segments
    coil_bins = [[] for _ in range(len(segments))]
    for i, coil in enumerate(coils):
        distances = [distance_to([coil.x, 0, coil.z], seg)[0] for seg in segments]
        coil_bins[np.argmin(distances)].append(i)

    # Check if multiple coils are on the same segment and split the segments
    new_segments = []
    for segment, bin in zip(segments, coil_bins):
        if len(bin) < 1:
            bluemira_warn("There is a segment of the track which has no coils on it.")
        elif len(bin) == 1:
            new_segments.append(PathInterpolator(segment))
        else:
            coils = [coils[i] for i in bin]
            l_values = [
                segment.parameter_at([c.x, 0, c.z], tolerance=VERY_BIG) for c in coils
            ]
            split_values = l_values[:-1] + 0.5 * np.diff(l_values)

            sub_segs = []
            for i, split in enumerate(split_values):
                sub_seg, segment = split_wire(segment, segment.value_at(alpha=split))
                if sub_seg:
                    sub_segs.append(PathInterpolator(sub_seg))

                if i == len(split_values) - 1:
                    if segment:
                        sub_segs.append(PathInterpolator(segment))

            new_segments.extend(sub_segs)

    return PositionMapper(new_segments)


def make_solenoid(r_cs, tk_cs, z_min, z_max, g_cs, tk_cs_ins, tk_cs_cas, n_CS):
    """
    Make a set of solenoid coils in an EU-DEMO fashion. If n_CS is odd, the central
    module is twice the size of the others. If n_CS is even, all the modules are the
    same size.

    Parameters
    ----------
    r_cs: float
        Radius of the solenoid
    tk_cs: float
        Half-thickness of the solenoid in the radial direction
    z_min: float
        Minimum vertical position of the solenoid
    z_max: float
        Maximum vertical position of the solenoid
    g_cs: float
        Gap between modules
    tk_cs_ins: float
        Insulation thickness around modules
    tk_cs_cas: float
        Casing thickness around modules
    n_CS: int
        Number of modules in the solenoid

    Returns
    -------
    coils: List[Coil]
        List of solenoid coil(s)
    """

    def make_CS_coil(z_coil, dz_coil, i):
        return Coil(
            r_cs,
            z_coil,
            current=0,
            dx=tk_cs,
            dz=dz_coil,
            control=True,
            ctype="CS",
            name=f"CS_{i+1}",
            flag_sizefix=True,
        )

    if z_max < z_min:
        z_min, z_max = z_max, z_min
    if np.isclose(z_max, z_min):
        raise BuilderError(f"Cannot make a solenoid with z_min==z_max=={z_min}")

    total_height = z_max - z_min
    tk_inscas = tk_cs_ins + tk_cs_cas
    total_gaps = (n_CS - 1) * g_cs + n_CS * 2 * tk_inscas
    if total_gaps >= total_height:
        raise BuilderError(
            "Cannot make a solenoid where the gaps and insulation + casing are larger than the height available."
        )

    coils = []
    if n_CS == 1:
        # Single CS module solenoid (no gaps)
        module_height = total_height - 2 * tk_inscas
        coil = make_CS_coil(0.5 * total_height, 0.5 * module_height, 0)
        coils.append(coil)

    elif n_CS % 2 == 0:
        # Equally-spaced CS modules for even numbers of CS coils
        module_height = (total_height - total_gaps) / n_CS
        dz_coil = 0.5 * module_height
        z_iter = z_max
        for i in range(n_CS):
            z_coil = z_iter - tk_inscas - dz_coil
            coil = make_CS_coil(z_coil, dz_coil, i)
            coils.append(coil)
            z_iter = z_coil - dz_coil - tk_inscas - g_cs

    else:
        # Odd numbers of modules -> Make a central module that is twice the size of the
        # others.
        module_height = (total_height - total_gaps) / (n_CS + 1)
        z_iter = z_max
        for i in range(n_CS):
            if i == n_CS // 2:
                # Central module
                dz_coil = module_height
                z_coil = z_iter - tk_inscas - dz_coil

            else:
                # All other modules
                dz_coil = 0.5 * module_height
                z_coil = z_iter - tk_inscas - dz_coil

            coil = make_CS_coil(z_coil, dz_coil, i)
            coils.append(coil)
            z_iter = z_coil - dz_coil - tk_inscas - g_cs

    return coils


def make_PF_coil_positions(tf_boundary, n_PF, R_0, kappa, delta):
    """
    Make a set of PF coil positions crudely with respect to the intended plasma
    shape.
    """
    # Project plasma centroid through plasma upper and lower extrema
    angle_upper = np.arctan2(kappa, -delta)
    angle_lower = np.arctan2(-kappa, -delta)
    scale = 1.1

    angles = np.linspace(scale * angle_upper, scale * angle_lower, n_PF)

    x_c, z_c = np.zeros(n_PF), np.zeros(n_PF)
    for i, angle in enumerate(angles):
        line = make_polygon(
            [
                [R_0, R_0 + VERY_BIG * np.cos(angle)],
                [0, 0],
                [0, VERY_BIG * np.sin(angle)],
            ]
        )
        _, intersection = distance_to(tf_boundary, line)
        x_c[i], _, z_c[i] = intersection[0][0]
    return x_c, z_c


def make_coilset(
    tf_boundary,
    R_0,
    kappa,
    delta,
    r_cs,
    tk_cs,
    g_cs,
    tk_cs_ins,
    tk_cs_cas,
    n_CS,
    n_PF,
    CS_jmax,
    CS_bmax,
    PF_jmax,
    PF_bmax,
):
    """
    Make an initial EU-DEMO-like coilset.
    """
    bb = tf_boundary.bounding_box
    z_min = bb.z_min
    z_max = bb.z_max
    solenoid = make_solenoid(r_cs, tk_cs, z_min, z_max, g_cs, tk_cs_ins, tk_cs_cas, n_CS)

    tf_track = offset_wire(tf_boundary, 1, join="arc")
    x_c, z_c = make_PF_coil_positions(
        tf_track,
        n_PF,
        R_0,
        kappa,
        delta,
    )
    pf_coils = []
    for i, (x, z) in enumerate(zip(x_c, z_c)):
        coil = Coil(
            x,
            z,
            current=0,
            ctype="PF",
            control=True,
            name=f"PF_{i+1}",
            flag_sizefix=False,
            j_max=PF_jmax,
            b_max=PF_bmax,
        )
        pf_coils.append(coil)
    coilset = CoilSet(pf_coils + solenoid)
    coilset.assign_coil_materials("PF", j_max=PF_jmax, b_max=PF_bmax)
    coilset.assign_coil_materials("CS", j_max=CS_jmax, b_max=CS_bmax)
    return coilset


def make_pf_coil_path(tf_boundary: BluemiraWire, offset_value: float) -> BluemiraWire:
    """
    Make an open wire along which the PF coils can move.

    Parameters
    ----------
    tf_boundary: BluemiraWire
        Outside edge of the TF coil in the x-z plane
    offset_value: float
        Offset value from the TF coil edge

    Returns
    -------
    pf_coil_path: BluemiraWire
        Path along which the PF coil centroids should be positioned
    """
    tf_offset = offset_wire(tf_boundary, offset_value)

    # Find top-left and bottom-left "corners"
    coordinates = tf_offset.discretize(byedges=True, ndiscr=200)
    x_min = np.min(coordinates.x)
    z_min, z_max = 0.0, 0.0
    eps = 0.0
    while np.isclose(z_min, z_max):
        # This is unlikely, but if so, shifting x_min a little ensures the boolean cut
        # can be performed and that an open wire will be returned
        idx_inner = np.where(np.isclose(coordinates.x, x_min))[0]
        z_min = np.min(coordinates.z[idx_inner])
        z_max = np.max(coordinates.z[idx_inner])
        x_min += eps
        eps += 1e-3

    cutter = BluemiraFace(
        make_polygon(
            {"x": [0, x_min, x_min, 0], "z": [z_min, z_min, z_max, z_max]}, closed=True
        )
    )

    result = boolean_cut(tf_offset, cutter)
    if len(result) > 1:
        bluemira_warn(
            "Boolean cut of the TF boundary resulted in more than one wire.. returning the longest one. Fingers crossed."
        )
        result.sort(key=lambda wire: -wire.length)
    pf_coil_path = result[0]
    return pf_coil_path
