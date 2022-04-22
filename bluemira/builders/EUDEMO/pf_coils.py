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
from bluemira.equilibria.physics import calc_psib
from bluemira.geometry.constants import VERY_BIG
from bluemira.geometry.tools import (
    boolean_cut,
    distance_to,
    make_polygon,
    offset_wire,
    split_wire,
)
from bluemira.magnetostatics.baseclass import SourceGroup
from bluemira.magnetostatics.circular_arc import CircularArcCurrentSource
from bluemira.utilities.opt_problems import OptimisationConstraint
from bluemira.utilities.optimiser import Optimiser
from bluemira.utilities.positioning import PathInterpolator, PositionMapper
from bluemira.utilities.tools import get_class_from_module


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
    ]
    _required_config: List[str] = []
    _params: Configuration
    _param_class: Type[CoilSet]
    _default_runmode: str = "read"
    _design_problem: None

    def _extract_config(self, build_config: BuildConfig):
        super()._extract_config(build_config)

        if self._runmode.name.lower() == "read":
            if build_config.get("eqdsk_path") is None:
                raise BuilderError(
                    "Must supply eqdsk_path in build_config when using 'read' mode."
                )
            self._eqdsk_path = build_config["eqdsk_path"]

        if self._runmode.name.lower() == "run":
            self._problem_settings = build_config.get("problem_settings", {})
            self._algorithm_name = build_config.get("algorithm_name", "SLSQP")
            self._opt_conditions = build_config.get("opt_conditions", {"max_eval": 100})
            self._opt_parameters = build_config.get("opt_parameters", {})

            problem_class = build_config["problem_class"]
            if isinstance(problem_class, str):
                self._problem_class = get_class_from_module(
                    problem_class, default_module="bluemira.equilibria.opt_problems"
                )
            elif isinstance(problem_class, type):
                self._problem_class = problem_class
            else:
                raise BuilderError(
                    "problem_class must either be a str pointing to the class to be loaded "
                    f"or the class itself - got {problem_class}."
                )

    def reinitialise(self, params, **kwargs) -> None:
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

    def run(self, *args):
        """
        Build PF coils from a design optimisation problem.
        """
        self._design_problem = self._problem_class()

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
    if z_max == z_min:
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


from bluemira.equilibria.equilibrium import Breakdown, Equilibrium
from bluemira.equilibria.grid import Grid
from bluemira.equilibria.opt_constraints import (
    coil_field_constraints,
    coil_force_constraints,
)
from bluemira.equilibria.opt_problems import (
    InboardBreakdownZoneStrategy,
    OutboardBreakdownZoneStrategy,
    PremagnetisationCOP,
)
from bluemira.equilibria.profiles import CustomProfile


class PFSystemDesignProcedure:
    def __init__(self, params, build_config, tf_boundary, p_prime, ff_prime):
        self.params = params
        self.build_config = build_config
        self.tf_boundary = tf_boundary
        # self.profiles = CustomProfile(
        #     p_prime, ff_prime, params.R_0.value, params.B_0.value, Ip=params.I_p.value
        # )
        self.coilset = self._make_initial_coilset()

    def _make_initial_coilset(self):
        tk_cs = 0.5 * self.params.tk_cs.value
        r_cs = self.params.r_cs_in.value + tk_cs
        coilset = make_coilset(
            self.tf_boundary,
            self.params.R_0.value,
            self.params.kappa.value,
            self.params.delta.value,
            r_cs=r_cs,
            tk_cs=tk_cs,
            g_cs=self.params.g_cs_mod.value,
            n_CS=self.params.n_CS.value,
            n_PF=self.params.n_PF.value,
            tk_cs_ins=self.params.tk_cs_insulation.value,
            tk_cs_cas=self.params.tk_cs_casing.value,
            PF_jmax=self.params.PF_jmax.value,
            PF_bmax=self.params.PF_bmax.value,
            CS_jmax=self.params.CS_jmax.value,
            CS_bmax=self.params.CS_bmax.value,
        )
        return coilset

    def run_premagnetisation(self):
        R_0 = self.params.R_0.value
        strategy = OutboardBreakdownZoneStrategy(
            R_0, self.params.A.value, self.params.tk_sol_ib.value
        )
        # Not really important; mostly for plotting
        grid = Grid(0.1, R_0 * 2, -1.5 * R_0, 1.5 * R_0, 100, 100)
        optimiser = Optimiser(
            "COBYLA",
            opt_conditions={"max_eval": 5000, "ftol_rel": 1e-10},
        )
        scale = 1e6
        # self.coilset.set_control_currents(self.coilset.get_max_currents(scale*self.params.I_p))
        self.coilset.mesh_coils(0.1)

        breakdown = Breakdown(self.coilset, grid, R_0=R_0)
        constraints = [
            OptimisationConstraint(
                coil_field_constraints,
                f_constraint_args={
                    "eq": breakdown,
                    "B_max": self.coilset.get_max_fields(),
                    "scale": scale,
                },
                tolerance=1e-6 * np.ones(self.coilset.n_control),
            ),
            # OptimisationConstraint(
            #     coil_force_constraints,
            #     f_constraint_args={
            #         "eq": breakdown,
            #         "n_PF": self.coilset.n_PF,
            #         "n_CS": self.coilset.n_CS,
            #         "PF_Fz_max": self.params.F_pf_zmax.value,
            #         "CS_Fz_sum_max": self.params.F_cs_ztotmax.value,
            #         "CS_Fz_sep_max": self.params.F_cs_sepmax.value,
            #         "scale": scale,
            #     },
            #     tolerance=1e-6 * np.ones(self.coilset.n_control),
            # ),
        ]
        max_currents = self.coilset.get_max_currents(1.0 * scale * self.params.I_p.value)
        # TODO: Still the problem that the response matrices should in theory be
        # changed when the PF currents (not size-fixed) change.
        self.problem = PremagnetisationCOP(
            self.coilset,
            strategy,
            B_stray_max=self.params.B_premag_stray_max.value,
            B_stray_con_tol=1e-8,
            n_B_stray_points=20,
            optimiser=optimiser,
            max_currents=max_currents,
            constraints=constraints,
        )
        self.coilset = self.problem.optimise(max_currents / scale)
        breakdown = Breakdown(self.coilset, grid, R_0=R_0)
        breakdown.set_breakdown_point(*strategy.breakdown_point)
        psi_premag = breakdown.breakdown_psi
        bluemira_print(f"Premagnetisation flux = {2*np.pi * psi_premag:.2f} V.s")
        return breakdown

    def calculate_sof_eof_fluxes(self, psi_premag: float):
        """
        Calculate the SOF and EOF plasma boundary fluxes.
        """
        psi_sof = calc_psib(
            2 * np.pi * psi_premag,
            self.params.R_0.value,
            1e6 * self.params.I_p.value,
            self.params.l_i.value,
            self.params.C_Ejima.value,
        )
        psi_eof = psi_sof - self.params.tau_flattop.value * self.params.v_burn.value
        return psi_sof, psi_eof

    def optimise_positions(self):

        eq = Equilibrium(
            self.coilset,
            force_symmetry=False,
            vcontrol=None,
            limiter=None,
            profiles=self.profiles,
        )
        pass

    def consolidate_coilset(self):
        pass

    def consolidate_premagnetisation(self):
        pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    from bluemira.base.config import Configuration
    from bluemira.geometry.parameterisations import PrincetonD

    params = Configuration()

    design = PFSystemDesignProcedure(
        params,
        {},
        PrincetonD({"x1": {"value": 4}, "x2": {"value": 16}}).create_shape(),
        None,
        None,
    )

    breakdown = design.run_premagnetisation()
    psi_sof, psi_eof = design.calculate_sof_eof_fluxes(breakdown.breakdown_psi)

    f, ax = plt.subplots()
    breakdown.coilset.plot(ax=ax)
    breakdown.plot(ax=ax)

    positioner = make_coil_mapper(
        PrincetonD({"x1": {"value": 4}, "x2": {"value": 16}}).create_shape(),
        None,
        coils=breakdown.coilset.coils.values(),
    )
