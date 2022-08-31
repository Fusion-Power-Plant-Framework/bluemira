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

import enum
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from EUDEMO_builders.equilibria.tools import (
    EquilibriumParams,
    EUDEMOSingleNullConstraints,
    make_equilibrium,
)

from bluemira.base.parameter_frame import NewParameter as Parameter
from bluemira.base.parameter_frame import NewParameterFrame as ParameterFrame
from bluemira.base.parameter_frame import make_parameter_frame
from bluemira.base.solver import RunMode, SolverABC, Task
from bluemira.equilibria import Equilibrium
from bluemira.equilibria.opt_problems import (
    CoilsetOptimisationProblem,
    UnconstrainedTikhonovCurrentGradientCOP,
)
from bluemira.equilibria.solve import DudsonConvergence, PicardIterator
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.geometry.tools import make_polygon, offset_wire
from bluemira.geometry.wire import BluemiraWire


@dataclass
class UnconstrainedTikhonovSolverParams(ParameterFrame):
    """Parameters for running the `UnconstrainedTikhonovSolver`."""

    A: Parameter[float]
    B_0: Parameter[float]
    beta_p: Parameter[float]
    CS_bmax: Parameter[float]
    CS_jmax: Parameter[float]
    delta: Parameter[float]
    delta_95: Parameter[float]
    div_L2D_ib: Parameter[float]
    div_L2D_ob: Parameter[float]
    g_cs_mod: Parameter[float]
    I_p: Parameter[float]
    kappa: Parameter[float]
    kappa_95: Parameter[float]
    l_i: Parameter[float]
    n_CS: Parameter[int]
    n_PF: Parameter[int]
    PF_bmax: Parameter[float]
    PF_jmax: Parameter[float]
    q_95: Parameter[float]
    R_0: Parameter[float]
    r_cs_in: Parameter[float]
    r_tf_in_centre: Parameter[float]
    r_tf_out_centre: Parameter[float]
    shaf_shift: Parameter[float]
    tk_cs_casing: Parameter[float]
    tk_cs_insulation: Parameter[float]
    tk_cs: Parameter[float]


class UnconstrainedTikhonovRunMode(RunMode):
    """Run modes for `UnconstrainedTikhonovSolver`."""

    RUN = enum.auto()
    MOCK = enum.auto()
    READ = enum.auto()


class _Setup(Task):
    def run(self):
        eq = self._make_equilibrium()
        return eq, self._make_opt_problem(eq)

    def mock(self):
        return self._make_equilibrium()

    def _make_equilibrium(self) -> Equilibrium:
        """
        Make a reference MHD equilibrium for the plasma.
        """
        return make_equilibrium(
            EquilibriumParams.from_frame(self.params),
            _make_tf_boundary(
                self.params.r_tf_in_centre.value,
                self.params.r_tf_out_centre.value,
                self.params.delta_95.value,
            ),
        )

    def _make_opt_problem(self, eq: Equilibrium):
        """
        Create the `UnconstrainedTikhonovCurrentGradientCOP` optimisation problem.
        """
        kappa = 1.12 * self.params.kappa_95.value
        kappa_ul_tweak = 0.05
        kappa_u = (1 - kappa_ul_tweak) * kappa
        kappa_l = (1 + kappa_ul_tweak) * kappa

        eq_targets = EUDEMOSingleNullConstraints(
            R_0=self.params.R_0.value,
            Z_0=0.0,
            A=self.params.A.value,
            kappa_u=kappa_u,
            kappa_l=kappa_l,
            delta_u=self.params.delta_95.value,
            delta_l=self.params.delta_95.value,
            psi_u_neg=0.0,
            psi_u_pos=0.0,
            psi_l_neg=60.0,
            psi_l_pos=30.0,
            div_l_ib=self.params.div_L2D_ib.value,
            div_l_ob=self.params.div_L2D_ob.value,
            psibval=0.0,
            psibtol=1.0e-3,
            lower=True,
            n=100,
        )
        return UnconstrainedTikhonovCurrentGradientCOP(
            eq.coilset, eq, eq_targets, gamma=1e-8
        )


class _Run(Task):
    def __init__(
        self,
        params: ParameterFrame,
        file_path: Optional[str] = None,
        plot_optimisation: bool = False,
    ):
        super().__init__(params)
        self.file_path = file_path
        self.plot_optimisation = plot_optimisation

    def run(
        self, eq_and_opt_problem: Tuple[Equilibrium, CoilsetOptimisationProblem]
    ) -> Equilibrium:
        """Run the optimisation problem for the solver."""
        eq, opt_problem = eq_and_opt_problem
        iterator_program = PicardIterator(
            eq,
            opt_problem,
            convergence=DudsonConvergence(),
            relaxation=0.2,
            fixed_coils=True,
            plot=self.plot_optimisation,
        )
        iterator_program()
        return eq

    def read(self, _) -> Equilibrium:
        """Load an equilibrium from a file."""
        if not self.file_path:
            raise ValueError(
                f"Cannot execute {type(self).__name__} in READ mode: no file path specified."
            )
        return Equilibrium.from_eqdsk(self.file_path)


class _Teardown(Task):
    def run(self, eq: Equilibrium):
        """Update the solver's parameters from the equilibrium."""
        new_param_values = self._analyse_equilibrium(eq)
        self.params.update_values(new_param_values, source="UnconstrainedTikhonovSolver")
        return eq

    def _analyse_equilibrium(self, eq: Equilibrium):
        """
        Analyse an equilibrium and return important parameter values.

        Parameters
        ----------
        eq: Equilibrium
            The equilibrium to analyse for use with this builder.
        """
        plasma_dict = eq.analyse_plasma()
        return {
            "beta_p": plasma_dict["beta_p"],
            "delta_95": plasma_dict["delta_95"],
            "delta": plasma_dict["delta"],
            "I_p": plasma_dict["Ip"] / 1e6,
            "kappa_95": plasma_dict["kappa_95"],
            "kappa": plasma_dict["kappa"],
            "l_i": plasma_dict["li"],
            "q_95": plasma_dict["q_95"],
            "shaf_shift": np.hypot(plasma_dict["dx_shaf"], plasma_dict["dz_shaf"]),
        }


class UnconstrainedTikhonovSolver(SolverABC):
    """
    Solves an unconstrained Tikhnov current gradient coil-set
    optimisation problem, outputting an `Equilibrium`.

    Parameters
    ----------
    params: Union[Dict, NewParameterFrame]
        The parameters for the solver, the dictionary or frame must
        contain all the parameters present in
        `UnconstrainedTikhonovSolverParams`.
    build_config: Optional[Dict]
        The config for the solver. Optional keys:
        - `read_file_path`: str
            the path to an eqdsk file to read the equilibrium from,
            required in `READ` mode.
        - `plot_optimisation`: bool
            set to `True` to plot the iterations in the optimisation.
    """

    setup_cls = _Setup
    run_cls = _Run
    teardown_cls = _Teardown
    run_mode_cls = UnconstrainedTikhonovRunMode

    def __init__(self, params: ParameterFrame, build_config: Dict = None):
        self.params = make_parameter_frame(params, UnconstrainedTikhonovSolverParams)
        self.build_config = {} if build_config is None else build_config
        read_file_path = self.build_config.get("file_path", None)
        plot_optimisation = self.build_config.get("plot_optimisation", False)

        self._setup = self.setup_cls(self.params)
        self._run = self.run_cls(self.params, read_file_path, plot_optimisation)
        self._teardown = self.teardown_cls(self.params)


def _make_tf_boundary(
    r_tf_in_centre: float, r_tf_out_centre: float, delta_95: float
) -> BluemiraWire:
    """
    Make an initial TF coil shape to guide an equilibrium calculation.
    """
    rin, rout = r_tf_in_centre, r_tf_out_centre
    # TODO: Handle other TF coil parameterisations?
    shape = PrincetonD({"x1": {"value": rin}, "x2": {"value": rout}, "dz": {"value": 0}})
    tf_boundary = shape.create_shape()
    if delta_95 < 0:  # Negative triangularity
        tf_boundary.rotate(tf_boundary.center_of_mass, direction=(0, 1, 0), degree=180)
    tf_boundary = offset_wire(tf_boundary, -0.5)
    x, z = _flatten_shape(*tf_boundary.discretize(200, byedges=True).xz)
    return make_polygon({"x": x, "z": z})


def _flatten_shape(x, z):
    """
    Flattens a shape by dragging the lowest and highest point to the minimum
    radius point.
    """
    amin, amax = np.argmin(z), np.argmax(z)
    num_elements = amax - amin + 2

    xx = np.empty(num_elements)
    xx[0] = np.min(x)
    xx[1:-1] = x[amin:amax]
    xx[-1] = xx[0]

    zmin, zmax = z[amin], z[amax]
    zz = np.empty(num_elements)
    zz[0] = zmin
    zz[1:-1] = z[amin:amax]
    zz[-1] = zmax

    return xx, zz
