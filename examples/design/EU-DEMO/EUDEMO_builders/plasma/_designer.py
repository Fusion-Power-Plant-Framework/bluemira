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
A builder for Plasma properties and geometry.
"""

from typing import Dict, Union

import numpy as np
from EUDEMO_builders.equilibria import EUDEMOSingleNullConstraints
from EUDEMO_builders.equilibrium import EquilibriumParams, make_equilibrium

from bluemira.base.builder import Designer
from bluemira.base.parameter_frame import NewParameter as Parameter
from bluemira.base.parameter_frame import NewParameterFrame as ParameterFrame
from bluemira.base.parameter_frame import parameter_frame
from bluemira.equilibria import Equilibrium
from bluemira.equilibria.opt_problems import (
    CoilsetOptimisationProblem,
    UnconstrainedTikhonovCurrentGradientCOP,
)
from bluemira.equilibria.solve import DudsonConvergence, PicardIterator
from bluemira.geometry.parameterisations import PrincetonD
from bluemira.geometry.tools import make_polygon, offset_wire
from bluemira.geometry.wire import BluemiraWire


@parameter_frame
class UnconstrainedTikhonovDesignerParams(ParameterFrame):
    """ """

    A: Parameter[float]
    B_0: Parameter[float]
    beta_p: Parameter[float]
    CS_bmax: Parameter[float]
    CS_jmax: Parameter[float]
    delta_95: Parameter[float]
    div_L2D_ib: Parameter[float]
    div_L2D_ob: Parameter[float]
    g_cs_mod: Parameter[float]
    I_p: Parameter[float]
    kappa_95: Parameter[float]
    n_CS: Parameter[float]
    n_PF: Parameter[float]
    PF_bmax: Parameter[float]
    PF_jmax: Parameter[float]
    R_0: Parameter[float]
    r_cs_in: Parameter[float]
    r_tf_in_centre: Parameter[float]
    r_tf_out_centre: Parameter[float]
    tk_cs_casing: Parameter[float]
    tk_cs_insulation: Parameter[float]
    tk_cs: Parameter[float]


class UnconstrainedTikhonovDesigner(Designer[Equilibrium]):
    def __init__(
        self,
        params: Union[UnconstrainedTikhonovDesignerParams, Dict],
        plot_optimisation: bool = False,
    ):
        self.params: UnconstrainedTikhonovDesignerParams = self._init_params(params)
        self._plot_optimisation = plot_optimisation

    def run(self):
        eq = self._make_equilibrium()
        opt_problem = self._make_opt_problem(eq)
        self._run_picard_iter(eq, opt_problem, self._plot_optimisation)
        return eq

    def _run_picard_iter(
        self,
        eq: Equilibrium,
        opt_problem: CoilsetOptimisationProblem,
        plot: bool = False,
    ):
        """Run a PicardIterator to optimise the given `Equilibrium`."""
        iterator_program = PicardIterator(
            eq,
            opt_problem,
            convergence=DudsonConvergence(),
            relaxation=0.2,
            fixed_coils=True,
            plot=plot,
        )
        iterator_program()

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

    def _make_equilibrium(self) -> Equilibrium:
        """
        Make a reference MHD equilibrium for the plasma.
        """
        return make_equilibrium(
            EquilibriumParams.from_frame(self.params, allow_unknown=True),
            _make_tf_boundary(
                self.params.r_tf_in_centre.value,
                self.params.r_tf_out_centre.value,
                self.params.delta_95.value,
            ),
        )


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
    xmin = np.min(x)
    zmin, zmax = np.min(z), np.max(z)
    xx = np.array(xmin)
    xx = np.append(xx, x[amin:amax])
    xx = np.append(xx, xmin)
    zz = np.array(zmin)
    zz = np.append(zz, z[amin:amax])
    zz = np.append(zz, zmax)
    return xx, zz
